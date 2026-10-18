package `in`.iisc.vaani.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import `in`.iisc.vaani.dsp.NeonAudioProcessor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.exp
import kotlin.math.min

/** Language the decoder is prompted with: step-0 auto-detection or a manual override. */
enum class LanguageMode(val label: String) {
    AUTO("Auto EN/HI"), ENGLISH("English"), HINGLISH("Hinglish");
}

/** Result of a single on-device transcription, with a per-stage latency breakdown. */
data class SttResult(
    val text: String,              // raw Whisper text (pre-codec)
    val language: String,          // language token the text was decoded with ("en" / "hi")
    val detectedLanguage: String,  // step-0 2-way LID verdict
    val pHindi: Float,             // P(hi) from the step-0 <|en|> vs <|hi|> logits
    val tokenIds: List<Int>,
    val sttLatencyMs: Long,        // mel + encoder(s) + LID + decode
    val sttRtf: Double,
    val melMs: Long,
    val encoderMs: Long,
    val lidMs: Long,
    val decodeMs: Long,
    val tokensPerSecond: Double,
    val audioSec: Double,
    val model: String = "",        // label of the checkpoint that produced [text]
)

/** One Whisper checkpoint on disk: encoder, step-0 decoder and (optional) KV-cached decoder. */
data class WhisperFiles(val encoder: String, val decoder: String, val decoderWithPast: String?, val label: String)

/**
 * First-principles Whisper-style STT engine on ONNX Runtime with a KV-cached decoder.
 *
 * Pipeline:
 *   PCM --[NEON log-Mel]--> [1,80,3000] --> encoder --> last_hidden_state [1,1500,d]
 *     --> decoder([<|startoftranscript|>], hidden)   (step 0: computes cross-attention K/V
 *         once AND yields the 2-way language-ID logits <|en|> vs <|hi|> for free)
 *     --> decoder_with_past(<|lang|>), (<|transcribe|>), (<|notimestamps|>), then one
 *         token per step, reusing the encoder K/V and the growing self-attention cache
 *     --> greedy argmax with logit-level script masking --> byte-level BPE --> text
 *
 * Two checkpoints share that pipeline, the Mel features and the tokenizer:
 *  - [primary] (multilingual Whisper-tiny) does the step-0 language ID and all English;
 *  - [hindi], an optional Hindi-trained specialist, takes every Hindi turn: after an
 *    `hi` LID verdict the same Mel tensor is re-encoded by the specialist and decoded with
 *    `<|hi|>` (multilingual Whisper-tiny mostly translates or garbles Hindi). In the manual
 *    Hinglish mode the multilingual pass is skipped entirely.
 *
 * Latency engineering:
 *  - the encoder input is built from a reused *direct* buffer (zero JNI copy);
 *  - `last_hidden_state` and the step-0 `present.*.encoder.*` tensors are produced once
 *    and passed by reference to every decoder call — no per-step 2.3 MB re-copy, and the
 *    1500 x d cross-attention projections are never recomputed;
 *  - sessions use ALL_OPT, 4 intra-op threads and `session.intra_op.allow_spinning=1`.
 *
 * Logit-level script masking sets every suppressed-script, Whisper non-speech and
 * special/timestamp token to -inf before the argmax, so Hinglish speech can never drift
 * into Urdu/Perso-Arabic, Cyrillic or CJK output or emit `[Music]`-style annotations.
 * If a checkpoint has no `decoderWithPast` the engine falls back to the cache-free loop.
 */
class VaaniSttEngine(
    primary: WhisperFiles,
    private val vocab: VaaniVocab,
    private val melFilters: FloatArray,
    hindi: WhisperFiles? = null,
    threads: Int = 4,
    private val maxDecodeTokens: Int = 224,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    /** One loaded checkpoint (sessions + input names). */
    private inner class WhisperModel(files: WhisperFiles, opts: OrtSession.SessionOptions) : AutoCloseable {
        val label = files.label
        val encoder: OrtSession = env.createSession(files.encoder, opts)
        val decoder: OrtSession = env.createSession(files.decoder, opts)
        val decoderPast: OrtSession? = files.decoderWithPast?.let { env.createSession(it, opts) }
        val encInputName: String = encoder.inputNames.first()
        val decIdsName: String = decoder.inputNames.firstOrNull { it.contains("input_ids") } ?: decoder.inputNames.first()
        val decEncName: String = decoder.inputNames.firstOrNull { it.contains("encoder_hidden_states") }
            ?: decoder.inputNames.first { it != decIdsName }

        override fun close() {
            runCatching { encoder.close() }
            runCatching { decoder.close() }
            runCatching { decoderPast?.close() }
        }
    }

    private val primary: WhisperModel
    private val hindi: WhisperModel?

    // Zero-copy input buffers, reused for every utterance / step.
    private val melBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(NeonAudioProcessor.N_MELS * NeonAudioProcessor.TARGET_FRAMES * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val idBuffer: LongBuffer = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder()).asLongBuffer()

    init {
        val opts = sessionOptions(threads)
        this.primary = WhisperModel(primary, opts)
        this.hindi = hindi?.let { WhisperModel(it, opts) }
    }

    val usesKvCache: Boolean get() = primary.decoderPast != null

    /** Label of the Hindi specialist, or null when only the multilingual checkpoint is loaded. */
    val hindiModelLabel: String? get() = hindi?.label

    /** Transcribes 16 kHz mono float PCM and reports latency, RTF and the LID verdict. */
    @Synchronized
    fun transcribe(pcm: FloatArray, mode: LanguageMode = LanguageMode.AUTO): SttResult {
        val t0 = System.nanoTime()
        val audioSec = pcm.size.toDouble() / NeonAudioProcessor.SAMPLE_RATE

        // 1) NEON log-Mel feature [1, 80, 3000] into the reused direct buffer (shared by both checkpoints).
        val mel = NeonAudioProcessor.logMel(pcm, melFilters)
        melBuffer.clear(); melBuffer.put(mel); melBuffer.flip()
        val tMel = System.nanoTime()

        val encInput = OnnxTensor.createTensor(
            env, melBuffer,
            longArrayOf(1, NeonAudioProcessor.N_MELS.toLong(), NeonAudioProcessor.TARGET_FRAMES.toLong()),
        )
        try {
            val specialist = hindi
            if (mode == LanguageMode.HINGLISH && specialist != null) {
                val p = decodeWith(specialist, encInput, LanguageMode.HINGLISH, audioSec, handOffHindi = false)
                return result(p, t0, tMel, audioSec, detected = "hi", pHi = 1f, lidNs = 0L,
                    encoderNs = p.encoderNs, decodeNs = p.step0Ns + p.decodeNs)
            }
            val p = decodeWith(primary, encInput, mode, audioSec, handOffHindi = specialist != null)
            if (p.handedOff && specialist != null) {
                val h = decodeWith(specialist, encInput, LanguageMode.HINGLISH, audioSec, handOffHindi = false)
                return result(h, t0, tMel, audioSec, detected = p.detected, pHi = p.pHindi, lidNs = p.step0Ns,
                    encoderNs = p.encoderNs + h.encoderNs, decodeNs = h.step0Ns + h.decodeNs)
            }
            return result(p, t0, tMel, audioSec, detected = p.detected, pHi = p.pHindi, lidNs = p.step0Ns,
                encoderNs = p.encoderNs, decodeNs = p.decodeNs)
        } finally {
            encInput.close()
        }
    }

    private class Pass(
        val tokens: List<Int>, val language: String, val detected: String, val pHindi: Float,
        val encoderNs: Long, val step0Ns: Long, val decodeNs: Long, val handedOff: Boolean, val model: String,
    )

    private fun result(
        p: Pass, t0: Long, tMel: Long, audioSec: Double, detected: String, pHi: Float,
        lidNs: Long, encoderNs: Long, decodeNs: Long,
    ): SttResult {
        val totalMs = (System.nanoTime() - t0) / 1_000_000
        val decodeMs = decodeNs / 1_000_000
        return SttResult(
            text = vocab.decode(p.tokens),
            language = p.language,
            detectedLanguage = detected,
            pHindi = pHi,
            tokenIds = p.tokens,
            sttLatencyMs = totalMs,
            sttRtf = if (audioSec > 0) (totalMs / 1000.0) / audioSec else 0.0,
            melMs = (tMel - t0) / 1_000_000,
            encoderMs = encoderNs / 1_000_000,
            lidMs = lidNs / 1_000_000,
            decodeMs = decodeMs,
            tokensPerSecond = if (decodeMs > 0) p.tokens.size * 1000.0 / decodeMs else 0.0,
            audioSec = audioSec,
            model = p.model,
        )
    }

    /**
     * Encoder + step 0 on [m]. With [handOffHindi], a resolved `hi` stops after the LID so the
     * specialist can transcribe; otherwise the whole utterance is decoded here.
     */
    private fun decodeWith(m: WhisperModel, input: OnnxTensor, mode: LanguageMode, audioSec: Double, handOffHindi: Boolean): Pass {
        val tE0 = System.nanoTime()
        val encResult = m.encoder.run(mapOf(m.encInputName to input))
        try {
            // Created ONCE by the encoder; every decoder step below receives this exact tensor.
            val hidden = encResult.get(0) as OnnxTensor
            val tE1 = System.nanoTime()
            val sotIds = idsTensor(vocab.sot)
            val step0 = try { m.decoder.run(mapOf(m.decIdsName to sotIds, m.decEncName to hidden)) } finally { sotIds.close() }
            try {
                val (detected, pHi) = lid(logitsOf(step0), 0)
                val tLid = System.nanoTime()
                val language = resolve(mode, detected)
                if (handOffHindi && language == "hi") {
                    return Pass(emptyList(), language, detected, pHi, tE1 - tE0, tLid - tE1, 0L, handedOff = true, m.label)
                }
                val tokens = if (m.decoderPast != null) decodeKv(m, step0, language, audioSec)
                else decodeCacheFree(m, hidden, language, audioSec)
                return Pass(tokens, language, detected, pHi, tE1 - tE0, tLid - tE1, System.nanoTime() - tLid,
                    handedOff = false, m.label)
            } finally {
                step0.close()
            }
        } finally {
            encResult.close()
        }
    }

    private fun idsTensor(vararg ids: Int): OnnxTensor {
        if (ids.size == 1) {
            idBuffer.clear(); idBuffer.put(ids[0].toLong()); idBuffer.flip()
            return OnnxTensor.createTensor(env, idBuffer, longArrayOf(1, 1))
        }
        return OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { ids[it].toLong() }),
            longArrayOf(1, ids.size.toLong()))
    }

    private fun lid(logits: FloatBuffer, row: Int): Pair<String, Float> {
        val base = row * vocab.vocabSize
        val lEn = logits.get(base + vocab.languageToken("en"))
        val lHi = logits.get(base + vocab.languageToken("hi"))
        val pHi = (1.0 / (1.0 + exp((lEn - lHi).toDouble()))).toFloat()
        return (if (lHi > lEn) "hi" else "en") to pHi
    }

    private fun resolve(mode: LanguageMode, detected: String) = when (mode) {
        LanguageMode.AUTO -> detected
        LanguageMode.ENGLISH -> "en"
        LanguageMode.HINGLISH -> "hi"
    }

    private fun tokenCap(audioSec: Double) = min(maxDecodeTokens, 32 + (audioSec * 20).toInt())

    // ---- KV cache: step 0 already ran (it is the LID pass) ---------------------
    private fun decodeKv(m: WhisperModel, step0: OrtSession.Result, language: String, audioSec: Double): List<Int> {
        val past = m.decoderPast!!
        var prev: OrtSession.Result? = null
        try {
            // Cross-attention K/V: computed once at step 0, reused by reference every step.
            val encKv = HashMap<String, OnnxTensor>()
            var decKv = HashMap<String, OnnxTensor>()
            for ((name, value) in step0) {
                if (!name.startsWith("present.")) continue
                val pastName = name.replace("present.", "past_key_values.")
                if (".encoder." in name) encKv[pastName] = value as OnnxTensor else decKv[pastName] = value as OnnxTensor
            }

            fun step(token: Int): FloatBuffer {
                val ids = idsTensor(token)
                val inputs = HashMap<String, OnnxTensor>(decKv.size + encKv.size + 1)
                inputs.putAll(decKv); inputs.putAll(encKv); inputs["input_ids"] = ids
                val r = try { past.run(inputs) } finally { ids.close() }
                prev?.close()                       // owner of the previous self-attention cache
                prev = r
                val next = HashMap<String, OnnxTensor>()
                for ((name, value) in r) {
                    if (name.startsWith("present.")) next[name.replace("present.", "past_key_values.")] = value as OnnxTensor
                }
                decKv = next
                return logitsOf(r)
            }

            step(vocab.languageToken(language))
            step(vocab.transcribe)
            var logits = step(vocab.noTimestamps)
            val out = ArrayList<Int>()
            val cap = tokenCap(audioSec)
            for (i in 0 until cap) {
                val next = maskedArgmax(logits, 0, if (i == 0) vocab.firstStepMask else vocab.textMask)
                if (next == vocab.eot) break
                out.add(next)
                if (repeatingTail(out)) break
                logits = step(next)
            }
            return out
        } finally {
            prev?.close()
        }
    }

    // ---- Fallback: cache-free loop (re-feeds the whole prefix) -----------------
    private fun decodeCacheFree(m: WhisperModel, hidden: OnnxTensor, language: String, audioSec: Double): List<Int> {
        val tokens = arrayListOf(vocab.sot, vocab.languageToken(language), vocab.transcribe, vocab.noTimestamps)
        val prompt = tokens.size
        for (i in 0 until tokenCap(audioSec)) {
            val ids = idsTensor(*tokens.toIntArray())
            val next = try {
                m.decoder.run(mapOf(m.decIdsName to ids, m.decEncName to hidden)).use { r ->
                    maskedArgmax(logitsOf(r), tokens.size - 1, if (i == 0) vocab.firstStepMask else vocab.textMask)
                }
            } finally { ids.close() }
            if (next == vocab.eot) break
            tokens.add(next)
            if (repeatingTail(tokens.subList(prompt, tokens.size))) break
        }
        return tokens.subList(prompt, tokens.size).toList()
    }

    private fun logitsOf(r: OrtSession.Result): FloatBuffer =
        (r.get("logits").orElse(r.get(0)) as OnnxTensor).floatBuffer

    /** argmax over row [row] of [1, L, V] logits with masked ids forced to -inf. */
    private fun maskedArgmax(logits: FloatBuffer, row: Int, mask: BooleanArray): Int {
        val base = row * vocab.vocabSize
        val n = min(vocab.vocabSize, mask.size)
        var bestId = vocab.eot
        var bestVal = Float.NEGATIVE_INFINITY
        for (id in 0 until n) {
            if (mask[id]) continue
            val v = logits.get(base + id)
            if (v > bestVal) { bestVal = v; bestId = id }
        }
        return bestId
    }

    override fun close() {
        primary.close()
        hindi?.close()
    }

    companion object {
        /** ALL_OPT, [threads] intra-op threads, spinning on (minimal thread-wake latency). */
        fun sessionOptions(threads: Int): OrtSession.SessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(threads)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            addConfigEntry("session.intra_op.allow_spinning", "1")
        }

        /** True if the tail of [tokens] is one short pattern repeated (Whisper loop guard). */
        fun repeatingTail(tokens: List<Int>, maxPeriod: Int = 8, minRepeats: Int = 4): Boolean {
            for (p in 1..maxPeriod) {
                val need = p * minRepeats
                if (tokens.size < need) continue
                val off = tokens.size - need
                var ok = true
                for (i in 0 until need) {
                    if (tokens[off + i] != tokens[off + (i % p)]) { ok = false; break }
                }
                if (ok) return true
            }
            return false
        }
    }
}
