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
    val sttLatencyMs: Long,        // mel + encoder + LID + decode
    val sttRtf: Double,
    val melMs: Long,
    val encoderMs: Long,
    val lidMs: Long,
    val decodeMs: Long,
    val tokensPerSecond: Double,
    val audioSec: Double,
)

/**
 * First-principles Whisper-style STT engine on ONNX Runtime with a KV-cached decoder.
 *
 * Pipeline:
 *   PCM --[NEON log-Mel]--> [1,80,3000] --> encoder --> last_hidden_state [1,1500,384]
 *     --> decoder([<|startoftranscript|>], hidden)   (step 0: computes cross-attention K/V
 *         once AND yields the 2-way language-ID logits <|en|> vs <|hi|> for free)
 *     --> decoder_with_past(<|lang|>), (<|transcribe|>), (<|notimestamps|>), then one
 *         token per step, reusing the encoder K/V and the growing self-attention cache
 *     --> greedy argmax with logit-level script masking --> byte-level BPE --> text
 *
 * Latency engineering:
 *  - the encoder input is built from a reused *direct* buffer (zero JNI copy);
 *  - `last_hidden_state` and the step-0 `present.*.encoder.*` tensors are produced once
 *    and passed by reference to every decoder call — no per-step 2.3 MB re-copy, and the
 *    1500x384 cross-attention projections are never recomputed;
 *  - sessions use ALL_OPT, 4 intra-op threads and `session.intra_op.allow_spinning=1`.
 *
 * Logit-level script masking sets every suppressed-script, Whisper non-speech and
 * special/timestamp token to -inf before the argmax, so Hinglish speech can never drift
 * into Urdu/Perso-Arabic, Cyrillic or CJK output or emit `[Music]`-style annotations.
 * If `decoderWithPastPath` is null the engine falls back to the cache-free loop.
 */
class VaaniSttEngine(
    encoderPath: String,
    decoderPath: String,
    decoderWithPastPath: String?,
    private val vocab: VaaniVocab,
    private val melFilters: FloatArray,
    threads: Int = 4,
    private val maxDecodeTokens: Int = 224,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession
    private val decoderPast: OrtSession?

    private val encInputName: String
    private val decIdsName: String
    private val decEncName: String

    // Zero-copy input buffers, reused for every utterance / step.
    private val melBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(NeonAudioProcessor.N_MELS * NeonAudioProcessor.TARGET_FRAMES * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val idBuffer: LongBuffer = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder()).asLongBuffer()

    init {
        val opts = sessionOptions(threads)
        encoder = env.createSession(encoderPath, opts)
        decoder = env.createSession(decoderPath, opts)
        decoderPast = decoderWithPastPath?.let { env.createSession(it, opts) }

        encInputName = encoder.inputNames.first()
        decIdsName = decoder.inputNames.firstOrNull { it.contains("input_ids") } ?: decoder.inputNames.first()
        decEncName = decoder.inputNames.firstOrNull { it.contains("encoder_hidden_states") }
            ?: decoder.inputNames.first { it != decIdsName }
    }

    val usesKvCache: Boolean get() = decoderPast != null

    /** Transcribes 16 kHz mono float PCM and reports latency, RTF and the LID verdict. */
    @Synchronized
    fun transcribe(pcm: FloatArray, mode: LanguageMode = LanguageMode.AUTO): SttResult {
        val t0 = System.nanoTime()
        val audioSec = pcm.size.toDouble() / NeonAudioProcessor.SAMPLE_RATE

        // 1) NEON log-Mel feature [1, 80, 3000] into the reused direct buffer.
        val mel = NeonAudioProcessor.logMel(pcm, melFilters)
        melBuffer.clear(); melBuffer.put(mel); melBuffer.flip()
        val tMel = System.nanoTime()

        val encInput = OnnxTensor.createTensor(
            env, melBuffer,
            longArrayOf(1, NeonAudioProcessor.N_MELS.toLong(), NeonAudioProcessor.TARGET_FRAMES.toLong()),
        )
        val encResult = encoder.run(mapOf(encInputName to encInput))
        try {
            // Created ONCE by the encoder; every decoder step below receives this exact tensor.
            val hidden = encResult.get(0) as OnnxTensor
            val tEnc = System.nanoTime()
            val dec = if (decoderPast != null) decodeKv(hidden, mode, audioSec) else decodeCacheFree(hidden, mode, audioSec)
            val tEnd = System.nanoTime()

            val text = vocab.decode(dec.tokens)
            val totalMs = (tEnd - t0) / 1_000_000
            val decodeMs = (tEnd - tEnc) / 1_000_000 - dec.lidMs
            return SttResult(
                text = text,
                language = dec.language,
                detectedLanguage = dec.detected,
                pHindi = dec.pHindi,
                tokenIds = dec.tokens,
                sttLatencyMs = totalMs,
                sttRtf = if (audioSec > 0) (totalMs / 1000.0) / audioSec else 0.0,
                melMs = (tMel - t0) / 1_000_000,
                encoderMs = (tEnc - tMel) / 1_000_000,
                lidMs = dec.lidMs,
                decodeMs = decodeMs,
                tokensPerSecond = if (decodeMs > 0) dec.tokens.size * 1000.0 / decodeMs else 0.0,
                audioSec = audioSec,
            )
        } finally {
            encResult.close()
            encInput.close()
        }
    }

    private class Decoded(
        val tokens: List<Int>, val language: String, val detected: String, val pHindi: Float, val lidMs: Long,
    )

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

    // ---- Session 2: step-0 LID + KV cache ------------------------------------
    private fun decodeKv(hidden: OnnxTensor, mode: LanguageMode, audioSec: Double): Decoded {
        val past = decoderPast!!
        val tLid0 = System.nanoTime()
        val sotIds = idsTensor(vocab.sot)
        val step0 = decoder.run(mapOf(decIdsName to sotIds, decEncName to hidden))
        sotIds.close()
        var prev: OrtSession.Result? = null
        try {
            val (detected, pHi) = lid(logitsOf(step0), 0)
            val lidMs = (System.nanoTime() - tLid0) / 1_000_000
            val language = resolve(mode, detected)

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
            return Decoded(out, language, detected, pHi, lidMs)
        } finally {
            prev?.close()
            step0.close()
        }
    }

    // ---- Fallback: Session-1 cache-free loop (re-feeds the whole prefix) -----
    private fun decodeCacheFree(hidden: OnnxTensor, mode: LanguageMode, audioSec: Double): Decoded {
        val tLid0 = System.nanoTime()
        val sotIds = idsTensor(vocab.sot)
        val (detected, pHi) = decoder.run(mapOf(decIdsName to sotIds, decEncName to hidden)).use { lid(logitsOf(it), 0) }
        sotIds.close()
        val lidMs = (System.nanoTime() - tLid0) / 1_000_000
        val language = resolve(mode, detected)
        val tokens = arrayListOf(vocab.sot, vocab.languageToken(language), vocab.transcribe, vocab.noTimestamps)
        val prompt = tokens.size
        for (i in 0 until tokenCap(audioSec)) {
            val ids = idsTensor(*tokens.toIntArray())
            val next = try {
                decoder.run(mapOf(decIdsName to ids, decEncName to hidden)).use { r ->
                    maskedArgmax(logitsOf(r), tokens.size - 1, if (i == 0) vocab.firstStepMask else vocab.textMask)
                }
            } finally { ids.close() }
            if (next == vocab.eot) break
            tokens.add(next)
            if (repeatingTail(tokens.subList(prompt, tokens.size))) break
        }
        return Decoded(tokens.subList(prompt, tokens.size).toList(), language, detected, pHi, lidMs)
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
        runCatching { encoder.close() }
        runCatching { decoder.close() }
        runCatching { decoderPast?.close() }
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
