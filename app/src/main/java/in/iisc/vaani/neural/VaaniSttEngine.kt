package `in`.iisc.vaani.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import `in`.iisc.vaani.dsp.NeonAudioProcessor
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** Result of a single on-device transcription. */
data class SttResult(
    val text: String,
    val tokenIds: List<Int>,
    val sttLatencyMs: Long,
    val sttRtf: Double,
)

/**
 * First-principles Whisper-style STT engine on ONNX Runtime.
 *
 * Pipeline:
 *   PCM --[NEON log-Mel]--> [1,80,3000] --> encoder --> hidden states
 *        --> greedy autoregressive decode (logit-level script masking) --> ids
 *        --> byte-level BPE detokenise --> raw text
 *
 * The **logit-level script masking** sets every suppressed-script and timestamp
 * token logit to -inf before the argmax, so Hinglish speech can never drift into
 * Urdu/Perso-Arabic, Cyrillic, or CJK output — solving the script-drift problem
 * at the decode step rather than with fragile post-hoc string cleanup.
 *
 * Decoding here is the simple (cache-free) greedy loop that re-feeds the growing
 * token sequence each step: correct and self-contained for Session 1. A KV-cache
 * decoder path (`decoder_model_merged`) is a Session 2 throughput optimisation.
 */
class VaaniSttEngine(
    encoderBytes: ByteArray,
    decoderBytes: ByteArray,
    private val vocab: VaaniVocab,
    private val melFilters: FloatArray,
    threads: Int = 4,
    private val maxDecodeTokens: Int = 224,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val decoder: OrtSession

    private val encInputName: String
    private val encOutputName: String
    private val decIdsName: String
    private val decEncName: String
    private val decLogitsName: String

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        encoder = env.createSession(encoderBytes, opts)
        decoder = env.createSession(decoderBytes, opts)

        encInputName = encoder.inputNames.first()
        encOutputName = encoder.outputNames.first()
        decIdsName = decoder.inputNames.firstOrNull { it.contains("input_ids") } ?: decoder.inputNames.first()
        decEncName = decoder.inputNames.firstOrNull { it.contains("encoder_hidden_states") }
            ?: decoder.inputNames.first { it != decIdsName }
        decLogitsName = decoder.outputNames.firstOrNull { it.contains("logits") } ?: decoder.outputNames.first()
    }

    /** Transcribes 16 kHz mono float PCM and reports latency + real-time factor. */
    fun transcribe(pcm: FloatArray, language: String = "hi"): SttResult {
        val t0 = System.nanoTime()

        // 1) NEON log-Mel feature [1, 80, 3000].
        val mel = NeonAudioProcessor.logMel(pcm, melFilters)
        val encInput = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(mel),
            longArrayOf(1, NeonAudioProcessor.N_MELS.toLong(), NeonAudioProcessor.TARGET_FRAMES.toLong()),
        )

        // 2) Encoder.
        val encResult = encoder.run(mapOf(encInputName to encInput))
        val hidden = encResult.get(0) as OnnxTensor   // reused across decode steps

        // 3) Greedy autoregressive decode with logit-level script masking.
        val tokens = ArrayList<Int>()
        tokens.add(vocab.sot)
        tokens.add(vocab.languageToken(language))
        tokens.add(vocab.transcribe)
        tokens.add(vocab.noTimestamps)
        val promptLen = tokens.size

        try {
            for (step in 0 until maxDecodeTokens) {
                val idsArr = LongArray(tokens.size) { tokens[it].toLong() }
                val idsTensor = OnnxTensor.createTensor(
                    env, LongBuffer.wrap(idsArr), longArrayOf(1, tokens.size.toLong()),
                )
                val out = decoder.run(mapOf(decIdsName to idsTensor, decEncName to hidden))
                val next: Int
                try {
                    val logits = out.get(decLogitsName).get().value as Array<*>          // [1, L, V]
                    val lastRow = (logits[0] as Array<*>)[tokens.size - 1] as FloatArray
                    next = maskedArgmax(lastRow)
                } finally {
                    out.close()
                    idsTensor.close()
                }
                if (next == vocab.eot) break
                tokens.add(next)
            }
        } finally {
            encResult.close()
            encInput.close()
        }

        val generated = tokens.subList(promptLen, tokens.size).toList()
        val text = vocab.decode(generated).trim()

        val latencyMs = (System.nanoTime() - t0) / 1_000_000
        val audioSec = pcm.size.toDouble() / NeonAudioProcessor.SAMPLE_RATE
        val rtf = if (audioSec > 0) (latencyMs / 1000.0) / audioSec else 0.0
        return SttResult(text, generated, latencyMs, rtf)
    }

    /** argmax over the vocabulary with suppressed/timestamp logits forced to -inf. */
    private fun maskedArgmax(logits: FloatArray): Int {
        var bestId = 0
        var bestVal = Float.NEGATIVE_INFINITY
        for (id in logits.indices) {
            if (vocab.isSuppressed(id) || id >= vocab.timestampBegin) continue
            val v = logits[id]
            if (v > bestVal) { bestVal = v; bestId = id }
        }
        return bestId
    }

    override fun close() {
        runCatching { encoder.close() }
        runCatching { decoder.close() }
    }
}
