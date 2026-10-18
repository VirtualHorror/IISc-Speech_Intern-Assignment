package `in`.iisc.vaani.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.util.Log
import `in`.iisc.vaani.codec.IndicPhonemizer
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.abs

/** One synthesised clause plus how long the network took; [failedSentences] were skipped. */
data class TtsAudio(val samples: FloatArray, val sampleRate: Int, val synthMs: Long, val failedSentences: Int = 0) {
    val durationSec: Double get() = samples.size.toDouble() / sampleRate
    val rtf: Double get() = if (durationSec > 0) synthMs / 1000.0 / durationSec else 0.0
}

/**
 * On-device Indic neural TTS: VITS (`vaani_tts_vits.fp16.onnx`) via ONNX Runtime.
 *
 * Phonemisation is done by our own [IndicPhonemizer]; this class only owns the
 * ONNX session and the VITS input contract:
 *   input         int64 [1, L]  phoneme ids (Piper layout, one sentence)
 *   input_lengths int64 [1]     = L
 *   scales        float [3]     = [noise_scale, length_scale, noise_w]
 * and returns the float waveform at [VaaniTtsConfig.sampleRate]. Like Piper, each
 * sentence is synthesised separately and joined with a short silence; the result is
 * peak-normalised so every clause plays at the same loudness.
 *
 * The voice ships fp16-stored / fp32-computed: ORT constant-folds the weight Casts at
 * session creation, and no node computes in fp16.
 *
 * On the OnePlus 11R, ORT intermittently aborted the stochastic duration predictor with
 * `ScatterND ... Invalid indice found, indice = 4656722024375255042` (also `Expand`,
 * `GatherElements`). That "index" is 0x40A00002_40A00002, i.e. two float32 5.000001: the
 * spline's tail bound plus its search epsilon. Stale float memory was read as int64 indices,
 * a buffer-reuse fault rather than a numeric overflow. Hence: no memory-pattern buffer
 * reuse for this session, every run serialised on [sessionLock], empty sentences never
 * reach ORT, and a failed sentence is retried once with deterministic durations
 * ([FALLBACK_SCALES]) before it is skipped and logged.
 */
class VaaniTtsEngine(
    modelPath: String,
    private val config: VaaniTtsConfig,
    threads: Int = 4,
    private val scales: FloatArray = DEFAULT_SCALES,
    private val sentenceSilenceSec: Float = 0.15f,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, sessionOptions(threads))
    private val sessionLock = Any()

    private val inputName: String
    private val lengthsName: String
    private val scalesName: String?

    /** Piper's structural ids (BOS ^, PAD _, EOS $, word gap): a sentence of only these is empty. */
    private val structuralIds: Set<Int> =
        listOf("^", "_", "$", " ").mapNotNull { config.phonemeIdMap[it]?.firstOrNull() }.toSet()

    val sampleRate: Int get() = config.sampleRate

    init {
        val inNames = session.inputNames
        inputName = inNames.firstOrNull { it == "input" || it.contains("phoneme") } ?: inNames.first()
        lengthsName = inNames.firstOrNull { it.contains("length") } ?: "input_lengths"
        scalesName = inNames.firstOrNull { it.contains("scale") }
    }

    /**
     * Synthesises acoustic-Devanagari [text] (phonemised internally), sentence by sentence.
     * Thread-safe: concurrent callers (loopback, the clause queue, warm-up) are serialised.
     */
    fun synthesize(text: String): TtsAudio = synchronized(sessionLock) { synthesizeLocked(text) }

    private fun synthesizeLocked(text: String): TtsAudio {
        val t0 = System.nanoTime()
        if (text.isBlank()) return TtsAudio(FloatArray(0), sampleRate, 0)
        val sentences = IndicPhonemizer.textToSentenceIds(text, config.phonemeIdMap)
            .filter { ids -> ids.any { it !in structuralIds } }
        if (sentences.isEmpty()) return TtsAudio(FloatArray(0), sampleRate, 0)
        var failed = 0
        val pieces = sentences.mapNotNull { ids -> runWithRetry(ids).also { if (it == null) failed++ } }
        val gap = (sentenceSilenceSec * sampleRate).toInt()
        val total = pieces.sumOf { it.size } + gap * (pieces.size - 1).coerceAtLeast(0)
        val out = FloatArray(total)
        var off = 0
        for ((i, p) in pieces.withIndex()) {
            p.copyInto(out, off); off += p.size
            if (i < pieces.size - 1) off += gap
        }
        normalisePeak(out)
        return TtsAudio(out, sampleRate, (System.nanoTime() - t0) / 1_000_000, failed)
    }

    /** One sentence; on an ORT failure retries once with deterministic durations, else null. */
    private fun runWithRetry(ids: IntArray): FloatArray? = try {
        run(ids, scales)
    } catch (e: OrtException) {
        Log.w(TAG, "VITS failed (${e.message?.take(160)}); retrying with deterministic durations")
        try {
            run(ids, FALLBACK_SCALES)
        } catch (e2: OrtException) {
            Log.e(TAG, "VITS retry failed; sentence of ${ids.size} ids skipped", e2)
            null
        }
    }

    /** Runs VITS on one Piper id sequence and returns the raw waveform (caller holds [sessionLock]). */
    private fun run(ids: IntArray, scales: FloatArray): FloatArray {
        val toClose = ArrayList<AutoCloseable>()
        try {
            val input = OnnxTensor.createTensor(
                env, LongBuffer.wrap(LongArray(ids.size) { ids[it].toLong() }), longArrayOf(1, ids.size.toLong()),
            ).also { toClose.add(it) }
            val lengths = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(ids.size.toLong())), longArrayOf(1),
            ).also { toClose.add(it) }
            val inputs = HashMap<String, OnnxTensor>()
            inputs[inputName] = input
            inputs[lengthsName] = lengths
            scalesName?.let {
                inputs[it] = OnnxTensor.createTensor(env, FloatBuffer.wrap(scales), longArrayOf(scales.size.toLong()))
                    .also { t -> toClose.add(t) }
            }
            session.run(inputs).use { result ->
                val buf = (result.get(0) as OnnxTensor).floatBuffer      // [1,1,T] -> flat T
                return FloatArray(buf.remaining()).also { buf.get(it) }
            }
        } finally {
            toClose.forEach { runCatching { it.close() } }
        }
    }

    private fun normalisePeak(x: FloatArray, target: Float = 0.92f) {
        var peak = 0f
        for (v in x) peak = maxOf(peak, abs(v))
        if (peak < 1e-4f) return
        val g = target / peak
        for (i in x.indices) x[i] *= g
    }

    override fun close() {
        synchronized(sessionLock) { runCatching { session.close() } }
    }

    companion object {
        private const val TAG = "VaaniTts"

        /** [noise_scale, length_scale, noise_w]: low duration noise keeps the predictor stable. */
        val DEFAULT_SCALES = floatArrayOf(0.667f, 1.0f, 0.2f)

        /** Retry settings: deterministic durations, calmer pitch/energy noise. */
        val FALLBACK_SCALES = floatArrayOf(0.5f, 1.0f, 0.0f)

        /** The STT options minus ORT's memory-pattern buffer reuse (see the class comment). */
        fun sessionOptions(threads: Int): OrtSession.SessionOptions =
            VaaniSttEngine.sessionOptions(threads).apply { setMemoryPatternOptimization(false) }
    }
}
