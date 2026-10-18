package `in`.iisc.vaani.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import `in`.iisc.vaani.codec.IndicPhonemizer
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.abs

/** One synthesised clause plus how long the network took. */
data class TtsAudio(val samples: FloatArray, val sampleRate: Int, val synthMs: Long) {
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
 * session creation. (The Session-1 INT8 build used ConvInteger, which ORT Android 1.18
 * cannot load and which ran 3.2x slower on CPU — see speech_lab/evaluation_metrics.json.)
 */
class VaaniTtsEngine(
    modelPath: String,
    private val config: VaaniTtsConfig,
    threads: Int = 4,
    private val scales: FloatArray = floatArrayOf(0.667f, 1.0f, 0.8f),
    private val sentenceSilenceSec: Float = 0.15f,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath, VaaniSttEngine.sessionOptions(threads))

    private val inputName: String
    private val lengthsName: String
    private val scalesName: String?

    val sampleRate: Int get() = config.sampleRate

    init {
        val inNames = session.inputNames
        inputName = inNames.firstOrNull { it == "input" || it.contains("phoneme") } ?: inNames.first()
        lengthsName = inNames.firstOrNull { it.contains("length") } ?: "input_lengths"
        scalesName = inNames.firstOrNull { it.contains("scale") }
    }

    /** Synthesises acoustic-Devanagari [text] (phonemised internally), sentence by sentence. */
    fun synthesize(text: String): TtsAudio {
        val t0 = System.nanoTime()
        val sentences = IndicPhonemizer.textToSentenceIds(text, config.phonemeIdMap)
        if (sentences.isEmpty()) return TtsAudio(FloatArray(0), sampleRate, 0)
        val gap = (sentenceSilenceSec * sampleRate).toInt()
        val pieces = sentences.map { synthesizeIds(it) }
        val total = pieces.sumOf { it.size } + gap * (pieces.size - 1)
        val out = FloatArray(total)
        var off = 0
        for ((i, p) in pieces.withIndex()) {
            p.copyInto(out, off); off += p.size
            if (i < pieces.size - 1) off += gap
        }
        normalisePeak(out)
        return TtsAudio(out, sampleRate, (System.nanoTime() - t0) / 1_000_000)
    }

    /** Runs VITS on one Piper id sequence and returns the raw waveform. */
    @Synchronized
    fun synthesizeIds(ids: IntArray): FloatArray {
        if (ids.isEmpty()) return FloatArray(0)
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
        runCatching { session.close() }
    }
}
