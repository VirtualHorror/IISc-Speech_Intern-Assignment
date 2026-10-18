package `in`.iisc.vaani.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import `in`.iisc.vaani.codec.IndicPhonemizer
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * On-device Indic neural TTS: VITS (`vaani_tts_vits.int8.onnx`) via ONNX Runtime.
 *
 * Phonemisation is done by our own [IndicPhonemizer]; this class only owns the
 * ONNX session and the VITS input contract:
 *   input         int64 [1, L]  phoneme ids
 *   input_lengths int64 [1]     = L
 *   scales        float [3]     = [noise_scale, length_scale, noise_w]
 * and returns the single-pass float waveform at [VaaniTtsConfig.sampleRate].
 */
class VaaniTtsEngine(
    modelBytes: ByteArray,
    private val config: VaaniTtsConfig,
    threads: Int = 4,
    private val scales: FloatArray = floatArrayOf(0.667f, 1.0f, 0.8f),
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val inputName: String
    private val lengthsName: String
    private val scalesName: String?
    private val outputName: String

    val sampleRate: Int get() = config.sampleRate

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, opts)

        val inNames = session.inputNames
        inputName = inNames.firstOrNull { it == "input" || it.contains("phoneme") } ?: inNames.first()
        lengthsName = inNames.firstOrNull { it.contains("length") } ?: "input_lengths"
        scalesName = inNames.firstOrNull { it.contains("scale") }
        outputName = session.outputNames.first()
    }

    /** Synthesises a float waveform for [text] (phonemised internally). */
    fun synthesize(text: String): FloatArray {
        val ids = IndicPhonemizer.textToPhonemeIds(text, config.phonemeIdMap)
        if (ids.isEmpty()) return FloatArray(0)
        return synthesizeIds(ids)
    }

    fun synthesizeIds(ids: IntArray): FloatArray {
        val idLongs = LongArray(ids.size) { ids[it].toLong() }
        val inputs = HashMap<String, OnnxTensor>()
        val toClose = ArrayList<AutoCloseable>()
        try {
            val input = OnnxTensor.createTensor(
                env, LongBuffer.wrap(idLongs), longArrayOf(1, ids.size.toLong()),
            )
            val lengths = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(ids.size.toLong())), longArrayOf(1),
            )
            inputs[inputName] = input; toClose.add(input)
            inputs[lengthsName] = lengths; toClose.add(lengths)
            if (scalesName != null) {
                val sc = OnnxTensor.createTensor(env, FloatBuffer.wrap(scales), longArrayOf(scales.size.toLong()))
                inputs[scalesName] = sc; toClose.add(sc)
            }

            session.run(inputs).use { result ->
                return flattenAudio(result.get(outputName).get().value)
            }
        } finally {
            toClose.forEach { runCatching { it.close() } }
        }
    }

    /** VITS output is [1,1,T]; flatten to a bare float waveform. */
    private fun flattenAudio(value: Any?): FloatArray {
        val out = ArrayList<Float>()
        fun walk(o: Any?) {
            when (o) {
                is FloatArray -> o.forEach { out.add(it) }
                is Array<*> -> o.forEach { walk(it) }
                is Float -> out.add(o)
            }
        }
        walk(value)
        return out.toFloatArray()
    }

    override fun close() {
        runCatching { session.close() }
    }
}
