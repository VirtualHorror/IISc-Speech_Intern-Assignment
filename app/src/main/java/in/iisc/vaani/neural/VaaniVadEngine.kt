package `in`.iisc.vaani.neural

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * On-device Voice Activity Detection, running the Silero VAD ONNX graph
 * (`vaani_vad.onnx`) frame-by-frame via ONNX Runtime.
 *
 * The recurrent state tensors are carried across calls (this is a stateful
 * streaming model), and a trailing-silence auto-endpointer fires after
 * [endpointSilenceMs] of continuous non-speech once speech has started (500 ms by
 * default: long enough not to cut "Aaj ka schedule kya hai, ... aur" at the comma pause).
 *
 * Supports both the v5 single-`state` layout and the v4 `h`/`c` layout; the
 * active layout is detected from the model's input names. For v5 each call is fed
 * the last 64 samples of the previous call + the new 512-sample chunk (576 total),
 * exactly like Silero's reference wrapper: measured on speech + noise this lowers the
 * silence false-alarm probability from ~0.25 to ~0.05 (speech_lab notes, Session 2).
 */
class VaaniVadEngine(
    modelPath: String,
    private val sampleRate: Int = 16_000,
    private val speechThreshold: Float = 0.5f,
    var endpointSilenceMs: Int = 500,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputNames: Set<String>
    private val usesCombinedState: Boolean

    // Recurrent state (v5: [2,1,128]=256 ; v4: h,c each [2,1,64]=128).
    private var state = FloatArray(2 * 1 * 128)
    private val context = FloatArray(CONTEXT_SAMPLES)
    private var hState = FloatArray(2 * 1 * 64)
    private var cState = FloatArray(2 * 1 * 64)

    // Endpointer bookkeeping.
    private var speechStarted = false
    private var trailingSilenceMs = 0

    /** True once speech has been detected in the current utterance. */
    val inSpeech: Boolean get() = speechStarted

    /** Last P(speech) produced by [process]. */
    @Volatile var lastProbability: Float = 0f
        private set

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        }
        session = env.createSession(modelPath, opts)
        inputNames = session.inputNames
        usesCombinedState = inputNames.contains("state")
    }

    /** Resets recurrent state and endpointer; call before each new utterance. */
    fun reset() {
        state.fill(0f); hState.fill(0f); cState.fill(0f); context.fill(0f)
        speechStarted = false
        trailingSilenceMs = 0
    }

    /** Runs one audio chunk (ideally [CHUNK_SAMPLES] = 512 samples) and returns P(speech). */
    fun process(chunk: FloatArray): Float {
        val inputs = HashMap<String, OnnxTensor>()
        val toClose = ArrayList<AutoCloseable>()
        try {
            val x = if (usesCombinedState) FloatArray(CONTEXT_SAMPLES + chunk.size).also {
                context.copyInto(it, 0)
                chunk.copyInto(it, CONTEXT_SAMPLES)
            } else chunk
            val input = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(x), longArrayOf(1, x.size.toLong()),
            )
            inputs["input"] = input; toClose.add(input)
            if (usesCombinedState) x.copyInto(context, 0, x.size - CONTEXT_SAMPLES, x.size)

            val sr = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(sampleRate.toLong())), longArrayOf(),
            )
            inputs["sr"] = sr; toClose.add(sr)

            if (usesCombinedState) {
                val st = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128),
                )
                inputs["state"] = st; toClose.add(st)
            } else {
                val h = OnnxTensor.createTensor(env, FloatBuffer.wrap(hState), longArrayOf(2, 1, 64))
                val c = OnnxTensor.createTensor(env, FloatBuffer.wrap(cState), longArrayOf(2, 1, 64))
                inputs["h"] = h; inputs["c"] = c; toClose.add(h); toClose.add(c)
            }

            session.run(inputs).use { result ->
                val prob = firstFloat(result.get(0))
                updateState(result)
                lastProbability = prob
                return prob
            }
        } finally {
            toClose.forEach { runCatching { it.close() } }
        }
    }

    /**
     * Feeds a chunk, updates the endpointer, and returns true exactly once when
     * trailing silence has exceeded [endpointSilenceMs] after speech began.
     */
    fun processAndCheckEndpoint(chunk: FloatArray): Boolean {
        val prob = process(chunk)
        val chunkMs = (chunk.size * 1000) / sampleRate
        if (prob >= speechThreshold) {
            speechStarted = true
            trailingSilenceMs = 0
        } else if (speechStarted) {
            trailingSilenceMs += chunkMs
            if (trailingSilenceMs >= endpointSilenceMs) return true
        }
        return false
    }

    private fun updateState(result: OrtSession.Result) {
        try {
            if (usesCombinedState) {
                if (result.size() > 1) state = flatten(result.get(1).value)
            } else {
                if (result.size() > 1) hState = flatten(result.get(1).value)
                if (result.size() > 2) cState = flatten(result.get(2).value)
            }
        } catch (_: Exception) {
            // Keep previous state if the output layout is unexpected.
        }
    }

    private fun firstFloat(value: OnnxValue): Float = when (val v = value.value) {
        is Array<*> -> {
            val inner = v.firstOrNull()
            if (inner is FloatArray) inner.firstOrNull() ?: 0f else 0f
        }
        is FloatArray -> v.firstOrNull() ?: 0f
        else -> 0f
    }

    /** Flattens an arbitrarily-nested float ND array (as returned by ORT) into a FloatArray. */
    private fun flatten(value: Any?): FloatArray {
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

    companion object {
        /** 32 ms @ 16 kHz. */
        const val CHUNK_SAMPLES = 512

        /** Silero v5 context carried between calls at 16 kHz. */
        const val CONTEXT_SAMPLES = 64
    }
}
