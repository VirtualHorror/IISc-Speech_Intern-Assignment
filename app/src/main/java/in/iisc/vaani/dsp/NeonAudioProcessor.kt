package `in`.iisc.vaani.dsp

/**
 * Kotlin bridge to the custom ARM64 NEON DSP library (`libvaani_dsp_neon.so`).
 *
 * Every method here is implemented from first principles in
 * `app/src/main/cpp/vaani_dsp_neon.cpp` and bound through JNI to the
 * `Java_in_iisc_vaani_dsp_NeonAudioProcessor_*` symbols. No third-party DSP
 * wrapper is involved.
 *
 * The native library is loaded lazily on first use so that pure-JVM unit tests
 * (which never touch this object) do not require the `.so`.
 */
object NeonAudioProcessor {

    // --- Whisper front-end constants (16 kHz) --------------------------------
    const val SAMPLE_RATE = 16_000
    const val N_FFT = 400
    const val HOP_LENGTH = 160
    const val N_MELS = 80
    const val N_FFT_BINS = N_FFT / 2 + 1          // 201
    const val TARGET_FRAMES = 3000                // 30 s context window
    const val PRE_EMPHASIS_COEFF = 0.97f

    @Volatile private var loaded = false

    /** Loads `libvaani_dsp_neon.so`. Safe to call repeatedly. */
    @Synchronized
    fun ensureLoaded() {
        if (!loaded) {
            System.loadLibrary("vaani_dsp_neon")
            loaded = true
        }
    }

    // --- Native primitives ---------------------------------------------------

    /** SIMD pre-emphasis high-pass: `y[n] = x[n] - coeff * x[n-1]`. */
    external fun preEmphasis(input: FloatArray, coeff: Float): FloatArray

    /** Root-mean-square energy of the whole buffer. */
    external fun rmsEnergy(input: FloatArray): Float

    /** Framed RMS energy envelope (one value per analysis frame). */
    external fun rmsEnvelope(input: FloatArray, frameSize: Int, hopSize: Int): FloatArray

    /** Short-time energy noise gate; frames below `threshold` are attenuated. */
    external fun noiseGate(
        input: FloatArray,
        frameSize: Int,
        threshold: Float,
        floorGain: Float,
    ): FloatArray

    /**
     * Whisper-exact log-Mel spectrogram, returned flat in encoder tensor order
     * `[1, nMels, targetFrames]` i.e. `out[m * targetFrames + t]`.
     */
    external fun computeLogMelSpectrogram(
        pcm: FloatArray,
        melFilters: FloatArray,
        nMels: Int,
        nFftBins: Int,
        nFft: Int,
        hopSize: Int,
        targetFrames: Int,
    ): FloatArray

    /** Saturating, round-to-nearest float32 [-1,1] -> PCM16 conversion. */
    external fun floatToPcm16(input: FloatArray): ShortArray

    // --- Convenience wrappers (Whisper defaults) -----------------------------

    fun preEmphasize(pcm: FloatArray): FloatArray {
        ensureLoaded()
        return preEmphasis(pcm, PRE_EMPHASIS_COEFF)
    }

    /**
     * Computes the `[1, 80, 3000]` log-Mel feature expected by
     * `vaani_stt_encoder.int8.onnx`. `melFilters` is the 80x201 Slaney matrix
     * loaded from `vaani_mel_filters.bin` (row-major, little-endian float32).
     */
    fun logMel(pcm: FloatArray, melFilters: FloatArray): FloatArray {
        ensureLoaded()
        require(melFilters.size == N_MELS * N_FFT_BINS) {
            "mel filter matrix must be ${N_MELS * N_FFT_BINS} floats, got ${melFilters.size}"
        }
        return computeLogMelSpectrogram(
            pcm, melFilters, N_MELS, N_FFT_BINS, N_FFT, HOP_LENGTH, TARGET_FRAMES,
        )
    }

    fun toPcm16(samples: FloatArray): ShortArray {
        ensureLoaded()
        return floatToPcm16(samples)
    }
}
