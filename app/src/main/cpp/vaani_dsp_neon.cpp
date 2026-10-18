// =============================================================================
//  VaaniEdge — Custom ARM64 NEON Audio DSP  (libvaani_dsp_neon.so)
//  IISc Computational Intelligence Lab
//
//  First-principles C++17 signal-processing primitives for the on-device speech
//  pipeline, hand-written against <arm_neon.h> for the Snapdragon 8+ Gen 1
//  (Cortex-X2 / A710, arm64-v8a). Nothing here is generated or vendored.
//
//  JNI surface binds to Kotlin class  in.iisc.vaani.dsp.NeonAudioProcessor
//  => exported symbols are  Java_in_iisc_vaani_dsp_NeonAudioProcessor_*
//
//  Provided primitives:
//    1. preEmphasis        y[n] = x[n] - a*x[n-1]           (SIMD)
//    2. rmsEnergy          scalar RMS of a buffer           (SIMD reduce)
//    3. rmsEnvelope        framed RMS envelope              (SIMD reduce)
//    4. noiseGate          short-time energy noise gate     (SIMD)
//    5. computeLogMel      Whisper-exact 80-bin log-Mel     (FFT + NEON matmul)
//    6. floatToPcm16       saturating float32 -> PCM16      (SIMD)
// =============================================================================

#include <jni.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

#if defined(__aarch64__)
#include <arm_neon.h>
#define VAANI_NEON 1
#else
#define VAANI_NEON 0
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;

// --------------------------------------------------------------------------
// Horizontal sum of a float32x4 (AArch64 has a single-instruction reduce).
// --------------------------------------------------------------------------
#if VAANI_NEON
static inline float hsum(float32x4_t v) { return vaddvq_f32(v); }
#endif

// --------------------------------------------------------------------------
// Pre-emphasis high-pass:  y[n] = x[n] - coeff * x[n-1],  y[0] = x[0].
// Depends only on the input, so it vectorises cleanly (no feedback).
// --------------------------------------------------------------------------
static void preEmphasis(const float* x, float* y, int n, float coeff) {
    if (n <= 0) return;
    y[0] = x[0];
    int i = 1;
#if VAANI_NEON
    const float32x4_t c = vdupq_n_f32(coeff);
    for (; i + 4 <= n; i += 4) {
        float32x4_t cur  = vld1q_f32(x + i);
        float32x4_t prev = vld1q_f32(x + i - 1);
        // cur - prev*c
        vst1q_f32(y + i, vmlsq_f32(cur, prev, c));
    }
#endif
    for (; i < n; ++i) y[i] = x[i] - coeff * x[i - 1];
}

// --------------------------------------------------------------------------
// Root-mean-square energy of a whole buffer.
// --------------------------------------------------------------------------
static float rmsEnergy(const float* x, int n) {
    if (n <= 0) return 0.0f;
    int i = 0;
    double acc = 0.0;
#if VAANI_NEON
    float32x4_t v = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t s = vld1q_f32(x + i);
        v = vmlaq_f32(v, s, s);   // v += s*s
    }
    acc += (double) hsum(v);
#endif
    for (; i < n; ++i) acc += (double) x[i] * (double) x[i];
    return (float) std::sqrt(acc / (double) n);
}

// --------------------------------------------------------------------------
// Short-time energy noise gate. Frames below `threshold` RMS are attenuated
// by `floorGain`; a soft knee avoids clicks at the boundary.
// --------------------------------------------------------------------------
static void noiseGate(const float* x, float* y, int n, int frame,
                      float threshold, float floorGain) {
    if (frame <= 0) frame = 400;
    for (int start = 0; start < n; start += frame) {
        int len = std::min(frame, n - start);
        float r = rmsEnergy(x + start, len);
        float gain = (r >= threshold) ? 1.0f
                     : floorGain + (1.0f - floorGain) * (r / (threshold + 1e-9f));
        int i = 0;
#if VAANI_NEON
        float32x4_t g = vdupq_n_f32(gain);
        for (; i + 4 <= len; i += 4)
            vst1q_f32(y + start + i, vmulq_f32(vld1q_f32(x + start + i), g));
#endif
        for (; i < len; ++i) y[start + i] = x[start + i] * gain;
    }
}

// --------------------------------------------------------------------------
// Recursive FFT: radix-2 decimation-in-time when the length is even, direct
// DFT fallback for the odd factor (n_fft = 400 = 2^4 * 25). `out` is
// interleaved complex (re, im) of length 2*n; `in` is real of length n.
// --------------------------------------------------------------------------
static void dft(const float* in, int n, float* out) {
    for (int k = 0; k < n; ++k) {
        double re = 0.0, im = 0.0;
        for (int t = 0; t < n; ++t) {
            double ang = -2.0 * (double) kPi * (double) k * (double) t / (double) n;
            re += (double) in[t] * std::cos(ang);
            im += (double) in[t] * std::sin(ang);
        }
        out[2 * k]     = (float) re;
        out[2 * k + 1] = (float) im;
    }
}

static void fft(const float* in, int n, float* out) {
    if (n == 1) { out[0] = in[0]; out[1] = 0.0f; return; }
    if (n % 2 == 1) { dft(in, n, out); return; }

    const int h = n / 2;
    std::vector<float> even(h), odd(h), eo(2 * h), oo(2 * h);
    for (int i = 0; i < h; ++i) { even[i] = in[2 * i]; odd[i] = in[2 * i + 1]; }
    fft(even.data(), h, eo.data());
    fft(odd.data(), h, oo.data());

    for (int k = 0; k < h; ++k) {
        double ang = -2.0 * (double) kPi * (double) k / (double) n;
        float cr = (float) std::cos(ang), ci = (float) std::sin(ang);
        float orr = oo[2 * k], oii = oo[2 * k + 1];
        float tr = cr * orr - ci * oii;
        float ti = cr * oii + ci * orr;
        out[2 * k]           = eo[2 * k]     + tr;
        out[2 * k + 1]       = eo[2 * k + 1] + ti;
        out[2 * (k + h)]     = eo[2 * k]     - tr;
        out[2 * (k + h) + 1] = eo[2 * k + 1] - ti;
    }
}

// NEON dot product filter_row . power_spectrum  (length nBins).
static float melDot(const float* filt, const float* pw, int nBins) {
    int i = 0;
    float acc = 0.0f;
#if VAANI_NEON
    float32x4_t v = vdupq_n_f32(0.0f);
    for (; i + 4 <= nBins; i += 4)
        v = vmlaq_f32(v, vld1q_f32(filt + i), vld1q_f32(pw + i));
    acc = hsum(v);
#endif
    for (; i < nBins; ++i) acc += filt[i] * pw[i];
    return acc;
}

// --------------------------------------------------------------------------
// Whisper-exact log-Mel spectrogram.
//   - trims/zero-pads the input to targetFrames*hop samples (30 s @ 16 kHz),
//   - reflect-pads by nFft/2 (center=True STFT),
//   - periodic Hann window, hop `hop`, power spectrum (nFftBins = nFft/2+1),
//   - Mel projection with the provided [nMels x nFftBins] Slaney matrix,
//   - log10, global dynamic-range clamp (max-8), (x+4)/4 normalisation.
// Output layout matches the encoder tensor [1, nMels, targetFrames]:
//   out[m*targetFrames + t].
// --------------------------------------------------------------------------
static void computeLogMel(const float* pcm, int pcmLen,
                          const float* mel, int nMels, int nFftBins,
                          int nFft, int hop, int targetFrames,
                          float* out) {
    const int N = targetFrames * hop;          // canonical sample count (480000)
    const int P = nFft / 2;                    // reflect pad width (200)

    // Zero-pad / trim to N, then reflect-pad by P on both ends.
    std::vector<float> padded(N + 2 * P, 0.0f);
    const int copy = std::min(pcmLen, N);
    for (int i = 0; i < copy; ++i) padded[P + i] = pcm[i];
    for (int k = 0; k < P; ++k) {
        padded[k]             = padded[2 * P - k];       // left  reflect (no edge repeat)
        padded[P + N + k]     = padded[P + N - 2 - k];   // right reflect
    }

    // Periodic Hann window.
    std::vector<float> hann(nFft);
    for (int i = 0; i < nFft; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * (float) i / (float) nFft));

    std::vector<float> frame(nFft), spec(2 * nFft), power(nFftBins);

    float globalMax = -1e30f;
    for (int t = 0; t < targetFrames; ++t) {
        const int off = t * hop;
        for (int i = 0; i < nFft; ++i) frame[i] = padded[off + i] * hann[i];
        fft(frame.data(), nFft, spec.data());
        for (int b = 0; b < nFftBins; ++b) {
            float re = spec[2 * b], im = spec[2 * b + 1];
            power[b] = re * re + im * im;       // |STFT|^2
        }
        for (int m = 0; m < nMels; ++m) {
            float e = melDot(mel + (size_t) m * nFftBins, power.data(), nFftBins);
            float lg = std::log10(std::max(e, 1e-10f));
            out[(size_t) m * targetFrames + t] = lg;
            if (lg > globalMax) globalMax = lg;
        }
    }

    // Dynamic-range clamp to (globalMax - 8) then normalise to ~[-1, 1].
    const float floor = globalMax - 8.0f;
    const size_t total = (size_t) nMels * targetFrames;
    size_t i = 0;
#if VAANI_NEON
    float32x4_t vfloor = vdupq_n_f32(floor);
    float32x4_t vfour  = vdupq_n_f32(4.0f);
    float32x4_t vinv4  = vdupq_n_f32(0.25f);
    for (; i + 4 <= total; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        v = vmaxq_f32(v, vfloor);
        v = vmulq_f32(vaddq_f32(v, vfour), vinv4);
        vst1q_f32(out + i, v);
    }
#endif
    for (; i < total; ++i) {
        float v = std::max(out[i], floor);
        out[i] = (v + 4.0f) * 0.25f;
    }
}

// --------------------------------------------------------------------------
// Saturating float32 [-1,1] -> PCM16 with round-to-nearest.
// --------------------------------------------------------------------------
static void floatToPcm16(const float* x, int16_t* y, int n) {
    int i = 0;
#if VAANI_NEON
    const float32x4_t scale = vdupq_n_f32(32767.0f);
    const float32x4_t lo = vdupq_n_f32(-1.0f), hi = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vmaxq_f32(lo, vminq_f32(hi, v));
        v = vmulq_f32(v, scale);
        int32x4_t  xi = vcvtaq_s32_f32(v);   // round to nearest, ties away
        int16x4_t  s  = vqmovn_s32(xi);      // saturate to int16
        vst1_s16(y + i, s);
    }
#endif
    for (; i < n; ++i) {
        float v = std::max(-1.0f, std::min(1.0f, x[i]));
        int s = (int) std::lround(v * 32767.0f);
        y[i] = (int16_t) std::max(-32768, std::min(32767, s));
    }
}

} // namespace

// ============================= JNI bindings ================================
extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_preEmphasis(
        JNIEnv* env, jobject, jfloatArray input, jfloat coeff) {
    const jsize n = env->GetArrayLength(input);
    jfloat* in = env->GetFloatArrayElements(input, nullptr);
    std::vector<float> out(n);
    preEmphasis(in, out.data(), (int) n, coeff);
    env->ReleaseFloatArrayElements(input, in, JNI_ABORT);
    jfloatArray res = env->NewFloatArray(n);
    env->SetFloatArrayRegion(res, 0, n, out.data());
    return res;
}

JNIEXPORT jfloat JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_rmsEnergy(
        JNIEnv* env, jobject, jfloatArray input) {
    const jsize n = env->GetArrayLength(input);
    jfloat* in = env->GetFloatArrayElements(input, nullptr);
    float r = rmsEnergy(in, (int) n);
    env->ReleaseFloatArrayElements(input, in, JNI_ABORT);
    return r;
}

JNIEXPORT jfloatArray JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_rmsEnvelope(
        JNIEnv* env, jobject, jfloatArray input, jint frameSize, jint hopSize) {
    const jsize n = env->GetArrayLength(input);
    jfloat* in = env->GetFloatArrayElements(input, nullptr);
    if (frameSize <= 0) frameSize = 400;
    if (hopSize <= 0) hopSize = 160;
    const int frames = (n >= frameSize) ? (1 + (n - frameSize) / hopSize) : 0;
    std::vector<float> env_out(std::max(0, frames));
    for (int t = 0; t < frames; ++t)
        env_out[t] = rmsEnergy(in + (size_t) t * hopSize, frameSize);
    env->ReleaseFloatArrayElements(input, in, JNI_ABORT);
    jfloatArray res = env->NewFloatArray(frames);
    if (frames > 0) env->SetFloatArrayRegion(res, 0, frames, env_out.data());
    return res;
}

JNIEXPORT jfloatArray JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_noiseGate(
        JNIEnv* env, jobject, jfloatArray input, jint frameSize,
        jfloat threshold, jfloat floorGain) {
    const jsize n = env->GetArrayLength(input);
    jfloat* in = env->GetFloatArrayElements(input, nullptr);
    std::vector<float> out(n);
    noiseGate(in, out.data(), (int) n, frameSize, threshold, floorGain);
    env->ReleaseFloatArrayElements(input, in, JNI_ABORT);
    jfloatArray res = env->NewFloatArray(n);
    env->SetFloatArrayRegion(res, 0, n, out.data());
    return res;
}

JNIEXPORT jfloatArray JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_computeLogMelSpectrogram(
        JNIEnv* env, jobject, jfloatArray pcm, jfloatArray melFilters,
        jint nMels, jint nFftBins, jint nFft, jint hopSize, jint targetFrames) {
    const jsize pcmLen = env->GetArrayLength(pcm);
    jfloat* pcmp = env->GetFloatArrayElements(pcm, nullptr);
    jfloat* melp = env->GetFloatArrayElements(melFilters, nullptr);

    const size_t total = (size_t) nMels * targetFrames;
    std::vector<float> out(total);
    computeLogMel(pcmp, (int) pcmLen, melp, nMels, nFftBins, nFft, hopSize,
                  targetFrames, out.data());

    env->ReleaseFloatArrayElements(pcm, pcmp, JNI_ABORT);
    env->ReleaseFloatArrayElements(melFilters, melp, JNI_ABORT);
    jfloatArray res = env->NewFloatArray((jsize) total);
    env->SetFloatArrayRegion(res, 0, (jsize) total, out.data());
    return res;
}

JNIEXPORT jshortArray JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_floatToPcm16(
        JNIEnv* env, jobject, jfloatArray input) {
    const jsize n = env->GetArrayLength(input);
    jfloat* in = env->GetFloatArrayElements(input, nullptr);
    std::vector<int16_t> out(n);
    floatToPcm16(in, out.data(), (int) n);
    env->ReleaseFloatArrayElements(input, in, JNI_ABORT);
    jshortArray res = env->NewShortArray(n);
    env->SetShortArrayRegion(res, 0, n, reinterpret_cast<const jshort*>(out.data()));
    return res;
}

} // extern "C"
