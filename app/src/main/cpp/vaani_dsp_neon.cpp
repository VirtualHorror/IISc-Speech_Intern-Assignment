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
//  A plain C ABI (vaani_log_mel / vaani_resample*) exposes the same kernels to
//  the host-side lab (speech_lab/verify_closed_loop.py loads a host build via
//  ctypes), so the closed-loop verification exercises THIS file, not a port.
//  On non-AArch64 hosts the scalar fallbacks below compute identical math.
//
//  Provided primitives:
//    1. preEmphasis        y[n] = x[n] - a*x[n-1]           (SIMD)
//    2. rmsEnergy          scalar RMS of a buffer           (SIMD reduce)
//    3. rmsEnvelope        framed RMS envelope              (SIMD reduce)
//    4. noiseGate          short-time energy noise gate     (SIMD)
//    5. computeLogMel      Whisper-exact 80-bin log-Mel     (planned FFT + sparse NEON Mel)
//    6. floatToPcm16       saturating float32 -> PCM16      (SIMD)
//    7. resampleSinc       Kaiser-windowed sinc resampler   (SIMD MAC, e.g. 22.05k -> 16k)
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

#define VAANI_API extern "C" __attribute__((visibility("default")))

namespace {

constexpr double kPi = 3.14159265358979323846;

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
// Planned mixed-radix FFT for real input of length n = 2^p * m (Whisper:
// 400 = 2^4 * 25). Radix-2 decimation-in-time down to the odd factor m,
// which is evaluated as a direct DFT from a precomputed m x m table.
// All twiddles are computed ONCE in double precision; recursion scratch is
// preallocated per level, so the per-frame hot path does no allocation and
// no trigonometry. `out` is interleaved complex (re, im) of length 2*n.
// --------------------------------------------------------------------------
struct FftPlan {
    int n = 0, base = 0, levels = 0;
    std::vector<float> twRe, twIm;          // exp(-2*pi*i*k/n), k in [0, n)
    std::vector<float> dftRe, dftIm;        // exp(-2*pi*i*k*t/base), [k*base + t]
    std::vector<std::vector<float>> evenIn, oddIn, evenOut, oddOut;

    explicit FftPlan(int nfft) : n(nfft) {
        base = n;
        while (base % 2 == 0 && base > 1) { base /= 2; ++levels; }
        twRe.resize(n); twIm.resize(n);
        for (int k = 0; k < n; ++k) {
            const double a = -2.0 * kPi * (double) k / (double) n;
            twRe[k] = (float) std::cos(a); twIm[k] = (float) std::sin(a);
        }
        dftRe.resize((size_t) base * base); dftIm.resize((size_t) base * base);
        for (int k = 0; k < base; ++k)
            for (int t = 0; t < base; ++t) {
                const double a = -2.0 * kPi * (double) ((k * t) % base) / (double) base;
                dftRe[(size_t) k * base + t] = (float) std::cos(a);
                dftIm[(size_t) k * base + t] = (float) std::sin(a);
            }
        evenIn.resize(levels); oddIn.resize(levels); evenOut.resize(levels); oddOut.resize(levels);
        int len = n;
        for (int l = 0; l < levels; ++l) {
            const int h = len / 2;
            evenIn[l].resize(h); oddIn[l].resize(h);
            evenOut[l].resize(2 * h); oddOut[l].resize(2 * h);
            len = h;
        }
    }

    void dftBase(const float* in, float* out) const {
        for (int k = 0; k < base; ++k) {
            const float* cr = &dftRe[(size_t) k * base];
            const float* ci = &dftIm[(size_t) k * base];
            float re = 0.0f, im = 0.0f;
            for (int t = 0; t < base; ++t) { re += in[t] * cr[t]; im += in[t] * ci[t]; }
            out[2 * k] = re; out[2 * k + 1] = im;
        }
    }

    // FFT of length n >> level, twiddle stride (1 << level) into the master table.
    void run(const float* in, float* out, int level) {
        const int len = n >> level;
        if (level == levels) { dftBase(in, out); return; }
        const int h = len / 2;
        float* ev = evenIn[level].data(); float* od = oddIn[level].data();
        for (int i = 0; i < h; ++i) { ev[i] = in[2 * i]; od[i] = in[2 * i + 1]; }
        float* eo = evenOut[level].data(); float* oo = oddOut[level].data();
        run(ev, eo, level + 1);
        run(od, oo, level + 1);
        const int stride = 1 << level;
        for (int k = 0; k < h; ++k) {
            const float cr = twRe[(size_t) k * stride], ci = twIm[(size_t) k * stride];
            const float orr = oo[2 * k], oii = oo[2 * k + 1];
            const float tr = cr * orr - ci * oii;
            const float ti = cr * oii + ci * orr;
            out[2 * k]           = eo[2 * k]     + tr;
            out[2 * k + 1]       = eo[2 * k + 1] + ti;
            out[2 * (k + h)]     = eo[2 * k]     - tr;
            out[2 * (k + h) + 1] = eo[2 * k + 1] - ti;
        }
    }
};

// NEON dot product over the non-zero band [lo, hi) of one Mel filter row.
static float melDotRange(const float* filt, const float* pw, int lo, int hi) {
    int i = lo;
    float acc = 0.0f;
#if VAANI_NEON
    float32x4_t v = vdupq_n_f32(0.0f);
    for (; i + 4 <= hi; i += 4)
        v = vmlaq_f32(v, vld1q_f32(filt + i), vld1q_f32(pw + i));
    acc = hsum(v);
#endif
    for (; i < hi; ++i) acc += filt[i] * pw[i];
    return acc;
}

// --------------------------------------------------------------------------
// Whisper-exact log-Mel spectrogram (== whisper.audio.log_mel_spectrogram on
// pad_or_trim(audio)):
//   - trims/zero-pads the input to targetFrames*hop samples (30 s @ 16 kHz),
//   - reflect-pads by nFft/2 (torch.stft center=True, pad_mode="reflect"),
//   - periodic Hann window, hop `hop`, power spectrum |X|^2 (nFftBins = nFft/2+1),
//   - Mel projection with the provided [nMels x nFftBins] Slaney matrix,
//   - log10(max(x, 1e-10)), global clamp to (max - 8), then (x + 4) / 4.
// Frames whose analysis window lies wholly inside the zero padding have
// |X|^2 == 0 exactly, so they are filled with log10(1e-10) without an FFT:
// bit-identical output, ~10x less work for a 3 s command in a 30 s window.
// Output layout matches the encoder tensor [1, nMels, targetFrames]:
//   out[m*targetFrames + t].
// Returns the number of frames that needed an FFT.
// --------------------------------------------------------------------------
static int computeLogMel(const float* pcm, int pcmLen,
                         const float* mel, int nMels, int nFftBins,
                         int nFft, int hop, int targetFrames,
                         float* out) {
    const int N = targetFrames * hop;          // canonical sample count (480000)
    const int P = nFft / 2;                    // reflect pad width (200)

    // Zero-pad / trim to N, then reflect-pad by P on both ends.
    std::vector<float> padded((size_t) N + 2 * P, 0.0f);
    const int copy = std::max(0, std::min(pcmLen, N));
    for (int i = 0; i < copy; ++i) padded[P + i] = pcm[i];
    for (int k = 0; k < P; ++k) {
        padded[k]         = padded[2 * P - k];       // left  reflect (no edge repeat)
        padded[P + N + k] = padded[P + N - 2 - k];   // right reflect
    }

    // Periodic Hann window (torch.hann_window default).
    std::vector<float> hann(nFft);
    for (int i = 0; i < nFft; ++i)
        hann[i] = (float) (0.5 - 0.5 * std::cos(2.0 * kPi * (double) i / (double) nFft));

    // Non-zero band of every Mel row (Slaney triangles are narrow).
    std::vector<int> lo(nMels), hi(nMels);
    for (int m = 0; m < nMels; ++m) {
        const float* row = mel + (size_t) m * nFftBins;
        int a = 0, b = nFftBins;
        while (a < nFftBins && row[a] == 0.0f) ++a;
        while (b > a && row[b - 1] == 0.0f) --b;
        lo[m] = a; hi[m] = b;
    }

    // Frames t >= firstSilent start past the last real sample (+ right reflect
    // of zeros), so their power spectrum is identically zero.
    int firstSilent = targetFrames;
    if (copy < N) firstSilent = std::min(targetFrames, (copy + P + hop - 1) / hop);

    FftPlan plan(nFft);
    std::vector<float> frame(nFft), spec(2 * (size_t) nFft), power(nFftBins);
    const float silentLog = std::log10(1e-10f);

    float globalMax = -1e30f;
    for (int t = 0; t < firstSilent; ++t) {
        const int off = t * hop;
        for (int i = 0; i < nFft; ++i) frame[i] = padded[(size_t) off + i] * hann[i];
        plan.run(frame.data(), spec.data(), 0);
        for (int b = 0; b < nFftBins; ++b) {
            const float re = spec[2 * b], im = spec[2 * b + 1];
            power[b] = re * re + im * im;       // |STFT|^2
        }
        for (int m = 0; m < nMels; ++m) {
            const float e = melDotRange(mel + (size_t) m * nFftBins, power.data(), lo[m], hi[m]);
            const float lg = std::log10(std::max(e, 1e-10f));
            out[(size_t) m * targetFrames + t] = lg;
            if (lg > globalMax) globalMax = lg;
        }
    }
    if (firstSilent < targetFrames) {
        for (int m = 0; m < nMels; ++m) {
            float* row = out + (size_t) m * targetFrames;
            std::fill(row + firstSilent, row + targetFrames, silentLog);
        }
        globalMax = std::max(globalMax, silentLog);
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
    return firstSilent;
}

// --------------------------------------------------------------------------
// Band-limited resampler: Kaiser-windowed sinc (half-width 16 taps, beta 8.6,
// cutoff 0.95 * min(1, out/in) Nyquist), evaluated from a 512-phase
// precomputed table. y[j] = sum_k x[k] * h(j*in/out - k). Used to feed the
// 22.05 kHz VITS waveform back into the 16 kHz Whisper front-end (loopback).
// --------------------------------------------------------------------------
constexpr int kHalfTaps = 16;
constexpr int kPhases = 512;
constexpr double kKaiserBeta = 8.6;

static double besselI0(double x) {
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 64; ++k) {
        term *= q / ((double) k * (double) k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

static int resampleLen(int inLen, int inRate, int outRate) {
    if (inLen <= 0 || inRate <= 0 || outRate <= 0) return 0;
    return (int) (((int64_t) inLen * outRate + inRate - 1) / inRate);
}

static int resampleSinc(const float* x, int inLen, int inRate, int outRate, float* y, int outCap) {
    const int outLen = std::min(outCap, resampleLen(inLen, inRate, outRate));
    if (outLen <= 0) return 0;
    if (inRate == outRate) { std::copy(x, x + outLen, y); return outLen; }

    const double ratio = (double) outRate / (double) inRate;
    const double fc = 0.95 * std::min(1.0, ratio);          // normalised to input Nyquist
    const int taps = 2 * kHalfTaps;
    // table[p][i]: filter tap for fractional phase p/kPhases at offset (i - kHalfTaps + 1)
    std::vector<float> table((size_t) (kPhases + 1) * taps);
    const double i0b = besselI0(kKaiserBeta);
    for (int p = 0; p <= kPhases; ++p) {
        const double frac = (double) p / kPhases;
        for (int i = 0; i < taps; ++i) {
            const double d = (double) (i - kHalfTaps + 1) - frac;   // x index minus t
            const double arg = kPi * fc * d;
            const double sinc = (std::fabs(d) < 1e-12) ? 1.0 : std::sin(arg) / arg;
            const double r = d / (double) kHalfTaps;
            const double w = (std::fabs(r) >= 1.0) ? 0.0
                             : besselI0(kKaiserBeta * std::sqrt(1.0 - r * r)) / i0b;
            table[(size_t) p * taps + i] = (float) (fc * sinc * w);
        }
    }

    const double step = (double) inRate / (double) outRate;
    for (int j = 0; j < outLen; ++j) {
        const double t = j * step;
        const int base = (int) std::floor(t);
        const double frac = t - base;
        const int p = (int) std::lround(frac * kPhases);
        const float* h = &table[(size_t) p * taps];
        const int first = base - kHalfTaps + 1;
        float acc = 0.0f;
        if (first >= 0 && first + taps <= inLen) {
            int i = 0;
#if VAANI_NEON
            float32x4_t v = vdupq_n_f32(0.0f);
            for (; i + 4 <= taps; i += 4)
                v = vmlaq_f32(v, vld1q_f32(x + first + i), vld1q_f32(h + i));
            acc = hsum(v);
#endif
            for (; i < taps; ++i) acc += x[first + i] * h[i];
        } else {
            for (int i = 0; i < taps; ++i) {
                const int k = first + i;
                if (k >= 0 && k < inLen) acc += x[k] * h[i];
            }
        }
        y[j] = acc;
    }
    return outLen;
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

// ======================= plain C ABI (host lab + JNI) ======================

VAANI_API int vaani_log_mel(const float* pcm, int pcmLen, const float* mel,
                            int nMels, int nFftBins, int nFft, int hop,
                            int targetFrames, float* out) {
    return computeLogMel(pcm, pcmLen, mel, nMels, nFftBins, nFft, hop, targetFrames, out);
}

VAANI_API int vaani_resample_len(int inLen, int inRate, int outRate) {
    return resampleLen(inLen, inRate, outRate);
}

VAANI_API int vaani_resample(const float* in, int inLen, int inRate, int outRate,
                             float* out, int outCap) {
    return resampleSinc(in, inLen, inRate, outRate, out, outCap);
}

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

JNIEXPORT jfloatArray JNICALL
Java_in_iisc_vaani_dsp_NeonAudioProcessor_resampleSinc(
        JNIEnv* env, jobject, jfloatArray input, jint inRate, jint outRate) {
    const jsize n = env->GetArrayLength(input);
    const int outLen = resampleLen((int) n, inRate, outRate);
    jfloat* in = env->GetFloatArrayElements(input, nullptr);
    std::vector<float> out(std::max(0, outLen));
    const int wrote = resampleSinc(in, (int) n, inRate, outRate, out.data(), outLen);
    env->ReleaseFloatArrayElements(input, in, JNI_ABORT);
    jfloatArray res = env->NewFloatArray(wrote);
    if (wrote > 0) env->SetFloatArrayRegion(res, 0, wrote, out.data());
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
