package `in`.iisc.vaani.audio

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import `in`.iisc.vaani.codec.HinglishCodec
import `in`.iisc.vaani.dsp.NeonAudioProcessor
import `in`.iisc.vaani.neural.SttResult
import `in`.iisc.vaani.neural.VaaniSttEngine
import `in`.iisc.vaani.neural.VaaniTtsEngine
import `in`.iisc.vaani.neural.VaaniVadEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/**
 * End-to-end on-device voice loop.
 *
 * Capture path:
 *   AudioRecord(VOICE_RECOGNITION, 16 kHz mono PCM16)
 *     -> NEON pre-emphasis + RMS
 *     -> Silero VAD with 320 ms trailing-silence auto-endpointing
 *     -> onSpeechStart() (Session 2 uses this to pre-warm the LLM HTTP/2 socket)
 *     -> 4-thread Whisper STT
 *     -> HinglishCodec.normalizeSttTranscript()
 *     -> onTranscript(SttResult) with exact sttLatencyMs / sttRtf
 *
 * Playback path (Micro-Clause Streaming TTS):
 *   Channel<String> of clauses
 *     -> VITS synthesis on a background coroutine (4 threads)
 *     -> NEON SIMD float -> PCM16
 *     -> pre-allocated low-latency AudioTrack (MODE_STREAM, LOW_LATENCY)
 *   [interrupt] flushes the track and cancels synthesis for instant Barge-In.
 */
class AudioPipelineManager(
    private val vad: VaaniVadEngine,
    private val stt: VaaniSttEngine,
    private val tts: VaaniTtsEngine,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob()),
    private val languageHint: String = "hi",
    private val applyPreEmphasis: Boolean = true,
) {
    var onSpeechStart: () -> Unit = {}
    var onTranscript: (SttResult, String) -> Unit = { _, _ -> }

    @Volatile var lastSttLatencyMs: Long = 0L; private set
    @Volatile var lastSttRtf: Double = 0.0; private set

    private val listening = AtomicBoolean(false)
    private var captureJob: Job? = null

    private val clauseChannel = Channel<String>(capacity = Channel.UNLIMITED)
    private var playbackJob: Job? = null
    private var audioTrack: AudioTrack? = null
    @Volatile private var bargeIn = false

    // ======================= CAPTURE =========================================

    @SuppressLint("MissingPermission") // caller must hold RECORD_AUDIO
    fun startListening() {
        if (!listening.compareAndSet(false, true)) return
        captureJob = scope.launch {
            val minBuf = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            ).coerceAtLeast(VaaniVadEngine.CHUNK_SAMPLES * 2)

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                minBuf * 2,
            )

            val chunk = ShortArray(VaaniVadEngine.CHUNK_SAMPLES)
            val utterance = ArrayList<Float>(SAMPLE_RATE * 10)
            var speaking = false
            vad.reset()

            try {
                recorder.startRecording()
                while (isActive && listening.get()) {
                    val n = recorder.read(chunk, 0, chunk.size)
                    if (n <= 0) continue

                    val frame = FloatArray(n) { chunk[it] / 32768f }
                    val processed = if (applyPreEmphasis) NeonAudioProcessor.preEmphasize(frame) else frame

                    val endpoint = vad.processAndCheckEndpoint(processed)
                    if (!speaking && NeonAudioProcessor.rmsEnergy(processed) > SPEECH_RMS_GATE) {
                        speaking = true
                        onSpeechStart()
                    }
                    if (speaking) utterance.addAll(processed.toList())

                    if (endpoint && utterance.isNotEmpty()) {
                        val pcm = utterance.toFloatArray()
                        utterance.clear(); speaking = false; vad.reset()
                        transcribeUtterance(pcm)
                    }
                }
            } finally {
                runCatching { recorder.stop() }
                runCatching { recorder.release() }
            }
        }
    }

    private fun transcribeUtterance(pcm: FloatArray) {
        val result = stt.transcribe(pcm, languageHint)
        lastSttLatencyMs = result.sttLatencyMs
        lastSttRtf = result.sttRtf
        val normalized = HinglishCodec.normalizeSttTranscript(result.text, languageHint)
        onTranscript(result, normalized)
    }

    fun stopListening() {
        listening.set(false)
        captureJob = null
    }

    // ======================= PLAYBACK (streaming TTS) ========================

    /** Starts the clause-consumer coroutine if not already running. */
    fun startPlaybackLoop() {
        if (playbackJob?.isActive == true) return
        audioTrack = buildAudioTrack(tts.sampleRate)
        playbackJob = scope.launch {
            audioTrack?.play()
            for (clause in clauseChannel) {
                if (bargeIn) { drainPending(); continue }
                val devanagari = HinglishCodec.toAcousticDevanagariScript(clause)
                val wave = tts.synthesize(devanagari)
                if (wave.isEmpty()) continue
                val pcm16 = NeonAudioProcessor.toPcm16(wave)
                var off = 0
                while (off < pcm16.size && !bargeIn) {
                    val wrote = audioTrack?.write(pcm16, off, pcm16.size - off) ?: break
                    if (wrote <= 0) break
                    off += wrote
                }
            }
        }
    }

    /** Queues a clause for streaming synthesis + playback. */
    fun speak(clause: String) {
        bargeIn = false
        clauseChannel.trySend(clause)
    }

    /** Instant Barge-In: stop current audio, flush the track, drop pending clauses. */
    fun interrupt() {
        bargeIn = true
        audioTrack?.let {
            runCatching { it.pause() }
            runCatching { it.flush() }
            runCatching { it.play() }
        }
        drainPending()
    }

    private fun drainPending() {
        while (clauseChannel.tryReceive().isSuccess) { /* discard */ }
    }

    private fun buildAudioTrack(sampleRate: Int): AudioTrack {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(sampleRate / 10 * 2)

        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(minBuf)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()
    }

    // ======================= LIFECYCLE =======================================

    fun release() {
        stopListening()
        playbackJob = null
        runCatching { audioTrack?.stop() }
        runCatching { audioTrack?.release() }
        audioTrack = null
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        private const val SPEECH_RMS_GATE = 0.004f
    }
}
