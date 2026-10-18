package `in`.iisc.vaani.audio

import android.annotation.SuppressLint
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.util.Log
import `in`.iisc.vaani.dsp.NeonAudioProcessor
import `in`.iisc.vaani.neural.LanguageMode
import `in`.iisc.vaani.neural.SttResult
import `in`.iisc.vaani.neural.TtsAudio
import `in`.iisc.vaani.neural.VaaniSttEngine
import `in`.iisc.vaani.neural.VaaniTtsEngine
import `in`.iisc.vaani.neural.VaaniVadEngine
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * End-to-end on-device voice loop.
 *
 * Capture path:
 *   AudioRecord(VOICE_RECOGNITION, 16 kHz mono PCM16), 512-sample (32 ms) chunks
 *     -> NEON RMS (live waveform)  -> Silero VAD on the raw signal
 *     -> onSpeechStart()  (2 speech chunks: the LLM HTTP/2 socket is pre-warmed here)
 *     -> utterance = ~380 ms pre-roll + speech, endpoint after [endpointSilenceMs] of
 *        trailing silence (hysteresis 0.5 / 0.35), hard cap [MAX_UTTERANCE_MS]
 *     -> onEndOfSpeech(t_end)   <- the end-to-end latency clock starts here
 *     -> 4-thread KV-cached Whisper STT with step-0 EN/HI language ID
 *     -> onTranscript(SttResult, t_end)
 *
 * Playback path (clause-level streaming TTS, three stages so clause N+1 is synthesised
 * while clause N is still playing):
 *   speak(clause) -> [synth worker: VITS] -> [PCM queue] -> [writer: low-latency AudioTrack]
 *   onFirstAudio(t) fires when the first PCM sample of a turn is handed to the AudioTrack.
 *   [interrupt] bumps the turn id, flushes the track and drops queued clauses (barge-in).
 *
 * Pre-emphasis is deliberately NOT applied: neither Whisper nor Silero was trained on it.
 */
class AudioPipelineManager(
    private val vad: VaaniVadEngine,
    private val stt: VaaniSttEngine,
    private val tts: VaaniTtsEngine,
    private val scope: CoroutineScope,
) {
    var onSpeechStart: () -> Unit = {}
    var onEndOfSpeech: (endNanos: Long, audioSec: Double) -> Unit = { _, _ -> }
    var onTranscript: (SttResult, endNanos: Long) -> Unit = { _, _ -> }
    var onNoSpeech: () -> Unit = {}
    var onMicLevel: (Float) -> Unit = {}
    var onPlaybackLevel: (Float) -> Unit = {}
    var onClauseSynthesized: (clause: String, audio: TtsAudio) -> Unit = { _, _ -> }
    var onFirstAudio: (nanos: Long) -> Unit = {}
    var onPlaybackIdle: () -> Unit = {}
    var onError: (String) -> Unit = {}

    @Volatile var languageMode: LanguageMode = LanguageMode.AUTO
    @Volatile var endpointSilenceMs: Int = 500

    private val listening = AtomicBoolean(false)
    private var captureJob: Job? = null

    // ======================= CAPTURE =========================================

    val isListening: Boolean get() = listening.get()

    @SuppressLint("MissingPermission") // caller holds RECORD_AUDIO
    fun startListening() {
        if (!listening.compareAndSet(false, true)) return
        captureJob = scope.launch(Dispatchers.IO) {
            val minBuf = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            ).coerceAtLeast(VaaniVadEngine.CHUNK_SAMPLES * 8)
            val recorder = try {
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf * 2,
                )
            } catch (e: Exception) {
                listening.set(false); onError("Microphone unavailable: ${e.message}"); return@launch
            }
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release(); listening.set(false); onError("Microphone could not be opened"); return@launch
            }

            val chunk = ShortArray(VaaniVadEngine.CHUNK_SAMPLES)
            val preRoll = ArrayDeque<FloatArray>()
            val utterance = FloatBuilder(SAMPLE_RATE * 8)
            var speechRun = 0
            var started = false
            var silentMs = 0
            val chunkMs = VaaniVadEngine.CHUNK_SAMPLES * 1000 / SAMPLE_RATE
            val t0 = System.nanoTime()
            vad.reset()
            try {
                recorder.startRecording()
                while (isActive && listening.get()) {
                    val n = recorder.read(chunk, 0, chunk.size, AudioRecord.READ_BLOCKING)
                    if (n <= 0) continue
                    val frame = FloatArray(VaaniVadEngine.CHUNK_SAMPLES) { if (it < n) chunk[it] / 32768f else 0f }
                    onMicLevel(NeonAudioProcessor.rms(frame))
                    val p = vad.process(frame)
                    if (!started) {
                        preRoll.addLast(frame)
                        if (preRoll.size > PRE_ROLL_CHUNKS) preRoll.removeFirst()
                        speechRun = if (p >= SPEECH_ON) speechRun + 1 else 0
                        if (speechRun >= 2) {
                            started = true
                            onSpeechStart()
                            preRoll.forEach { utterance.add(it) }
                            preRoll.clear()
                        } else if ((System.nanoTime() - t0) / 1_000_000 > NO_SPEECH_TIMEOUT_MS) {
                            listening.set(false)
                            onNoSpeech()
                            break
                        }
                    } else {
                        utterance.add(frame)
                        silentMs = if (p >= SPEECH_OFF) 0 else silentMs + chunkMs
                        if (silentMs >= endpointSilenceMs || utterance.size >= SAMPLE_RATE * MAX_UTTERANCE_MS / 1000) {
                            val endNanos = System.nanoTime()
                            listening.set(false)
                            runCatching { recorder.stop() }
                            val keepTail = (silentMs - TAIL_KEEP_MS).coerceAtLeast(0) * SAMPLE_RATE / 1000
                            val pcm = utterance.toArray(dropTail = keepTail)
                            onEndOfSpeech(endNanos, pcm.size.toDouble() / SAMPLE_RATE)
                            transcribe(pcm, endNanos)
                            break
                        }
                    }
                }
            } catch (e: Exception) {
                onError("Capture failed: ${e.message}")
            } finally {
                listening.set(false)
                runCatching { recorder.stop() }
                runCatching { recorder.release() }
            }
        }
    }

    fun stopListening() {
        listening.set(false)
        captureJob = null
    }

    /** Runs STT on an in-memory 16 kHz buffer (loopback benchmark path). */
    fun transcribeBuffer(pcm16k: FloatArray, endNanos: Long = System.nanoTime()) {
        scope.launch(Dispatchers.Default) { transcribe(pcm16k, endNanos) }
    }

    private fun transcribe(pcm: FloatArray, endNanos: Long) {
        try {
            val result = stt.transcribe(pcm, languageMode)
            onTranscript(result, endNanos)
        } catch (e: Exception) {
            Log.e(TAG, "STT failed", e)                    // details stay in logcat, never in the feed
            onError("Speech recognition failed. Please try again.")
        }
    }

    // ======================= PLAYBACK (streaming TTS) ========================

    private sealed class ClauseItem {
        class Text(val turn: Int, val text: String) : ClauseItem()
        class End(val turn: Int) : ClauseItem()
    }

    private sealed class PcmItem {
        class Chunk(val turn: Int, val pcm: ShortArray, val level: Float) : PcmItem()
        class End(val turn: Int, val done: CompletableDeferred<Unit>?) : PcmItem()
    }

    private val turn = AtomicInteger(0)
    private val clauses = Channel<ClauseItem>(Channel.UNLIMITED)
    private val pcmQueue = Channel<PcmItem>(Channel.UNLIMITED)
    private var audioTrack: AudioTrack? = null
    private var synthJob: Job? = null
    private var writerJob: Job? = null
    @Volatile private var firstAudioPending = false

    /** Starts the synth + writer coroutines (idempotent). */
    fun startPlaybackLoop() {
        if (writerJob?.isActive == true) return
        val track = buildAudioTrack(tts.sampleRate).also { audioTrack = it }
        synthJob = scope.launch(Dispatchers.Default) {
            for (item in clauses) {
                when (item) {
                    is ClauseItem.Text -> {
                        if (item.turn != turn.get()) continue
                        // A failed clause is skipped and logged; the turn keeps playing the rest.
                        val audio = try { tts.synthesize(item.text) } catch (e: Exception) {
                            Log.w(TAG, "TTS clause skipped", e); continue
                        }
                        if (audio.failedSentences > 0) Log.w(TAG, "TTS: ${audio.failedSentences} sentence(s) skipped after retry")
                        if (item.turn != turn.get() || audio.samples.isEmpty()) continue
                        onClauseSynthesized(item.text, audio)
                        enqueuePcm(item.turn, audio.samples)
                    }
                    is ClauseItem.End -> pcmQueue.trySend(PcmItem.End(item.turn, null))
                }
            }
        }
        writerJob = scope.launch(Dispatchers.IO) {
            track.play()
            // Frames are counted per turn against the head position at the turn's first
            // write: flush() on barge-in resets the hardware head, so a running total would
            // make every later turn wait out the drain timeout.
            fun head() = track.playbackHeadPosition.toLong() and 0xffffffffL
            var currentTurn = -1
            var headAtTurnStart = 0L
            var turnFrames = 0L
            for (item in pcmQueue) {
                when (item) {
                    is PcmItem.Chunk -> {
                        if (item.turn != turn.get()) continue
                        if (item.turn != currentTurn) {
                            currentTurn = item.turn; headAtTurnStart = head(); turnFrames = 0
                        }
                        var off = 0
                        while (off < item.pcm.size && item.turn == turn.get()) {
                            val n = track.write(item.pcm, off, item.pcm.size - off)
                            if (n <= 0) break
                            if (firstAudioPending) { firstAudioPending = false; onFirstAudio(System.nanoTime()) }
                            off += n
                            turnFrames += n
                        }
                        onPlaybackLevel(item.level)
                    }
                    is PcmItem.End -> {
                        // Let the hardware drain what this turn wrote before reporting idle.
                        val waitUntil = System.nanoTime() + 3_000_000_000L
                        while (item.turn == turn.get() && item.turn == currentTurn && System.nanoTime() < waitUntil) {
                            if (head() - headAtTurnStart >= turnFrames) break
                            delay(15)
                        }
                        onPlaybackLevel(0f)
                        item.done?.complete(Unit)
                        if (item.turn == turn.get() && item.done == null) onPlaybackIdle()
                    }
                }
            }
        }
    }

    private fun enqueuePcm(turnId: Int, samples: FloatArray) {
        val pcm = NeonAudioProcessor.toPcm16(samples)
        val step = CHUNK_FRAMES
        var off = 0
        while (off < pcm.size) {
            val end = minOf(pcm.size, off + step)
            val part = pcm.copyOfRange(off, end)
            var acc = 0.0
            for (s in part) acc += (s / 32768.0) * (s / 32768.0)
            pcmQueue.trySend(PcmItem.Chunk(turnId, part, kotlin.math.sqrt(acc / part.size).toFloat()))
            off = end
        }
    }

    /** Starts a new spoken reply; the next PCM write reports [onFirstAudio]. */
    fun beginSpeechTurn(): Int {
        firstAudioPending = true
        return turn.get()
    }

    /** Queues one acoustic-Devanagari clause for synthesis + playback in the current turn. */
    fun speak(clause: String) {
        clauses.trySend(ClauseItem.Text(turn.get(), clause))
    }

    /** No more clauses this turn: [onPlaybackIdle] fires once everything has played. */
    fun endSpeechTurn() {
        clauses.trySend(ClauseItem.End(turn.get()))
    }

    /** Plays already-synthesised audio (e.g. the loopback prompt) and suspends until done. */
    suspend fun playAndAwait(audio: TtsAudio) {
        startPlaybackLoop()
        val done = CompletableDeferred<Unit>()
        val id = turn.get()
        enqueuePcm(id, audio.samples)
        pcmQueue.trySend(PcmItem.End(id, done))
        done.await()
    }

    /** Instant barge-in: invalidate the turn, flush the track, drop pending clauses. */
    fun interrupt() {
        turn.incrementAndGet()
        firstAudioPending = false
        while (clauses.tryReceive().isSuccess) { /* drop */ }
        audioTrack?.let {
            runCatching { it.pause() }
            runCatching { it.flush() }
            runCatching { it.play() }
        }
        onPlaybackLevel(0f)
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
        interrupt()
        synthJob?.cancel(); writerJob?.cancel()
        clauses.close(); pcmQueue.close()
        runCatching { audioTrack?.stop() }
        runCatching { audioTrack?.release() }
        audioTrack = null
    }

    /** Growable float buffer without boxing. */
    private class FloatBuilder(capacity: Int) {
        private var data = FloatArray(capacity)
        var size = 0
            private set

        fun add(x: FloatArray) {
            if (size + x.size > data.size) data = data.copyOf(maxOf(data.size * 2, size + x.size))
            x.copyInto(data, size)
            size += x.size
        }

        fun toArray(dropTail: Int = 0): FloatArray = data.copyOf((size - dropTail).coerceAtLeast(0))
    }

    companion object {
        private const val TAG = "VaaniAudio"
        const val SAMPLE_RATE = 16_000
        private const val PRE_ROLL_CHUNKS = 12          // ~384 ms before VAD onset
        private const val SPEECH_ON = 0.5f
        private const val SPEECH_OFF = 0.35f
        private const val TAIL_KEEP_MS = 200
        private const val MAX_UTTERANCE_MS = 15_000
        private const val NO_SPEECH_TIMEOUT_MS = 8_000L
        private const val CHUNK_FRAMES = 1024           // ~46 ms @ 22.05 kHz -> smooth level meter
    }
}
