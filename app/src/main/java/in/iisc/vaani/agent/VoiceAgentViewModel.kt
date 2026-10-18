package `in`.iisc.vaani.agent

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import `in`.iisc.vaani.audio.AudioPipelineManager
import `in`.iisc.vaani.codec.HinglishCodec
import `in`.iisc.vaani.codec.IndicPhonemizer
import `in`.iisc.vaani.dsp.NeonAudioProcessor
import `in`.iisc.vaani.llm.ChatRequest
import `in`.iisc.vaani.llm.ChatTurn
import `in`.iisc.vaani.llm.LlmEvent
import `in`.iisc.vaani.llm.LlmModel
import `in`.iisc.vaani.llm.LlmProvider
import `in`.iisc.vaani.llm.LlmSettings
import `in`.iisc.vaani.llm.LlmStats
import `in`.iisc.vaani.llm.LlmStreamingClient
import `in`.iisc.vaani.llm.PromptBuilder
import `in`.iisc.vaani.llm.ResponseStreamProcessor
import `in`.iisc.vaani.neural.LanguageMode
import `in`.iisc.vaani.neural.SttResult
import `in`.iisc.vaani.runtime.VaaniRuntime
import `in`.iisc.vaani.tools.ActionType
import `in`.iisc.vaani.tools.AgentIntent
import `in`.iisc.vaani.tools.AlarmItem
import `in`.iisc.vaani.tools.LocalIntents
import `in`.iisc.vaani.tools.LocalResponder
import `in`.iisc.vaani.tools.LocalToolManager
import `in`.iisc.vaani.tools.ReminderItem
import `in`.iisc.vaani.tools.ScheduleGenerator
import `in`.iisc.vaani.tools.ToolAction
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicLong

/**
 * Voice-agent orchestrator: mic/loopback -> on-device STT -> streaming cloud LLM (or the
 * on-device LocalResponder) -> clause-level on-device TTS, plus grounded tool actions,
 * action cards and the research telemetry HUD.
 *
 * Latency accounting per turn (all System.nanoTime on the phone):
 *   STT            = mel + encoder + step-0 LID + KV decode           (VaaniSttEngine)
 *   LLM TTFT       = request issued -> first content token             (LlmStreamingClient)
 *   TTS first-audio= first clause queued -> first PCM handed to AudioTrack
 *   End-to-end     = VAD end-of-speech trigger -> first PCM sample written to AudioTrack
 */
class VoiceAgentViewModel(app: Application) : AndroidViewModel(app) {

    private val settings = LlmSettings(app)
    private val llm = LlmStreamingClient()
    private val tools = LocalToolManager(app, llm.http)
    private var runtime: VaaniRuntime? = null
    private var pipeline: AudioPipelineManager? = null

    private val _ui = MutableStateFlow(
        AgentUiState(
            languageMode = settings.languageMode,
            model = settings.effectiveModel(),
            baseModel = settings.model,
            modelIdOverride = settings.modelIdOverride,
            cloudEnabled = settings.cloudEnabled,
            cloudReady = settings.cascade().isNotEmpty(),
        ),
    )
    val ui: StateFlow<AgentUiState> = _ui.asStateFlow()

    /** Live RMS for the 60 fps waveform: microphone while listening, TTS while speaking. */
    val level = MutableStateFlow(0f)

    private val ids = AtomicLong(1)
    private val history = ArrayDeque<ChatTurn>()
    private var llmJob: Job? = null

    private class Turn(val source: String, val benchmark: BenchmarkPrompt?) {
        @Volatile var endOfSpeechNanos = 0L
        @Volatile var firstClauseNanos = 0L
        @Volatile var assistantId = 0L
        @Volatile var lang = "en"
        var telemetry = Telemetry(source = source)
    }

    @Volatile private var turn: Turn? = null

    init {
        viewModelScope.launch { loadRuntime() }
        viewModelScope.launch { runCatching { tools.refreshWeather() } }
        viewModelScope.launch { tools.forecast.collect { f -> _ui.update { it.copy(forecast = f) } } }
        viewModelScope.launch { tools.alarms.collect { a -> _ui.update { it.copy(alarms = a) } } }
        viewModelScope.launch { tools.reminders.collect { r -> _ui.update { it.copy(reminders = r) } } }
    }

    // ======================= runtime =========================================

    private suspend fun loadRuntime() {
        try {
            val rt = VaaniRuntime.load(getApplication()) { msg -> _ui.update { it.copy(statusDetail = msg) } }
            runtime = rt
            val p = AudioPipelineManager(rt.vad, rt.stt, rt.tts, viewModelScope)
            wire(p)
            p.languageMode = settings.languageMode
            p.startPlaybackLoop()
            pipeline = p
            val r = rt.report
            _ui.update {
                it.copy(
                    state = AgentState.IDLE,
                    runtimeReady = true,
                    statusDetail = "ARM64 NEON + ONNX INT8 Ready",
                    runtimeSummary = "models ${"%.0f".format(r.modelMb)} MB • load ${r.totalMs} ms" +
                        (if (r.kvCache) " • KV-cache" else "") + " • lexicon ${r.lexiconWords / 1000}k",
                )
            }
            Log.i(TAG, "runtime ready: $r")
            prewarm()
        } catch (t: Throwable) {
            Log.e(TAG, "runtime load failed", t)
            _ui.update { it.copy(state = AgentState.ERROR, statusDetail = "Model load failed: ${t.message}") }
        }
    }

    private fun wire(p: AudioPipelineManager) {
        p.onMicLevel = { level.value = it }
        p.onPlaybackLevel = { level.value = it }
        p.onSpeechStart = { prewarm() }
        p.onEndOfSpeech = { endNanos, _ ->
            turn?.endOfSpeechNanos = endNanos
            setState(AgentState.TRANSCRIBING)
        }
        p.onNoSpeech = {
            level.value = 0f
            setState(AgentState.IDLE)
            _ui.update { it.copy(highlightedPrompt = null) }
            notice("No speech detected — tap the mic and speak again.")
        }
        p.onTranscript = { r, endNanos -> onTranscript(r, endNanos) }
        p.onClauseSynthesized = { _, audio ->
            turn?.let { t -> synchronized(t) { if (t.telemetry.ttsRtf == null) t.telemetry = t.telemetry.copy(ttsRtf = audio.rtf) } }
        }
        p.onFirstAudio = { nanos -> onFirstAudio(nanos) }
        p.onPlaybackIdle = {
            level.value = 0f
            val s = _ui.value.state
            if (s == AgentState.SPEAKING || s == AgentState.STREAMING) setState(AgentState.IDLE)
            _ui.update { it.copy(busyPrompt = null, highlightedPrompt = null) }
        }
        p.onError = { msg ->
            notice(msg)
            setState(AgentState.IDLE)
            _ui.update { it.copy(busyPrompt = null) }
        }
    }

    // ======================= user actions ====================================

    fun setMicPermission(granted: Boolean) = _ui.update { it.copy(micPermission = granted) }

    fun onMicTap() {
        when (_ui.value.state) {
            AgentState.IDLE -> startMicTurn(_ui.value.highlightedPrompt?.let { BENCHMARK_PROMPTS[it] })
            AgentState.LISTENING -> { pipeline?.stopListening(); level.value = 0f; setState(AgentState.IDLE) }
            AgentState.STREAMING, AgentState.SPEAKING -> bargeIn()
            else -> Unit
        }
    }

    /** Chip action 1: highlight the prompt and listen for the user saying it live. */
    fun speakBenchmark(index: Int) {
        _ui.update { it.copy(highlightedPrompt = index) }
        when (_ui.value.state) {
            AgentState.IDLE -> startMicTurn(BENCHMARK_PROMPTS[index])
            AgentState.STREAMING, AgentState.SPEAKING -> { stopSpeaking(); startMicTurn(BENCHMARK_PROMPTS[index]) }
            else -> Unit
        }
    }

    /** Chip action 2: TTS the prompt on-device, feed that audio to STT -> LLM -> TTS. */
    fun runLoopback(index: Int) {
        val rt = runtime ?: return
        val p = pipeline ?: return
        if (_ui.value.state != AgentState.IDLE) return
        val prompt = BENCHMARK_PROMPTS[index]
        val t = Turn("loopback", prompt)
        turn = t
        _ui.update { it.copy(busyPrompt = index, highlightedPrompt = index, statusDetail = "Loopback: synthesising prompt on-device") }
        setState(AgentState.LISTENING)
        viewModelScope.launch(Dispatchers.Default) {
            try {
                prewarm()                                            // = onSpeechStart
                val acoustic = HinglishCodec.toAcousticDevanagariScript(prompt.text, prompt.lang)
                val audio = rt.tts.synthesize(acoustic)
                p.playAndAwait(audio)                                // the phone "speaks" the prompt
                val pcm16 = NeonAudioProcessor.resample(audio.samples, audio.sampleRate, AudioPipelineManager.SAMPLE_RATE)
                val end = System.nanoTime()                          // = VAD end-of-speech
                t.endOfSpeechNanos = end
                _ui.update { it.copy(statusDetail = "ARM64 NEON + ONNX INT8 Ready") }
                setState(AgentState.TRANSCRIBING)
                p.transcribeBuffer(pcm16, end)
            } catch (e: Exception) {
                notice("Loopback failed: ${e.message}")
                setState(AgentState.IDLE)
                _ui.update { it.copy(busyPrompt = null) }
            }
        }
    }

    fun replay(assistantId: Long) {
        val p = pipeline ?: return
        val item = _ui.value.feed.firstOrNull { it.id == assistantId } as? FeedItem.Assistant ?: return
        if (item.acoustic.isBlank()) return
        stopSpeaking()
        p.beginSpeechTurn()
        p.speak(item.acoustic)
        p.endSpeechTurn()
        setState(AgentState.SPEAKING)
    }

    fun cycleLanguageMode() {
        val next = LanguageMode.values()[(_ui.value.languageMode.ordinal + 1) % LanguageMode.values().size]
        settings.languageMode = next
        pipeline?.languageMode = next
        _ui.update { it.copy(languageMode = next) }
    }

    fun saveSettings(
        model: LlmModel, groqKey: String?, geminiKey: String?, mode: LanguageMode, cloudEnabled: Boolean,
        modelIdOverride: String,
    ) {
        groqKey?.let { settings.groqKey = it }
        geminiKey?.let { settings.geminiKey = it }
        settings.model = model
        settings.modelIdOverride = modelIdOverride
        settings.languageMode = mode
        settings.cloudEnabled = cloudEnabled
        pipeline?.languageMode = mode
        llm.resetHealth()                      // new keys / model: give every rung a fresh chance
        _ui.update {
            it.copy(
                model = settings.effectiveModel(), baseModel = settings.model, modelIdOverride = settings.modelIdOverride,
                languageMode = mode, cloudEnabled = cloudEnabled, cloudReady = settings.cascade().isNotEmpty(),
            )
        }
        prewarm()
    }

    fun keyStatus(): Map<String, String> = mapOf(
        "groq" to "${settings.maskedKey(LlmProvider.GROQ)} (${settings.keySource(LlmProvider.GROQ)})",
        "gemini" to "${settings.maskedKey(LlmProvider.GEMINI)} (${settings.keySource(LlmProvider.GEMINI)})",
    )

    fun commitAlarm(id: Long) {
        val a = _ui.value.alarms.firstOrNull { it.id == id } ?: return
        if (!tools.commitAlarmToClock(a)) notice("No Clock app found to receive the alarm.")
    }

    fun addReminderToCalendar(id: Long) {
        val r = _ui.value.reminders.firstOrNull { it.id == id } ?: return
        if (!tools.addReminderToCalendar(r)) notice("No Calendar app found to receive the reminder.")
    }

    fun clearFeed() = _ui.update { it.copy(feed = emptyList()) }

    // ======================= turn pipeline ===================================

    private fun startMicTurn(benchmark: BenchmarkPrompt?) {
        val p = pipeline ?: return
        if (!_ui.value.micPermission) { notice("Microphone permission is required for live speech."); return }
        turn = Turn("mic", benchmark)
        setState(AgentState.LISTENING)
        p.startListening()
    }

    private fun stopSpeaking() {
        llmJob?.cancel()
        pipeline?.interrupt()
        turn?.let { t -> updateAssistant(t.assistantId) { it.copy(streaming = false, note = it.note ?: "interrupted") } }
    }

    private fun bargeIn() {
        stopSpeaking()
        startMicTurn(null)
    }

    private fun onTranscript(r: SttResult, endNanos: Long) {
        val t = turn ?: return
        t.endOfSpeechNanos = endNanos
        val display = HinglishCodec.displayTranscript(r.text, r.language)
        if (display.isBlank()) {
            notice("Didn't catch that — please try again.")
            setState(AgentState.IDLE)
            _ui.update { it.copy(busyPrompt = null) }
            return
        }
        t.lang = r.language
        synchronized(t) {
            t.telemetry = t.telemetry.copy(
                sttMs = r.sttLatencyMs, sttRtf = r.sttRtf, melMs = r.melMs, encoderMs = r.encoderMs,
                decodeMs = r.decodeMs, tokensPerSec = r.tokensPerSecond, lid = r.detectedLanguage, pHindi = r.pHindi,
                llmLabel = activeEngineLabel(),
            )
        }
        val wer = t.benchmark?.let { Metrics.wordErrorRate(it.text, display) }
        addFeed(FeedItem.User(ids.getAndIncrement(), display, r.language, r.sttLatencyMs,
            r.text.takeIf { it.trim() != display }, t.source, t.benchmark, wer))
        _ui.update { it.copy(telemetry = t.telemetry) }
        respond(t, display, r.text)
    }

    /** The rung that will answer first: the primary unless it is cooling down after a failure. */
    private fun activeEngineLabel(): String {
        if (!settings.cloudEnabled) return "On-device LocalResponder"
        return llm.liveTargets(settings.cascade()).firstOrNull()?.model?.label ?: "On-device LocalResponder"
    }

    private fun respond(t: Turn, transcript: String, rawAsr: String) {
        val p = pipeline ?: return
        setState(AgentState.STREAMING)
        p.beginSpeechTurn()
        val snapshot = tools.snapshot()
        val understood = "$transcript $rawAsr"
        val intents = LocalIntents.detect(understood)
        val assistantId = ids.getAndIncrement()
        t.assistantId = assistantId
        val targets = settings.cascade()
        val useCloud = settings.cloudEnabled && targets.isNotEmpty()
        val firstLabel = activeEngineLabel()
        addFeed(FeedItem.Assistant(assistantId, "", t.lang, engine = firstLabel))

        llmJob = viewModelScope.launch(Dispatchers.Default) {
            val acoustic = StringBuilder()
            val processor = ResponseStreamProcessor(
                onClause = { clause -> speakClause(t, clause, acoustic) },
                onDisplay = { text -> updateAssistant(assistantId) { it.copy(text = text) } },
            )
            var stats: LlmStats? = null
            var error: String? = null
            var engine = firstLabel
            if (useCloud) {
                val request = ChatRequest(
                    targets = targets,
                    systemPrompt = PromptBuilder.systemPrompt(snapshot),
                    userMessage = PromptBuilder.userMessage(transcript, t.lang, rawAsr.takeIf { it != transcript }),
                    history = history.toList(),
                )
                llm.stream(request).collect { ev ->
                    when (ev) {
                        is LlmEvent.Token -> processor.push(ev.text)
                        is LlmEvent.Done -> stats = ev.stats
                        is LlmEvent.Error -> error = ev.message
                    }
                }
            }
            val local = if (!useCloud || (error != null && processor.displayText.isBlank())) {
                LocalResponder.respond(understood, t.lang, snapshot).also { reply ->
                    processor.push(reply.text)
                    engine = "On-device LocalResponder" + if (error != null) " (cloud error)" else ""
                }
            } else null
            processor.finish()
            p.endSpeechTurn()

            // Grounded actions: the LLM's line-1 tag, else the on-device intent backstop.
            val actions = local?.actions ?: processor.actions.ifEmpty { backstopActions(intents, understood) }
            val cardIntents = intents + (local?.intents ?: emptySet())
            stats?.let { s ->
                engine = if (s.fellBack) "${s.modelLabel} (failover)" else s.modelLabel
                synchronized(t) {
                    t.telemetry = t.telemetry.copy(
                        llmTtftMs = s.ttftMs, llmTotalMs = s.totalMs, connectionReused = s.reusedConnection,
                        connectMs = s.connectMs, llmLabel = engine,
                    )
                }
            } ?: synchronized(t) { t.telemetry = t.telemetry.copy(llmLabel = engine) }

            val reply = processor.displayText
            if (reply.isNotBlank()) {
                history.addLast(ChatTurn(transcript, reply))
                while (history.size > 2) history.removeFirst()
            }
            updateAssistant(assistantId) {
                it.copy(
                    text = reply.ifBlank { error ?: "(no reply)" },
                    acoustic = acoustic.toString().trim(),
                    ipa = IndicPhonemizer.textToIpa(acoustic.toString()),
                    streaming = false,
                    engine = engine,
                    telemetry = t.telemetry,
                    note = error ?: stats?.takeIf { it.fellBack }?.let { "failover: " + it.failovers.joinToString("; ") },
                )
            }
            for (a in actions) {
                when (val item = tools.apply(a, understood)) {
                    is AlarmItem -> addFeed(FeedItem.Card(ids.getAndIncrement(), ActionCard.Alarm(item.id)))
                    is ReminderItem -> addFeed(FeedItem.Card(ids.getAndIncrement(), ActionCard.Reminder(item.id)))
                }
            }
            if (AgentIntent.WEATHER in cardIntents) {
                addFeed(FeedItem.Card(ids.getAndIncrement(), ActionCard.Weather(tools.forecast.value)))
            }
            if (AgentIntent.SCHEDULE in cardIntents && actions.isEmpty()) {
                val now = tools.now().toLocalTime()
                addFeed(FeedItem.Card(ids.getAndIncrement(), ActionCard.Schedule(ScheduleGenerator.withStatus(tools.schedule(), now))))
            }
            _ui.update { it.copy(telemetry = t.telemetry) }
            if (acoustic.isBlank()) {
                setState(AgentState.IDLE)
                _ui.update { it.copy(busyPrompt = null, highlightedPrompt = null) }
            }
            logTelemetry(t, transcript, reply, stats)
        }
    }

    private fun backstopActions(intents: Set<AgentIntent>, text: String): List<ToolAction> {
        val out = ArrayList<ToolAction>()
        if (AgentIntent.ALARM in intents) {
            LocalIntents.parseTime(text, preferMorning = true)?.let { (h, m) ->
                out.add(ToolAction(ActionType.SET_ALARM, h, m, if (h < 12) "Morning Alarm" else "Alarm", "local-backstop"))
            }
        } else if (AgentIntent.REMINDER in intents) {
            LocalIntents.parseTime(text, preferMorning = false)?.let { (h, m) ->
                val label = if ("meeting" in text.lowercase()) "Meeting" else "Reminder"
                out.add(ToolAction(ActionType.SET_REMINDER, h, m, label, "local-backstop"))
            }
        }
        return out
    }

    private fun speakClause(t: Turn, clause: String, acoustic: StringBuilder) {
        if (turn !== t) return
        val dev = HinglishCodec.toAcousticDevanagariScript(clause, t.lang)
        if (dev.isBlank()) return
        if (t.firstClauseNanos == 0L) t.firstClauseNanos = System.nanoTime()
        acoustic.append(dev).append(' ')
        pipeline?.speak(dev)
    }

    private fun onFirstAudio(nanos: Long) {
        val t = turn ?: return
        synchronized(t) {
            t.telemetry = t.telemetry.copy(
                e2eLatencyMs = if (t.endOfSpeechNanos > 0) (nanos - t.endOfSpeechNanos) / 1_000_000 else null,
                ttsFirstAudioMs = if (t.firstClauseNanos > 0) (nanos - t.firstClauseNanos) / 1_000_000 else null,
            )
        }
        _ui.update { it.copy(telemetry = t.telemetry) }
        updateAssistant(t.assistantId) { it.copy(telemetry = t.telemetry) }
        if (_ui.value.state == AgentState.STREAMING) setState(AgentState.SPEAKING)
    }

    /** Warms every provider the cascade may use this turn (primary first), never sending a key. */
    private fun prewarm() {
        if (!settings.cloudEnabled) return
        val providers = llm.liveTargets(settings.cascade()).map { it.model.provider }.distinct()
        if (providers.isEmpty()) return
        providers.forEach { llm.prewarmConnection(it) }
        viewModelScope.launch {
            delay(1500)
            llm.prewarmFor(providers.first())?.let { pw ->
                _ui.update { it.copy(prewarm = if (pw.ok) "${pw.protocol} warm • ${pw.note}" else "pre-warm failed: ${pw.note}") }
            }
        }
    }

    private fun logTelemetry(t: Turn, transcript: String, reply: String, stats: LlmStats?) {
        val tm = t.telemetry
        val j = JSONObject()
            .put("source", tm.source).put("benchmark", t.benchmark?.text).put("lid", tm.lid).put("p_hi", tm.pHindi)
            .put("stt_ms", tm.sttMs).put("stt_rtf", tm.sttRtf).put("encoder_ms", tm.encoderMs).put("decode_ms", tm.decodeMs)
            .put("tokens_per_s", tm.tokensPerSec).put("llm", tm.llmLabel).put("llm_ttft_ms", tm.llmTtftMs)
            .put("conn_reused", tm.connectionReused).put("tts_first_audio_ms", tm.ttsFirstAudioMs)
            .put("e2e_ms", tm.e2eLatencyMs).put("transcript", transcript).put("reply", reply)
            .put("llm_model_id", stats?.modelId).put("failovers", stats?.failovers?.joinToString("; "))
        Log.i("VaaniTelemetry", j.toString())
    }

    // ======================= state helpers ===================================

    private fun setState(s: AgentState) = _ui.update { it.copy(state = s) }

    private fun addFeed(item: FeedItem) = _ui.update { it.copy(feed = (it.feed + item).takeLast(80)) }

    private fun notice(text: String) = addFeed(FeedItem.Notice(ids.getAndIncrement(), text))

    private fun updateAssistant(id: Long, f: (FeedItem.Assistant) -> FeedItem.Assistant) = _ui.update { st ->
        st.copy(feed = st.feed.map { if (it.id == id && it is FeedItem.Assistant) f(it) else it })
    }

    override fun onCleared() {
        llmJob?.cancel()
        pipeline?.release()
        runtime?.close()
        super.onCleared()
    }

    private companion object {
        const val TAG = "VaaniAgent"
    }
}

/** Pure helpers (unit-tested). */
object Metrics {
    private fun words(s: String) = s.lowercase().replace(Regex("[^a-z0-9\\s]"), " ").split(Regex("\\s+")).filter { it.isNotBlank() }

    /** Word error rate of [hyp] against [ref] after lower-casing and stripping punctuation. */
    fun wordErrorRate(ref: String, hyp: String): Double {
        val r = words(ref)
        val h = words(hyp)
        if (r.isEmpty()) return if (h.isEmpty()) 0.0 else 1.0
        var prev = IntArray(h.size + 1) { it }
        for (i in 1..r.size) {
            val cur = IntArray(h.size + 1)
            cur[0] = i
            for (j in 1..h.size) {
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (r[i - 1] == h[j - 1]) 0 else 1)
            }
            prev = cur
        }
        return prev[h.size].toDouble() / r.size
    }
}
