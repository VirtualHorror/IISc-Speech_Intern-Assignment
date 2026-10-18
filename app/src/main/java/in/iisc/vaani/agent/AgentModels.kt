package `in`.iisc.vaani.agent

import `in`.iisc.vaani.llm.LlmModel
import `in`.iisc.vaani.llm.LlmModels
import `in`.iisc.vaani.neural.LanguageMode
import `in`.iisc.vaani.tools.AlarmItem
import `in`.iisc.vaani.tools.DailyForecast
import `in`.iisc.vaani.tools.EventStatus
import `in`.iisc.vaani.tools.ReminderItem
import `in`.iisc.vaani.tools.ScheduleEvent

/** Mic button / status-bar state machine. */
enum class AgentState(val label: String) {
    LOADING("Loading on-device models…"),
    IDLE("Tap to Speak (Auto-VAD)"),
    LISTENING("Listening..."),
    TRANSCRIBING("Transcribing..."),
    STREAMING("Streaming..."),
    SPEAKING("Speaking (Tap to Barge-In)"),
    ERROR("Model load failed"),
}

/** The four official IISc evaluation prompts. */
data class BenchmarkPrompt(val index: Int, val text: String, val lang: String)

val BENCHMARK_PROMPTS = listOf(
    BenchmarkPrompt(0, "What is the weather forecast for Bangalore tomorrow?", "en"),
    BenchmarkPrompt(1, "Can you set a reminder for my meeting at 5 PM?", "en"),
    BenchmarkPrompt(2, "Mujhe kal subah 7 baje ka alarm set karna hai.", "hi"),
    BenchmarkPrompt(3, "Aaj ka schedule kya hai, aur meri next meeting kab hai?", "hi"),
)

/** Per-turn research telemetry (null = not measured this turn). */
data class Telemetry(
    val source: String = "-",
    val sttMs: Long? = null,
    val sttRtf: Double? = null,
    val melMs: Long? = null,
    val encoderMs: Long? = null,
    val decodeMs: Long? = null,
    val tokensPerSec: Double? = null,
    val lid: String? = null,
    val pHindi: Float? = null,
    val llmTtftMs: Long? = null,
    val llmTotalMs: Long? = null,
    val connectionReused: Boolean? = null,
    val connectMs: Long? = null,
    val ttsFirstAudioMs: Long? = null,
    val ttsRtf: Double? = null,
    val e2eLatencyMs: Long? = null,
    val llmLabel: String? = null,
)

sealed class ActionCard {
    data class Weather(val forecast: DailyForecast) : ActionCard()
    data class Alarm(val alarmId: Long) : ActionCard()
    data class Reminder(val reminderId: Long) : ActionCard()
    data class Schedule(val events: List<Pair<ScheduleEvent, EventStatus>>) : ActionCard()
}

sealed class FeedItem {
    abstract val id: Long

    data class User(
        override val id: Long,
        val text: String,
        val lang: String,
        val sttMs: Long?,
        val rawAsr: String?,
        val source: String,
        val benchmark: BenchmarkPrompt? = null,
        val wer: Double? = null,
        /** Set when the text is not a recognition result (e.g. the chip's prompt after STT failed). */
        val note: String? = null,
    ) : FeedItem()

    data class Assistant(
        override val id: Long,
        val text: String,
        val lang: String,
        val acoustic: String = "",
        val ipa: String = "",
        val streaming: Boolean = true,
        val engine: String = "",
        val telemetry: Telemetry? = null,
        val note: String? = null,
    ) : FeedItem()

    data class Card(override val id: Long, val card: ActionCard) : FeedItem()

    data class Notice(override val id: Long, val text: String) : FeedItem()
}

data class AgentUiState(
    val state: AgentState = AgentState.LOADING,
    val statusDetail: String = "Initialising…",
    val runtimeReady: Boolean = false,
    val runtimeSummary: String = "",
    val languageMode: LanguageMode = LanguageMode.AUTO,
    /** The primary rung as requested (override applied) — shown in the top bar. */
    val model: LlmModel = LlmModels.DEFAULT,
    /** The primary as selected in Settings (no override). */
    val baseModel: LlmModel = LlmModels.DEFAULT,
    val modelIdOverride: String = "",
    val cloudEnabled: Boolean = true,
    val cloudReady: Boolean = false,
    val prewarm: String? = null,
    val telemetry: Telemetry = Telemetry(),
    val feed: List<FeedItem> = emptyList(),
    val highlightedPrompt: Int? = null,
    val busyPrompt: Int? = null,
    val alarms: List<AlarmItem> = emptyList(),
    val reminders: List<ReminderItem> = emptyList(),
    val forecast: DailyForecast? = null,
    val micPermission: Boolean = false,
)
