package `in`.iisc.vaani.llm

/** Cloud LLM back-ends with an SSE streaming API. */
enum class LlmProvider(val label: String) { GROQ("Groq"), GEMINI("Google Gemini") }

/** How to keep a Gemini "thinking" model from spending TTFT and the token budget on thoughts. */
enum class GeminiThinking { NONE, BUDGET_ZERO, LEVEL_MINIMAL }

/**
 * A selectable model; [label] is what the telemetry HUD shows. [fallbackId] is tried once,
 * automatically, if the primary answers 404/429/503 before any token (retired or overloaded).
 */
data class LlmModel(
    val id: String,
    val provider: LlmProvider,
    val label: String,
    val thinking: GeminiThinking = GeminiThinking.NONE,
    val fallbackId: String? = null,
    val fallbackThinking: GeminiThinking = GeminiThinking.NONE,
)

object LlmModels {
    val GROQ_LLAMA_70B = LlmModel("llama-3.3-70b-versatile", LlmProvider.GROQ, "Groq Llama-3.3-70B",
        fallbackId = "llama-3.1-8b-instant")
    val GROQ_LLAMA_8B = LlmModel("llama-3.1-8b-instant", LlmProvider.GROQ, "Groq Llama-3.1-8B")

    // gemini-2.5-flash now answers HTTP 404 "no longer available to new users" (Oct 2026), so
    // the Gemini path uses the 3.5 generation. Measured from the build host with the app's
    // real prompt: Flash-Lite (thinking minimal) TTFT ~0.9 s, Flash (budget 0) ~1.45 s; both
    // emitted every alarm/reminder tag. speech_lab/evaluation_metrics.json -> llm_benchmark.
    val GEMINI_FLASH_LITE = LlmModel("gemini-3.5-flash-lite", LlmProvider.GEMINI, "Gemini 3.5 Flash-Lite",
        GeminiThinking.LEVEL_MINIMAL, fallbackId = "gemini-flash-lite-latest", fallbackThinking = GeminiThinking.LEVEL_MINIMAL)
    val GEMINI_FLASH = LlmModel("gemini-3.5-flash", LlmProvider.GEMINI, "Gemini 3.5 Flash",
        GeminiThinking.BUDGET_ZERO, fallbackId = "gemini-3.5-flash-lite", fallbackThinking = GeminiThinking.LEVEL_MINIMAL)

    val ALL = listOf(GROQ_LLAMA_70B, GROQ_LLAMA_8B, GEMINI_FLASH_LITE, GEMINI_FLASH)

    fun byId(id: String?): LlmModel? = ALL.firstOrNull { it.id == id }
}

/** One prior exchange, replayed for follow-up questions. */
data class ChatTurn(val user: String, val assistant: String)

/** Everything one streamed completion needs (no Android types: unit-testable). */
data class ChatRequest(
    val model: LlmModel,
    val apiKey: String,
    val systemPrompt: String,
    val userMessage: String,
    val history: List<ChatTurn> = emptyList(),
    val temperature: Double = 0.2,
    val maxTokens: Int = 160,
)

/** Timing evidence for one call; [connectMs]/[tlsMs] are 0 when a pre-warmed connection is reused. */
data class LlmStats(
    val ttftMs: Long,
    val totalMs: Long,
    val headersMs: Long,
    val connectMs: Long,
    val tlsMs: Long,
    val reusedConnection: Boolean,
    val protocol: String,
    val chars: Int,
    val chunks: Int,
    val modelId: String = "",
    val fellBack: Boolean = false,
)

sealed class LlmEvent {
    /** A content delta, [elapsedMs] after the request was issued. */
    data class Token(val text: String, val elapsedMs: Long) : LlmEvent()
    data class Done(val stats: LlmStats) : LlmEvent()
    data class Error(val message: String, val httpCode: Int = 0) : LlmEvent()
}
