package `in`.iisc.vaani.llm

/** Cloud LLM back-ends with an SSE streaming API. */
enum class LlmProvider(val label: String) { GROQ("Groq"), GEMINI("Google Gemini") }

/** How to keep a Gemini "thinking" model from spending TTFT and the token budget on thoughts. */
enum class GeminiThinking { NONE, BUDGET_ZERO, LEVEL_MINIMAL }

/**
 * Groq reasoning control. Qwen3 thinks by default (`reasoning_effort: none` turns it off);
 * GPT-OSS always reasons, so it runs at `low` effort with the reasoning stream hidden and a
 * larger completion budget, because reasoning tokens count against `max_completion_tokens`.
 */
enum class GroqReasoning(val extraTokens: Int) { NONE(0), OFF(0), LOW_HIDDEN(352) }

/** A selectable model; [label] is what the telemetry HUD shows, [note] the Settings hint. */
data class LlmModel(
    val id: String,
    val provider: LlmProvider,
    val label: String,
    val thinking: GeminiThinking = GeminiThinking.NONE,
    val reasoning: GroqReasoning = GroqReasoning.NONE,
    val note: String = "",
)

object LlmModels {
    // Groq retired Llama 3.3 70B and Llama 3.1 8B (HTTP 404 model_not_found, 18 Oct 2026).
    // One request each from the build host with the app's prompt (speech_lab/llm_smoke_once.py):
    // Qwen3.8-27B (reasoning off) TTFT 111 ms, GPT-OSS-20B 472 ms, GPT-OSS-120B 667 ms; all three
    // emitted the alarm tag. Free tier: 1,000 requests/day and 8,000 tokens/min per model.
    val GROQ_QWEN = LlmModel("qwen/qwen3.8-27b", LlmProvider.GROQ, "Groq Qwen3.8-27B",
        reasoning = GroqReasoning.OFF, note = "default • fastest (111 ms TTFT)")
    val GROQ_GPT_OSS_20B = LlmModel("openai/gpt-oss-20b", LlmProvider.GROQ, "Groq GPT-OSS-20B",
        reasoning = GroqReasoning.LOW_HIDDEN, note = "fast / economical (472 ms TTFT)")
    val GROQ_GPT_OSS_120B = LlmModel("openai/gpt-oss-120b", LlmProvider.GROQ, "Groq GPT-OSS-120B",
        reasoning = GroqReasoning.LOW_HIDDEN, note = "largest Groq model (667 ms TTFT)")

    // gemini-2.5-flash answers HTTP 404 "no longer available to new users" (Oct 2026), so the
    // Gemini rungs use the 3.5 generation. Flash-Lite (thinking minimal) TTFT ~0.8-1.1 s; the
    // free tier allows Flash only 20 requests/day (HTTP 429 PerDay quota when exhausted).
    val GEMINI_FLASH_LITE = LlmModel("gemini-3.5-flash-lite", LlmProvider.GEMINI, "Gemini 3.5 Flash-Lite",
        GeminiThinking.LEVEL_MINIMAL, note = "cross-provider failover (~1 s TTFT)")
    val GEMINI_FLASH = LlmModel("gemini-3.5-flash", LlmProvider.GEMINI, "Gemini 3.5 Flash",
        GeminiThinking.BUDGET_ZERO, note = "last cloud rung (20 requests/day free)")

    /** Default primary: lowest measured TTFT with correct tags and Roman Hinglish replies. */
    val DEFAULT = GROQ_QWEN

    /**
     * Automatic failover order. A rung that fails before its first token (HTTP error, network
     * error, first-token timeout, empty reply) hands the turn to the next one; after the last,
     * the on-device LocalResponder answers.
     */
    val CASCADE = listOf(GROQ_QWEN, GROQ_GPT_OSS_20B, GEMINI_FLASH_LITE, GEMINI_FLASH)

    /** Settings choices (the primary rung). */
    val ALL = listOf(GROQ_QWEN, GROQ_GPT_OSS_20B, GROQ_GPT_OSS_120B, GEMINI_FLASH_LITE, GEMINI_FLASH)

    fun byId(id: String?): LlmModel? = ALL.firstOrNull { it.id == id }

    /** [primary] first, then [CASCADE] in its fixed order, each model id once. */
    fun cascadeFrom(primary: LlmModel): List<LlmModel> = listOf(primary) + CASCADE.filter { it.id != primary.id }
}

/** One prior exchange, replayed for follow-up questions. */
data class ChatTurn(val user: String, val assistant: String)

/** One rung of the failover cascade: a model plus the API key of its provider. */
data class LlmTarget(val model: LlmModel, val apiKey: String)

/** Everything one streamed completion needs (no Android types: unit-testable). */
data class ChatRequest(
    /** Tried in order until one streams a token; rungs without a key are skipped. */
    val targets: List<LlmTarget>,
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
    val modelLabel: String = "",
    /** Rungs that failed before this one answered, e.g. `qwen/qwen3.8-27b: HTTP 429`. */
    val failovers: List<String> = emptyList(),
) {
    val fellBack: Boolean get() = failovers.isNotEmpty()
}

sealed class LlmEvent {
    /** A content delta, [elapsedMs] after the request was issued. */
    data class Token(val text: String, val elapsedMs: Long) : LlmEvent()
    data class Done(val stats: LlmStats) : LlmEvent()
    data class Error(val message: String, val httpCode: Int = 0) : LlmEvent()
}
