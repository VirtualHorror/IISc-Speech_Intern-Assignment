package `in`.iisc.vaani.llm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.Callback
import okhttp3.Connection
import okhttp3.ConnectionPool
import okhttp3.EventListener
import okhttp3.Handshake
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Proxy
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Zero-RTT streaming cloud LLM client (Groq OpenAI-compatible SSE + Google Gemini SSE) with a
 * cross-provider failover cascade.
 *
 *  - **Connection pre-warming**: [prewarmConnection] fires an async, unauthenticated HEAD to
 *    a provider the moment the VAD detects speech, so DNS + TCP + TLS 1.3 + HTTP/2 are already
 *    established in OkHttp's pool by the time STT finishes. The per-call
 *    [LlmStats.reusedConnection] / [LlmStats.connectMs] prove it (0 ms connect on reuse).
 *  - **Streaming**: Server-Sent Events are parsed line by line straight off the socket and
 *    emitted as [LlmEvent.Token]s, so the first clause can be synthesised while the rest of
 *    the reply is still being generated. [LlmStats.ttftMs] is time-to-first-token.
 *  - **Failover**: [ChatRequest.targets] are tried in order. A rung that fails before its
 *    first token (any HTTP error, a network error, no token within [firstTokenTimeoutMs], or
 *    an empty reply) silently hands the turn to the next rung; a network error also skips the
 *    provider's remaining rungs for this turn. Failed rungs cool down ([cooldownMs]: 429 honours
 *    Retry-After / Gemini `retryDelay`) so later turns go straight to a healthy rung. Once a
 *    token has been streamed the reply is never restarted on another model.
 *  - **Barge-in**: cancelling the collector cancels the OkHttp call, which aborts the
 *    blocking socket read immediately.
 *
 * No Android types: base URLs are injectable so tests drive it against MockWebServer.
 */
class LlmStreamingClient(
    private val groqBaseUrl: String = GROQ_BASE,
    private val geminiBaseUrl: String = GEMINI_BASE,
    client: OkHttpClient? = null,
    private val firstTokenTimeoutMs: Long = FIRST_TOKEN_TIMEOUT_MS,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    data class PrewarmResult(
        val provider: LlmProvider, val ok: Boolean, val ms: Long, val protocol: String, val note: String,
    )

    private val timings = ConcurrentHashMap<Call, CallTiming>()
    private val coolUntil = ConcurrentHashMap<String, Long>()
    private val prewarms = ConcurrentHashMap<LlmProvider, PrewarmResult>()

    val http: OkHttpClient = (client?.newBuilder() ?: OkHttpClient.Builder()
        .protocols(listOf(Protocol.HTTP_2, Protocol.HTTP_1_1))
        .connectionPool(ConnectionPool(4, 5, TimeUnit.MINUTES))
        .connectTimeout(6, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true))
        .eventListenerFactory(object : EventListener.Factory {
            override fun create(call: Call): EventListener = TimingListener(timings.getOrPut(call) { CallTiming() })
        })
        .build()

    @Volatile var lastPrewarm: PrewarmResult? = null
        private set

    fun prewarmFor(provider: LlmProvider): PrewarmResult? = prewarms[provider]

    /** Asynchronously opens (or refreshes) the pooled HTTP/2 connection to [provider]. */
    fun prewarmConnection(provider: LlmProvider) {
        val url = (if (provider == LlmProvider.GROQ) groqBaseUrl else geminiBaseUrl) + "models"
        val request = Request.Builder().url(url).head().header("User-Agent", USER_AGENT).build()
        val t0 = System.nanoTime()
        http.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                timings.remove(call)
                record(PrewarmResult(provider, false, (System.nanoTime() - t0) / 1_000_000, "-", e.message ?: "error"))
            }

            override fun onResponse(call: Call, response: Response) {
                val proto = response.protocol.toString()
                response.close()
                val t = timings.remove(call)
                val note = if (t == null || t.reused) "pool already warm" else "connect ${t.connectMs()} ms, TLS ${t.tlsMs()} ms"
                record(PrewarmResult(provider, true, (System.nanoTime() - t0) / 1_000_000, proto, note))
            }
        })
    }

    private fun record(r: PrewarmResult) {
        prewarms[r.provider] = r
        lastPrewarm = r
    }

    // ---- per-model health ----------------------------------------------------

    fun isCoolingDown(model: LlmModel): Boolean = (coolUntil[healthKey(model)] ?: 0L) > clock()

    /** The rungs a turn would actually try right now: keyed and not cooling down. */
    fun liveTargets(targets: List<LlmTarget>): List<LlmTarget> =
        targets.filter { it.apiKey.isNotBlank() && !isCoolingDown(it.model) }

    /** Forget every cool-down (e.g. after new keys or a new model are saved in Settings). */
    fun resetHealth() = coolUntil.clear()

    private fun penalize(model: LlmModel, ms: Long) {
        if (ms > 0) coolUntil[healthKey(model)] = clock() + ms
    }

    private fun healthKey(m: LlmModel) = "${m.provider}:${m.id}"

    // ---- streaming -----------------------------------------------------------

    /** Why a rung produced no token; the cascade moves on to the next rung. */
    private class Failure(val reason: String, val httpCode: Int, val cooldownMs: Long, val providerDown: Boolean)

    /**
     * Streams one completion through the failover cascade. Collect on any dispatcher;
     * cancellation aborts the request. Emits Tokens then Done, or a single Error when every
     * rung failed before its first token (the caller then answers on-device).
     */
    fun stream(request: ChatRequest): Flow<LlmEvent> = callbackFlow {
        val t0 = System.nanoTime()
        val current = AtomicReference<Call?>(null)
        val reader = launch(Dispatchers.IO) {
            try {
                val failovers = ArrayList<String>()
                val downProviders = HashSet<LlmProvider>()
                var lastCode = 0
                for (target in request.targets) {
                    if (!isActive) return@launch
                    val m = target.model
                    when {
                        target.apiKey.isBlank() -> continue
                        m.provider in downProviders -> { failovers += "${m.id}: skipped (${m.provider.label} unreachable)"; continue }
                        isCoolingDown(m) -> { failovers += "${m.id}: skipped (cooling down)"; continue }
                    }
                    val call = http.newCall(buildRequest(request, target))
                    current.set(call)
                    val failure = runAttempt(call, request, target, t0, failovers) { send(it) } ?: return@launch
                    failovers += "${m.id}: ${failure.reason}"
                    penalize(m, failure.cooldownMs)
                    if (failure.providerDown) downProviders += m.provider
                    if (failure.httpCode != 0) lastCode = failure.httpCode
                }
                send(LlmEvent.Error(
                    if (failovers.isEmpty()) "no cloud model has an API key"
                    else "all cloud models failed: " + failovers.joinToString("; "),
                    lastCode,
                ))
            } finally {
                close()
            }
        }
        awaitClose {
            current.get()?.cancel()
            reader.cancel()
        }
    }

    /**
     * One HTTP attempt. Returns null when the turn is finished (answered, failed after the
     * first token, or cancelled by barge-in) or a [Failure] when the next rung should try.
     */
    private suspend fun runAttempt(
        call: Call, request: ChatRequest, target: LlmTarget, t0: Long, failovers: List<String>,
        emit: suspend (LlmEvent) -> Unit,
    ): Failure? = coroutineScope {
        val gotToken = AtomicBoolean(false)
        val timedOut = AtomicBoolean(false)
        val watchdog = launch {
            delay(firstTokenTimeoutMs)
            if (!gotToken.get()) { timedOut.set(true); call.cancel() }
        }
        try {
            call.execute().use { resp ->
                val headersMs = (System.nanoTime() - t0) / 1_000_000
                if (!resp.isSuccessful) {
                    val body = runCatching { resp.peekBody(MAX_ERROR_BODY).string() }.getOrNull()
                    return@coroutineScope Failure(errorMessage(body, resp.code), resp.code,
                        cooldownMs(resp.code, resp.header("Retry-After"), body), providerDown = false)
                }
                val source = resp.body?.source()
                    ?: return@coroutineScope Failure("empty response body", 0, 0, providerDown = false)
                var firstMs = -1L
                var chars = 0
                var chunks = 0
                while (true) {
                    val line = source.readUtf8Line() ?: break
                    if (!line.startsWith("data:")) continue
                    val data = line.substring(5).trim()
                    if (data.isEmpty()) continue
                    if (data == "[DONE]") break
                    val delta = try {
                        if (target.model.provider == LlmProvider.GROQ) SseParsers.groq(data) else SseParsers.gemini(data)
                    } catch (e: Exception) {
                        SseParsers.Delta("", false)
                    }
                    if (delta.error != null) {
                        if (firstMs < 0) return@coroutineScope Failure("stream error: ${delta.error}", 0, SERVER_COOLDOWN_MS, false)
                        emit(LlmEvent.Error(delta.error))
                        return@coroutineScope null
                    }
                    if (delta.text.isNotEmpty()) {
                        val elapsed = (System.nanoTime() - t0) / 1_000_000
                        if (firstMs < 0) { firstMs = elapsed; gotToken.set(true); watchdog.cancel() }
                        chars += delta.text.length
                        chunks++
                        emit(LlmEvent.Token(delta.text, elapsed))
                    }
                }
                if (firstMs < 0) return@coroutineScope Failure("empty reply", 0, 0, providerDown = false)
                val t = timings.remove(call)
                emit(LlmEvent.Done(LlmStats(
                    ttftMs = firstMs,
                    totalMs = (System.nanoTime() - t0) / 1_000_000,
                    headersMs = headersMs,
                    connectMs = t?.connectMs() ?: 0,
                    tlsMs = t?.tlsMs() ?: 0,
                    reusedConnection = t?.reused ?: true,
                    protocol = resp.protocol.toString(),
                    chars = chars,
                    chunks = chunks,
                    modelId = target.model.id,
                    modelLabel = target.model.label,
                    failovers = failovers.toList(),
                )))
                null
            }
        } catch (e: IOException) {
            when {
                timedOut.get() -> Failure("no first token in $firstTokenTimeoutMs ms", 0, SERVER_COOLDOWN_MS, false)
                call.isCanceled() -> null                                       // barge-in
                gotToken.get() -> { runCatching { emit(LlmEvent.Error(e.message ?: "network error")) }; null }
                else -> Failure("network error: ${e.message ?: e.javaClass.simpleName}", 0, SERVER_COOLDOWN_MS, true)
            }
        } finally {
            watchdog.cancel()
            timings.remove(call)
        }
    }

    private fun buildRequest(r: ChatRequest, target: LlmTarget): Request {
        val json = "application/json; charset=utf-8".toMediaType()
        val model = target.model
        return when (model.provider) {
            LlmProvider.GROQ -> {
                val messages = JSONArray().put(JSONObject().put("role", "system").put("content", r.systemPrompt))
                for (t in r.history) {
                    messages.put(JSONObject().put("role", "user").put("content", t.user))
                    messages.put(JSONObject().put("role", "assistant").put("content", t.assistant))
                }
                messages.put(JSONObject().put("role", "user").put("content", r.userMessage))
                val body = JSONObject()
                    .put("model", model.id)
                    .put("messages", messages)
                    .put("stream", true)
                    .put("temperature", r.temperature)
                    .put("max_completion_tokens", r.maxTokens + model.reasoning.extraTokens)
                when (model.reasoning) {
                    GroqReasoning.OFF -> body.put("reasoning_effort", "none")
                    GroqReasoning.LOW_HIDDEN -> body.put("reasoning_effort", "low").put("include_reasoning", false)
                    GroqReasoning.NONE -> Unit
                }
                Request.Builder().url(groqBaseUrl + "chat/completions")
                    .header("Authorization", "Bearer ${target.apiKey}")
                    .header("Accept", "text/event-stream")
                    .header("User-Agent", USER_AGENT)
                    .post(body.toString().toRequestBody(json))
                    .build()
            }
            LlmProvider.GEMINI -> {
                val contents = JSONArray()
                fun turn(role: String, text: String) =
                    JSONObject().put("role", role).put("parts", JSONArray().put(JSONObject().put("text", text)))
                for (t in r.history) { contents.put(turn("user", t.user)); contents.put(turn("model", t.assistant)) }
                contents.put(turn("user", r.userMessage))
                val generation = JSONObject()
                    .put("temperature", r.temperature)
                    .put("maxOutputTokens", r.maxTokens)
                // Flash models "think" by default: that burns TTFT and the 160-token budget.
                when (model.thinking) {
                    GeminiThinking.BUDGET_ZERO -> generation.put("thinkingConfig", JSONObject().put("thinkingBudget", 0))
                    GeminiThinking.LEVEL_MINIMAL -> generation.put("thinkingConfig", JSONObject().put("thinkingLevel", "minimal"))
                    GeminiThinking.NONE -> Unit
                }
                val body = JSONObject()
                    .put("systemInstruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", r.systemPrompt))))
                    .put("contents", contents)
                    .put("generationConfig", generation)
                Request.Builder().url(geminiBaseUrl + "models/${model.id}:streamGenerateContent?alt=sse")
                    .header("x-goog-api-key", target.apiKey)
                    .header("Accept", "text/event-stream")
                    .header("User-Agent", USER_AGENT)
                    .post(body.toString().toRequestBody(json))
                    .build()
            }
        }
    }

    private fun errorMessage(body: String?, code: Int): String {
        val msg = runCatching { JSONObject(body ?: "").optJSONObject("error")?.optString("message") }.getOrNull()
        return "HTTP $code" + if (!msg.isNullOrBlank()) ": ${msg.lineSequence().first().take(160)}" else ""
    }

    companion object {
        const val GROQ_BASE = "https://api.groq.com/openai/v1/"
        const val GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/"
        const val FIRST_TOKEN_TIMEOUT_MS = 4_000L
        const val MAX_COOLDOWN_MS = 30 * 60_000L
        private const val RATE_LIMIT_COOLDOWN_MS = 60_000L
        private const val SERVER_COOLDOWN_MS = 20_000L
        private const val MAX_ERROR_BODY = 16_384L
        private const val USER_AGENT = "VaaniEdge/3.0 (IISc CI Lab; Android)"
        private val RETRY_DELAY = Regex(""""retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"""")

        /**
         * How long a rung sits out after failing with [code]: rate limits honour `Retry-After`
         * (Groq) or the `RetryInfo.retryDelay` in the body (Gemini); retired models, bad keys and
         * rejected requests wait the maximum (Settings changes reset it); 5xx, timeouts and
         * network errors briefly.
         */
        fun cooldownMs(code: Int, retryAfterHeader: String?, body: String?): Long = when (code) {
            429 -> {
                val header = retryAfterHeader?.trim()?.toDoubleOrNull()
                val gemini = body?.let { RETRY_DELAY.find(it)?.groupValues?.get(1)?.toDoubleOrNull() }
                ((header ?: gemini)?.let { (it * 1000).toLong() } ?: RATE_LIMIT_COOLDOWN_MS).coerceIn(0, MAX_COOLDOWN_MS)
            }
            400, 401, 403, 404 -> MAX_COOLDOWN_MS
            else -> SERVER_COOLDOWN_MS
        }
    }
}

/** Per-call socket timing captured by OkHttp's EventListener. */
private class CallTiming {
    @Volatile var connectStart = 0L
    @Volatile var connectEnd = 0L
    @Volatile var secureStart = 0L
    @Volatile var secureEnd = 0L
    @Volatile var reused = true

    fun connectMs(): Long = if (connectEnd > connectStart && connectStart > 0) (connectEnd - connectStart) / 1_000_000 else 0
    fun tlsMs(): Long = if (secureEnd > secureStart && secureStart > 0) (secureEnd - secureStart) / 1_000_000 else 0
}

private class TimingListener(private val t: CallTiming) : EventListener() {
    override fun connectStart(call: Call, inetSocketAddress: InetSocketAddress, proxy: Proxy) {
        t.connectStart = System.nanoTime(); t.reused = false
    }

    override fun secureConnectStart(call: Call) { t.secureStart = System.nanoTime() }
    override fun secureConnectEnd(call: Call, handshake: Handshake?) { t.secureEnd = System.nanoTime() }
    override fun connectEnd(call: Call, inetSocketAddress: InetSocketAddress, proxy: Proxy, protocol: Protocol?) {
        t.connectEnd = System.nanoTime()
    }

    override fun connectionAcquired(call: Call, connection: Connection) {}
}

/** SSE `data:` payload parsers for the two providers. */
internal object SseParsers {
    data class Delta(val text: String, val finished: Boolean, val error: String? = null)

    /** OpenAI-compatible chunk: `choices[0].delta.content`, `choices[0].finish_reason`. */
    fun groq(data: String): Delta {
        val o = JSONObject(data)
        o.optJSONObject("error")?.let { return Delta("", true, it.optString("message", "error")) }
        val choice = o.optJSONArray("choices")?.optJSONObject(0) ?: return Delta("", false)
        val delta = choice.optJSONObject("delta")
        val text = if (delta == null || delta.isNull("content")) "" else delta.optString("content", "")
        return Delta(text, !choice.isNull("finish_reason"))
    }

    /** Gemini chunk: `candidates[0].content.parts[*].text`, `candidates[0].finishReason`. */
    fun gemini(data: String): Delta {
        val o = JSONObject(data)
        o.optJSONObject("error")?.let { return Delta("", true, it.optString("message", "error")) }
        val cand = o.optJSONArray("candidates")?.optJSONObject(0) ?: return Delta("", false)
        val parts = cand.optJSONObject("content")?.optJSONArray("parts")
        val sb = StringBuilder()
        if (parts != null) {
            for (i in 0 until parts.length()) {
                val p = parts.optJSONObject(i) ?: continue
                if (p.optBoolean("thought", false)) continue
                if (!p.isNull("text")) sb.append(p.optString("text", ""))
            }
        }
        return Delta(sb.toString(), cand.has("finishReason"))
    }
}
