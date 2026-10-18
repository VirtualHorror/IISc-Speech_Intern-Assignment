package `in`.iisc.vaani.llm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
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
import java.util.concurrent.atomic.AtomicReference

/**
 * Zero-RTT streaming cloud LLM client (Groq OpenAI-compatible SSE + Google Gemini SSE).
 *
 *  - **Connection pre-warming**: [prewarmConnection] fires an async, unauthenticated HEAD to
 *    the active provider the moment the VAD detects speech, so DNS + TCP + TLS 1.3 + HTTP/2
 *    are already established in OkHttp's pool by the time STT finishes. The per-call
 *    [LlmStats.reusedConnection] / [LlmStats.connectMs] prove it (0 ms connect on reuse).
 *  - **Streaming**: Server-Sent Events are parsed line by line straight off the socket and
 *    emitted as [LlmEvent.Token]s, so the first clause can be synthesised while the rest of
 *    the reply is still being generated. [LlmStats.ttftMs] is time-to-first-token.
 *  - **Barge-in**: cancelling the collector cancels the OkHttp call, which aborts the
 *    blocking socket read immediately.
 *
 * No Android types: base URLs are injectable so tests drive it against MockWebServer.
 */
class LlmStreamingClient(
    private val groqBaseUrl: String = GROQ_BASE,
    private val geminiBaseUrl: String = GEMINI_BASE,
    client: OkHttpClient? = null,
) {
    data class PrewarmResult(
        val provider: LlmProvider, val ok: Boolean, val ms: Long, val protocol: String, val note: String,
    )

    private val timings = ConcurrentHashMap<Call, CallTiming>()

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

    /** Asynchronously opens (or refreshes) the pooled HTTP/2 connection to [provider]. */
    fun prewarmConnection(provider: LlmProvider) {
        val url = (if (provider == LlmProvider.GROQ) groqBaseUrl else geminiBaseUrl) + "models"
        val request = Request.Builder().url(url).head().header("User-Agent", USER_AGENT).build()
        val t0 = System.nanoTime()
        http.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                timings.remove(call)
                lastPrewarm = PrewarmResult(provider, false, (System.nanoTime() - t0) / 1_000_000, "-", e.message ?: "error")
            }

            override fun onResponse(call: Call, response: Response) {
                val proto = response.protocol.toString()
                response.close()
                val t = timings.remove(call)
                val note = if (t == null || t.reused) "pool already warm" else "connect ${t.connectMs()} ms, TLS ${t.tlsMs()} ms"
                lastPrewarm = PrewarmResult(provider, true, (System.nanoTime() - t0) / 1_000_000, proto, note)
            }
        })
    }

    /**
     * Streams one completion. Collect on any dispatcher; cancellation aborts the request.
     * If the model answers 404/429/503 before any token (retired / rate-limited /
     * overloaded) the request is retried once on [LlmModel.fallbackId].
     */
    fun stream(request: ChatRequest): Flow<LlmEvent> = callbackFlow {
        val t0 = System.nanoTime()
        val m = request.model
        val attempts = buildList {
            add(m.id to m.thinking)
            m.fallbackId?.let { if (it != m.id) add(it to m.fallbackThinking) }
        }
        val current = AtomicReference<Call?>(null)
        val reader = launch(Dispatchers.IO) {
            try {
                for ((i, attempt) in attempts.withIndex()) {
                    val call = http.newCall(buildRequest(request, attempt.first, attempt.second))
                    current.set(call)
                    val retry = runAttempt(call, request, t0, attempt.first, fellBack = i > 0,
                        canRetry = i < attempts.lastIndex) { send(it) }
                    if (!retry) break
                }
            } finally {
                close()
            }
        }
        awaitClose {
            current.get()?.cancel()
            reader.cancel()
        }
    }

    /** One HTTP attempt; returns true if the caller should retry on the fallback model. */
    private suspend fun runAttempt(
        call: Call, request: ChatRequest, t0: Long, modelId: String, fellBack: Boolean, canRetry: Boolean,
        emit: suspend (LlmEvent) -> Unit,
    ): Boolean {
        try {
            call.execute().use { resp ->
                val headersMs = (System.nanoTime() - t0) / 1_000_000
                if (!resp.isSuccessful) {
                    val body = resp.body?.string()
                    if (canRetry && resp.code in RETRYABLE) { timings.remove(call); return true }
                    emit(LlmEvent.Error(errorMessage(body, resp.code), resp.code))
                    return false
                }
                val source = resp.body?.source() ?: run { emit(LlmEvent.Error("empty response body")); return false }
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
                        if (request.model.provider == LlmProvider.GROQ) SseParsers.groq(data) else SseParsers.gemini(data)
                    } catch (e: Exception) {
                        SseParsers.Delta("", false)
                    }
                    if (delta.error != null) { emit(LlmEvent.Error(delta.error)); return false }
                    if (delta.text.isNotEmpty()) {
                        val elapsed = (System.nanoTime() - t0) / 1_000_000
                        if (firstMs < 0) firstMs = elapsed
                        chars += delta.text.length
                        chunks++
                        emit(LlmEvent.Token(delta.text, elapsed))
                    }
                }
                val t = timings.remove(call)
                emit(LlmEvent.Done(LlmStats(
                    ttftMs = if (firstMs >= 0) firstMs else headersMs,
                    totalMs = (System.nanoTime() - t0) / 1_000_000,
                    headersMs = headersMs,
                    connectMs = t?.connectMs() ?: 0,
                    tlsMs = t?.tlsMs() ?: 0,
                    reusedConnection = t?.reused ?: true,
                    protocol = resp.protocol.toString(),
                    chars = chars,
                    chunks = chunks,
                    modelId = modelId,
                    fellBack = fellBack,
                )))
                return false
            }
        } catch (e: IOException) {
            if (!call.isCanceled()) runCatching { emit(LlmEvent.Error(e.message ?: "network error")) }
            return false
        } finally {
            timings.remove(call)
        }
    }

    private fun buildRequest(r: ChatRequest, modelId: String, thinking: GeminiThinking): Request {
        val json = "application/json; charset=utf-8".toMediaType()
        return when (r.model.provider) {
            LlmProvider.GROQ -> {
                val messages = JSONArray().put(JSONObject().put("role", "system").put("content", r.systemPrompt))
                for (t in r.history) {
                    messages.put(JSONObject().put("role", "user").put("content", t.user))
                    messages.put(JSONObject().put("role", "assistant").put("content", t.assistant))
                }
                messages.put(JSONObject().put("role", "user").put("content", r.userMessage))
                val body = JSONObject()
                    .put("model", modelId)
                    .put("messages", messages)
                    .put("stream", true)
                    .put("temperature", r.temperature)
                    .put("max_tokens", r.maxTokens)
                Request.Builder().url(groqBaseUrl + "chat/completions")
                    .header("Authorization", "Bearer ${r.apiKey}")
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
                when (thinking) {
                    GeminiThinking.BUDGET_ZERO -> generation.put("thinkingConfig", JSONObject().put("thinkingBudget", 0))
                    GeminiThinking.LEVEL_MINIMAL -> generation.put("thinkingConfig", JSONObject().put("thinkingLevel", "minimal"))
                    GeminiThinking.NONE -> Unit
                }
                val body = JSONObject()
                    .put("systemInstruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", r.systemPrompt))))
                    .put("contents", contents)
                    .put("generationConfig", generation)
                Request.Builder().url(geminiBaseUrl + "models/$modelId:streamGenerateContent?alt=sse")
                    .header("x-goog-api-key", r.apiKey)
                    .header("Accept", "text/event-stream")
                    .header("User-Agent", USER_AGENT)
                    .post(body.toString().toRequestBody(json))
                    .build()
            }
        }
    }

    private fun errorMessage(body: String?, code: Int): String {
        val msg = runCatching { JSONObject(body ?: "").optJSONObject("error")?.optString("message") }.getOrNull()
        return "HTTP $code" + if (!msg.isNullOrBlank()) ": $msg" else ""
    }

    companion object {
        const val GROQ_BASE = "https://api.groq.com/openai/v1/"
        const val GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/"
        private val RETRYABLE = setOf(404, 429, 503)
        private const val USER_AGENT = "VaaniEdge/2.0 (IISc CI Lab; Android)"
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
