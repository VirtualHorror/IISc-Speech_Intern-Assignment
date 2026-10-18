package `in`.iisc.vaani.llm

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.concurrent.TimeUnit

/** Drives the real streaming client against a local SSE server (no network, no keys). */
class LlmStreamingClientTest {

    private lateinit var server: MockWebServer
    private lateinit var client: LlmStreamingClient

    @Before fun setUp() {
        server = MockWebServer().apply { start() }
        val base = server.url("/").toString()
        client = LlmStreamingClient(groqBaseUrl = base, geminiBaseUrl = base, firstTokenTimeoutMs = 1_500)
    }

    @After fun tearDown() = server.shutdown()

    private fun sse(vararg data: String) = data.joinToString("") { "data: $it\n\n" }

    private fun groqChunk(text: String) = """{"choices":[{"delta":{"content":${JSONObject.quote(text)}},"finish_reason":null}]}"""

    private fun geminiChunk(text: String) =
        """{"candidates":[{"content":{"parts":[{"text":${JSONObject.quote(text)}}],"role":"model"},"finishReason":"STOP"}]}"""

    private fun ok(body: String) = MockResponse().setHeader("Content-Type", "text/event-stream").setBody(body)

    private fun request(vararg models: LlmModel) = ChatRequest(
        targets = models.map { LlmTarget(it, "test-key") },
        systemPrompt = "sys", userMessage = "hello",
        history = listOf(ChatTurn("q1", "a1")),
    )

    private fun text(events: List<LlmEvent>) = events.filterIsInstance<LlmEvent.Token>().joinToString("") { it.text }

    // ===== wire formats ====================================================

    @Test fun groqOpenAiCompatibleSse() = runBlocking {
        server.enqueue(
            ok(sse(
                """{"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}""",
                groqChunk("Tomorrow in Bangalore"),
                groqChunk(" expect rain."),
                """{"choices":[{"delta":{},"finish_reason":"stop"}]}""",
                "[DONE]",
            )).throttleBody(64, 5, TimeUnit.MILLISECONDS),
        )
        val events = client.stream(request(LlmModels.GROQ_QWEN)).toList()
        assertEquals("Tomorrow in Bangalore expect rain.", text(events))
        val done = events.last() as LlmEvent.Done
        assertTrue(done.stats.ttftMs >= 0)
        assertEquals(2, done.stats.chunks)
        assertEquals("qwen/qwen3.8-27b", done.stats.modelId)
        assertFalse(done.stats.fellBack)

        val rec = server.takeRequest()
        assertEquals("/chat/completions", rec.path)
        assertEquals("Bearer test-key", rec.getHeader("Authorization"))
        val body = JSONObject(rec.body.readUtf8())
        assertEquals("qwen/qwen3.8-27b", body.getString("model"))
        assertEquals(true, body.getBoolean("stream"))
        assertEquals(160, body.getInt("max_completion_tokens"))
        assertEquals("none", body.getString("reasoning_effort"))       // Qwen3 thinking off
        assertEquals(0.2, body.getDouble("temperature"), 1e-9)
        assertEquals(4, body.getJSONArray("messages").length())   // system + history pair + user
    }

    @Test fun gptOssRunsLowEffortHiddenReasoningWithALargerBudget() = runBlocking {
        server.enqueue(ok(sse(groqChunk("ok"), "[DONE]")))
        client.stream(request(LlmModels.GROQ_GPT_OSS_20B)).toList()
        val body = JSONObject(server.takeRequest().body.readUtf8())
        assertEquals("openai/gpt-oss-20b", body.getString("model"))
        assertEquals("low", body.getString("reasoning_effort"))
        assertEquals(false, body.getBoolean("include_reasoning"))
        assertEquals(160 + 352, body.getInt("max_completion_tokens"))
    }

    @Test fun geminiSseWithThinkingDisabled() = runBlocking {
        server.enqueue(
            ok(sse(
                """{"candidates":[{"content":{"parts":[{"text":"Kal subah 7:00 AM ka "}],"role":"model"}}]}""",
                """{"candidates":[{"content":{"parts":[{"text":"alarm set ho gaya hai."}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":42}}""",
            )),
        )
        val events = client.stream(request(LlmModels.GEMINI_FLASH)).toList()
        assertEquals("Kal subah 7:00 AM ka alarm set ho gaya hai.", text(events))
        assertTrue(events.last() is LlmEvent.Done)

        val rec = server.takeRequest()
        assertEquals("/models/gemini-3.5-flash:streamGenerateContent?alt=sse", rec.path)
        assertEquals("test-key", rec.getHeader("x-goog-api-key"))
        val body = JSONObject(rec.body.readUtf8())
        val gen = body.getJSONObject("generationConfig")
        assertEquals(0, gen.getJSONObject("thinkingConfig").getInt("thinkingBudget"))
        assertEquals(160, gen.getInt("maxOutputTokens"))
        assertEquals("model", body.getJSONArray("contents").getJSONObject(1).getString("role"))
    }

    @Test fun flashLiteUsesMinimalThinkingLevel() = runBlocking {
        server.enqueue(ok(sse(geminiChunk("ok"))))
        client.stream(request(LlmModels.GEMINI_FLASH_LITE)).toList()
        val rec = server.takeRequest()
        assertEquals("/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse", rec.path)
        val think = JSONObject(rec.body.readUtf8()).getJSONObject("generationConfig").getJSONObject("thinkingConfig")
        assertEquals("minimal", think.getString("thinkingLevel"))
    }

    // ===== failover cascade ================================================

    @Test fun cascadeOrderPutsTheChosenModelFirst() {
        assertEquals(LlmModels.CASCADE, LlmModels.cascadeFrom(LlmModels.DEFAULT))
        assertEquals(
            listOf("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "gemini-3.5-flash-lite", "gemini-3.5-flash"),
            LlmModels.CASCADE.map { it.id },
        )
        assertEquals(
            listOf("gemini-3.5-flash-lite", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "gemini-3.5-flash"),
            LlmModels.cascadeFrom(LlmModels.GEMINI_FLASH_LITE).map { it.id },
        )
        assertEquals(5, LlmModels.cascadeFrom(LlmModels.GROQ_GPT_OSS_120B).size)
    }

    @Test fun failsOverAcrossProvidersBeforeAnyToken() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(429).setHeader("Retry-After", "7")
            .setBody("""{"error":{"message":"Rate limit reached for model qwen","code":"rate_limit_exceeded"}}"""))
        server.enqueue(MockResponse().setResponseCode(503).setBody("""{"error":{"message":"over capacity"}}"""))
        server.enqueue(ok(sse(geminiChunk("Hello from Gemini."))))
        val events = client.stream(request(*LlmModels.CASCADE.toTypedArray())).toList()

        assertEquals("Hello from Gemini.", text(events))
        val done = events.last() as LlmEvent.Done
        assertTrue(done.stats.fellBack)
        assertEquals("gemini-3.5-flash-lite", done.stats.modelId)
        assertEquals("Gemini 3.5 Flash-Lite", done.stats.modelLabel)
        assertEquals(2, done.stats.failovers.size)
        assertTrue(done.stats.failovers[0], done.stats.failovers[0].startsWith("qwen/qwen3.8-27b: HTTP 429"))
        assertTrue(done.stats.failovers[1], done.stats.failovers[1].startsWith("openai/gpt-oss-20b: HTTP 503"))
        assertEquals("/chat/completions", server.takeRequest().path)
        assertEquals("/chat/completions", server.takeRequest().path)
        assertEquals("/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse", server.takeRequest().path)
        assertEquals(3, server.requestCount)
    }

    @Test fun everyPreTokenHttpErrorFailsOver() {
        // (408 is left out: OkHttp itself replays a 408 once on the same rung.)
        for (code in listOf(400, 401, 403, 404, 429, 500, 502, 503, 504)) {
            server.enqueue(MockResponse().setResponseCode(code).setBody("""{"error":{"message":"boom $code"}}"""))
            server.enqueue(ok(sse(groqChunk("second rung"), "[DONE]")))
            val c = LlmStreamingClient(server.url("/").toString(), server.url("/").toString())
            val events = runBlocking { c.stream(request(LlmModels.GROQ_QWEN, LlmModels.GROQ_GPT_OSS_20B)).toList() }
            assertEquals("HTTP $code", "second rung", text(events))
            assertEquals("HTTP $code", "openai/gpt-oss-20b", (events.last() as LlmEvent.Done).stats.modelId)
        }
    }

    @Test fun firstTokenTimeoutFailsOver() = runBlocking {
        val fast = LlmStreamingClient(server.url("/").toString(), server.url("/").toString(), firstTokenTimeoutMs = 300)
        server.enqueue(ok(sse(groqChunk("too late"), "[DONE]")).setBodyDelay(3, TimeUnit.SECONDS))
        server.enqueue(ok(sse(geminiChunk("on time"))))
        val t0 = System.nanoTime()
        val events = fast.stream(request(LlmModels.GROQ_QWEN, LlmModels.GEMINI_FLASH_LITE)).toList()
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertEquals("on time", text(events))
        val done = events.last() as LlmEvent.Done
        assertTrue(done.stats.failovers.single(), done.stats.failovers.single().contains("no first token"))
        assertTrue("took $ms ms", ms < 2_500)
    }

    @Test fun emptyReplyFailsOver() = runBlocking {
        server.enqueue(ok(sse("""{"choices":[{"delta":{},"finish_reason":"stop"}]}""", "[DONE]")))
        server.enqueue(ok(sse(geminiChunk("Answer from rung two."))))
        val events = client.stream(request(LlmModels.GROQ_QWEN, LlmModels.GEMINI_FLASH_LITE)).toList()
        assertEquals("Answer from rung two.", text(events))
        assertTrue((events.last() as LlmEvent.Done).stats.failovers.single().contains("empty reply"))
    }

    @Test fun aRateLimitedRungIsSkippedOnTheNextTurn() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(429).setHeader("Retry-After", "30").setBody("{}"))
        server.enqueue(ok(sse(groqChunk("turn one"), "[DONE]")))
        server.enqueue(ok(sse(groqChunk("turn two"), "[DONE]")))
        val req = request(LlmModels.GROQ_QWEN, LlmModels.GROQ_GPT_OSS_20B)

        assertEquals("turn one", text(client.stream(req).toList()))
        val second = client.stream(req).toList()
        assertEquals("turn two", text(second))
        assertEquals(3, server.requestCount)                      // the 429'd rung was not retried
        assertTrue(client.isCoolingDown(LlmModels.GROQ_QWEN))
        assertEquals(LlmModels.GROQ_GPT_OSS_20B, client.liveTargets(req.targets).first().model)

        client.resetHealth()                                      // e.g. new key saved in Settings
        assertFalse(client.isCoolingDown(LlmModels.GROQ_QWEN))
    }

    @Test fun allRungsFailingEmitsOneErrorNamingEachRung() = runBlocking {
        repeat(3) { server.enqueue(MockResponse().setResponseCode(503).setBody("""{"error":{"message":"overloaded"}}""")) }
        val events = client.stream(request(LlmModels.GROQ_QWEN, LlmModels.GROQ_GPT_OSS_20B, LlmModels.GEMINI_FLASH_LITE)).toList()
        val err = events.single() as LlmEvent.Error
        assertEquals(503, err.httpCode)
        for (id in listOf("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "gemini-3.5-flash-lite")) assertTrue(err.message, id in err.message)
    }

    @Test fun keylessRungsAreNeverRequested() = runBlocking {
        server.enqueue(ok(sse(geminiChunk("gemini only"))))
        val req = ChatRequest(
            targets = listOf(LlmTarget(LlmModels.GROQ_QWEN, ""), LlmTarget(LlmModels.GEMINI_FLASH_LITE, "g-key")),
            systemPrompt = "sys", userMessage = "hi",
        )
        val done = client.stream(req).toList().last() as LlmEvent.Done
        assertEquals("gemini-3.5-flash-lite", done.stats.modelId)
        assertFalse(done.stats.fellBack)
        assertEquals(1, server.requestCount)
    }

    @Test fun midStreamErrorIsReportedNotRetried() = runBlocking {
        server.enqueue(ok(sse(groqChunk("Partial"), """{"error":{"message":"stream reset"}}""")))
        val events = client.stream(request(LlmModels.GROQ_QWEN, LlmModels.GEMINI_FLASH_LITE)).toList()
        assertEquals("Partial", text(events))
        assertTrue((events.last() as LlmEvent.Error).message.contains("stream reset"))
        assertEquals(1, server.requestCount)                      // the user already heard rung one
    }

    @Test fun httpErrorsSurfaceTheProviderMessage() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(404)
            .setBody("""{"error":{"message":"The model `llama-3.3-70b-versatile` does not exist","code":"model_not_found"}}"""))
        val events = client.stream(request(LlmModels.GROQ_QWEN)).toList()
        val err = events.single() as LlmEvent.Error
        assertEquals(404, err.httpCode)
        assertTrue(err.message, err.message.contains("does not exist"))
    }

    @Test fun cooldownHonoursRetryAfterAndGeminiRetryDelay() {
        assertEquals(7_000L, LlmStreamingClient.cooldownMs(429, retryAfterHeader = "7", body = null))
        assertEquals(60_000L, LlmStreamingClient.cooldownMs(429, retryAfterHeader = null, body = "{}"))
        val gemini = """{"error":{"code":429,"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"41108s"}]}}"""
        assertEquals(LlmStreamingClient.MAX_COOLDOWN_MS, LlmStreamingClient.cooldownMs(429, null, gemini))
        assertEquals(12_500L, LlmStreamingClient.cooldownMs(429, null, """{"retryDelay": "12.5s"}"""))
        assertEquals(LlmStreamingClient.MAX_COOLDOWN_MS, LlmStreamingClient.cooldownMs(404, null, null))
        assertEquals(20_000L, LlmStreamingClient.cooldownMs(503, null, null))
    }

    // ===== connection + parsers ============================================

    @Test fun prewarmOpensThePooledConnection() {
        server.enqueue(MockResponse().setResponseCode(401))
        client.prewarmConnection(LlmProvider.GROQ)
        val rec = server.takeRequest(5, TimeUnit.SECONDS)!!
        assertEquals("HEAD", rec.method)
        assertEquals(null, rec.getHeader("Authorization"))         // never sends the key
        val deadline = System.currentTimeMillis() + 5000
        while (client.lastPrewarm == null && System.currentTimeMillis() < deadline) Thread.sleep(10)
        assertTrue(client.lastPrewarm!!.ok)
    }

    @Test fun sseParsersIgnoreNullContentAndThoughts() {
        assertEquals("", SseParsers.groq("""{"choices":[{"delta":{"content":null},"finish_reason":null}]}""").text)
        val g = SseParsers.gemini("""{"candidates":[{"content":{"parts":[{"text":"x","thought":true},{"text":"y"}]}}]}""")
        assertEquals("y", g.text)
    }
}
