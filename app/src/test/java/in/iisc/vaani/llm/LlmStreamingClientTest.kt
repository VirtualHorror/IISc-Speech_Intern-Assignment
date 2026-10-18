package `in`.iisc.vaani.llm

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
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
        client = LlmStreamingClient(groqBaseUrl = base, geminiBaseUrl = base)
    }

    @After fun tearDown() = server.shutdown()

    private fun sse(vararg data: String) = data.joinToString("") { "data: $it\n\n" }

    private fun request(model: LlmModel) = ChatRequest(
        model = model, apiKey = "test-key", systemPrompt = "sys", userMessage = "hello",
        history = listOf(ChatTurn("q1", "a1")),
    )

    @Test fun groqOpenAiCompatibleSse() = runBlocking {
        server.enqueue(
            MockResponse().setHeader("Content-Type", "text/event-stream").setBody(
                sse(
                    """{"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}""",
                    """{"choices":[{"delta":{"content":"Tomorrow in Bangalore"},"finish_reason":null}]}""",
                    """{"choices":[{"delta":{"content":" expect rain."},"finish_reason":null}]}""",
                    """{"choices":[{"delta":{},"finish_reason":"stop"}]}""",
                    "[DONE]",
                ),
            ).throttleBody(64, 5, TimeUnit.MILLISECONDS),
        )
        val events = client.stream(request(LlmModels.GROQ_LLAMA_70B)).toList()
        val text = events.filterIsInstance<LlmEvent.Token>().joinToString("") { it.text }
        assertEquals("Tomorrow in Bangalore expect rain.", text)
        val done = events.last() as LlmEvent.Done
        assertTrue(done.stats.ttftMs >= 0)
        assertEquals(2, done.stats.chunks)

        val rec = server.takeRequest()
        assertEquals("/chat/completions", rec.path)
        assertEquals("Bearer test-key", rec.getHeader("Authorization"))
        val body = JSONObject(rec.body.readUtf8())
        assertEquals("llama-3.3-70b-versatile", body.getString("model"))
        assertEquals(true, body.getBoolean("stream"))
        assertEquals(160, body.getInt("max_tokens"))
        assertEquals(0.2, body.getDouble("temperature"), 1e-9)
        assertEquals(4, body.getJSONArray("messages").length())   // system + history pair + user
    }

    @Test fun geminiSseWithThinkingDisabled() = runBlocking {
        server.enqueue(
            MockResponse().setHeader("Content-Type", "text/event-stream").setBody(
                sse(
                    """{"candidates":[{"content":{"parts":[{"text":"Kal subah 7:00 AM ka "}],"role":"model"}}]}""",
                    """{"candidates":[{"content":{"parts":[{"text":"alarm set ho gaya hai."}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":42}}""",
                ),
            ),
        )
        val events = client.stream(request(LlmModels.GEMINI_FLASH)).toList()
        assertEquals("Kal subah 7:00 AM ka alarm set ho gaya hai.",
            events.filterIsInstance<LlmEvent.Token>().joinToString("") { it.text })
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
        server.enqueue(MockResponse().setBody(sse("""{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}""")))
        client.stream(request(LlmModels.GEMINI_FLASH_LITE)).toList()
        val rec = server.takeRequest()
        assertEquals("/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse", rec.path)
        val think = JSONObject(rec.body.readUtf8()).getJSONObject("generationConfig").getJSONObject("thinkingConfig")
        assertEquals("minimal", think.getString("thinkingLevel"))
    }

    @Test fun retiredModelFallsBackOnceBeforeAnyToken() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(404)
            .setBody("""{"error":{"code":404,"message":"This model is no longer available to new users."}}"""))
        server.enqueue(MockResponse().setBody(sse("""{"candidates":[{"content":{"parts":[{"text":"Hello from the fallback."}]},"finishReason":"STOP"}]}""")))
        val events = client.stream(request(LlmModels.GEMINI_FLASH_LITE)).toList()
        assertEquals("Hello from the fallback.", events.filterIsInstance<LlmEvent.Token>().joinToString("") { it.text })
        val done = events.last() as LlmEvent.Done
        assertTrue(done.stats.fellBack)
        assertEquals("gemini-flash-lite-latest", done.stats.modelId)
        assertEquals("/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse", server.takeRequest().path)
        assertEquals("/models/gemini-flash-lite-latest:streamGenerateContent?alt=sse", server.takeRequest().path)
    }

    @Test fun httpErrorsSurfaceTheProviderMessage() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(400)
            .setBody("""{"error":{"message":"The model `llama-x` has been decommissioned","code":"model_decommissioned"}}"""))
        val events = client.stream(request(LlmModels.GROQ_LLAMA_8B)).toList()
        val err = events.single() as LlmEvent.Error
        assertEquals(400, err.httpCode)
        assertTrue(err.message.contains("decommissioned"))
    }

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
