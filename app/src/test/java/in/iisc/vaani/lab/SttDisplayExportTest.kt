package `in`.iisc.vaani.lab

import `in`.iisc.vaani.codec.EnglishLexicon
import `in`.iisc.vaani.codec.HinglishCodec
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import java.io.File

/**
 * Second half of the closed-loop bridge (the first is [ClosedLoopFixtureTest]): applies the
 * app's exact STT display pipeline, [HinglishCodec.displayTranscript] — the text the feed shows
 * and the LLM receives — to the raw Whisper transcripts written by
 * `speech_lab/verify_closed_loop.py`, so the lab scores what ships instead of a Python port.
 * A no-op unless the lab wrote `build/vaani/closed_loop_raw_transcripts.json`.
 */
class SttDisplayExportTest {

    private val moduleDir = File(".").absoluteFile
    private val inFile = File(moduleDir, "build/vaani/closed_loop_raw_transcripts.json")
    private val outFile = File(moduleDir, "build/vaani/closed_loop_display_transcripts.json")

    @Test fun exportDisplayTranscripts() {
        if (!inFile.exists()) return
        // As in the app (VaaniRuntime installs it before the first transcript).
        HinglishCodec.englishLexicon = EnglishLexicon.parse(File(moduleDir, "src/main/assets/lexicon/vaani_en_cmudict.txt").readText())
        val items = JSONArray(inFile.readText())
        repeat(200) { HinglishCodec.displayTranscript("Mohjee kaal subah, 7 bajyakal alarm sate karna hai", "hi") } // JIT warm-up
        val out = JSONArray()
        for (i in 0 until items.length()) {
            val o = items.getJSONObject(i)
            val raw = o.getString("raw")
            val lang = o.getString("lang")
            val display = HinglishCodec.displayTranscript(raw, lang)
            val t0 = System.nanoTime()
            repeat(20) { HinglishCodec.displayTranscript(raw, lang) }
            val us = (System.nanoTime() - t0) / 20 / 1000.0
            out.put(JSONObject().put("key", o.getString("key")).put("display", display).put("us", us))
        }
        outFile.writeText(out.toString(1))
    }
}
