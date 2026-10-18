package `in`.iisc.vaani.codec

import `in`.iisc.vaani.neural.VaaniTtsConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * Language-aware acoustic script: English replies are read as Indian English, Hinglish
 * replies with Hindi numerals/clock idioms; every Latin word is routed to Devanagari.
 */
class HinglishCodecLanguageTest {

    companion object {
        @BeforeClass @JvmStatic fun installLexicon() {
            HinglishCodec.englishLexicon =
                EnglishLexicon.parse(File("src/main/assets/lexicon/vaani_en_cmudict.txt").readText())
        }
    }

    private fun tts(s: String, lang: String) = HinglishCodec.toAcousticDevanagariScript(s, lang)

    @Test fun lexiconLoadsAndLooksUp() {
        val lex = HinglishCodec.englishLexicon!!
        assertTrue(lex.size > 100_000)
        assertEquals(listOf("W", "EH1", "DH", "ER0"), lex.phones("Weather"))
        assertEquals(null, lex.phones("vaaniedgeqq"))
    }

    @Test fun englishNumbersTimesAndTemperatures() {
        assertEquals("फ़ोर थर्टी पी एम", tts("4:30 PM", "en"))
        assertEquals("सेवन ए एम", tts("7:00 AM", "en"))
        assertEquals("ट्वेंटी एईट डिग्रीज़ सेल्सियस", tts("28°C", "en"))
        assertEquals("एटी फ़ोर परसेंट", tts("84%", "en"))
    }

    @Test fun hindiClockIdioms() {
        assertEquals("साढ़े चार पी एम", tts("4:30 PM", "hi"))
        assertEquals("सवा सात ए एम", tts("7:15 AM", "hi"))
        assertEquals("पौने छह पी एम", tts("5:45 PM", "hi"))
        assertEquals("चौरासी प्रतिशत", tts("84%", "hi"))
        assertEquals("उन्नीस दशमलव सात", tts("19.7", "hi"))
    }

    @Test fun englishSentenceUsesTheLexiconNotLetterSpelling() {
        val s = tts("What is the weather forecast for Bangalore tomorrow?", "en")
        assertEquals("व्हाट इज़ द वेदर फोरकास्ट फॉर बैंगलोर टुमारो?", s)
        // CMUdict path for words outside the curated list:
        assertEquals("ड्रिज़ल", tts("drizzle", "en"))
        assertTrue(tts("umbrella pleasant", "en").startsWith("अम्ब्रेला "))
    }

    @Test fun hinglishWordsStayHindiEvenWhenTheyAreEnglishSpellings() {
        // "to", "the", "is", "tak" are Hindi words in a Hinglish sentence.
        assertEquals("तो कल बारिश थे तक", tts("to kal baarish the tak", "hi"))
        assertEquals("कल बैंगलोर में बारिश की संभावना है, टेम्परेचर बीस से अट्ठाईस डिग्री तक रहेगा.",
            tts("Kal Bangalore mein baarish ki sambhavna hai, temperature 20 se 28 degree tak rahega.", "hi"))
    }

    @Test fun acronymsAreSpelled() {
        assertEquals("डी एस पी", tts("DSP", "en"))
        assertEquals("आई आई एस सी", tts("IISc", "en"))
        assertEquals("ओके", tts("OK", "en"))
    }

    @Test fun languageDetection() {
        assertEquals("en", HinglishCodec.detectLanguage("Your alarm is set for 7:00 AM tomorrow."))
        assertEquals("hi", HinglishCodec.detectLanguage("Aapka alarm kal subah 7 baje ke liye set ho gaya hai."))
        assertEquals("hi", HinglishCodec.detectLanguage("5 PM"))
    }

    @Test fun actionTagsAreNeverVoiced() {
        assertEquals("डन", tts("[ACTION:SET_REMINDER|17:00|Meeting] Done", "en"))
    }

    @Test fun asrStuttersAreRemovedBeforeTheLlm() {
        assertEquals("muji kalsuba, 7 baji ka alarm is 8 karna hai",
            HinglishCodec.cleanAsrText("muji kalsuba, 7-7 baji ka alarm is 8 karna hai"))
        assertEquals("set the reminder", HinglishCodec.cleanAsrText("set the the reminder"))
        assertEquals("4:30 PM", HinglishCodec.cleanAsrText("4:30 PM"))
    }

    @Test fun romanHindiTransliteration() {
        assertEquals("बारिश", HinglishCodec.romanHindiToDevanagari("baarish"))
        assertEquals("मेरा", HinglishCodec.romanHindiToDevanagari("mera"))
        assertEquals("सुबह", HinglishCodec.romanHindiToDevanagari("subah"))
    }

    @Test fun responsesInBothLanguagesHaveFullPhonemeCoverage() {
        val cfg = VaaniTtsConfig.fromJson(File("src/main/assets/models/vaani_tts_config.json").readText())
        val replies = listOf(
            "Tomorrow in Bangalore, expect thunderstorms with a high of 28°C and a 84% chance of rain." to "en",
            "Today you have three meetings, and your next one is the IISc Computational Intelligence Lab Review at 4:30 PM." to "en",
            "Theek hai, maine aapki meeting ke liye 5:00 PM ka reminder set kar diya hai." to "hi",
            "Aaj aapki teen meetings hain, aur aapki next meeting 6:00 PM par Edge DSP Telemetry Benchmark hai." to "hi",
        )
        for ((text, lang) in replies) {
            val dev = tts(text, lang)
            assertTrue("Latin left in: $dev", dev.none { it in 'a'..'z' || it in 'A'..'Z' })
            val cov = IndicPhonemizer.coverage(dev, cfg.phonemeIdMap)
            assertEquals("dropped ${cov.missing} for $dev", 0, cov.dropped)
        }
    }

    @Test fun eightNoLongerSoundsLikeAt() {
        // एट was shared by "eight" and "at": "28.3" was heard as "twenty at point three".
        assertEquals("ट्वेंटी एईट पॉइंट थ्री डिग्रीज़ सेल्सियस", tts("28.3°C", "en"))
        assertEquals("एटी फ़ोर परसेंट", tts("84%", "en"))
        assertEquals("एटीन", tts("18", "en"))
        assertEquals("मीटिंग एट फ़ाइव पी एम", tts("meeting at 5 PM", "en"))
    }

    @Test fun amPmIsNeverSplitIntoTwoSentences() {
        for (s in listOf("5 PM", "5 P.M.", "5 P. M.", "5 p.m.", "5 pm")) assertEquals(s, "फ़ाइव पी एम", tts(s, "en").trimEnd('.'))
        assertEquals("पाँच पी एम", tts("5:00 P. M.", "hi").trimEnd('.'))
        assertEquals("सात ए एम", tts("7 a. m.", "hi").trimEnd('.'))
        assertTrue(!tts("Your meeting is at 5 P. M. today.", "en").contains("पी. एम"))
    }
}
