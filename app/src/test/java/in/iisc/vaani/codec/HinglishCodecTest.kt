package `in`.iisc.vaani.codec

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/**
 * Verifies the Hinglish Codec against the four IISc evaluation prompts in both
 * directions, numeric/temporal/temperature handling, the Hunterian transliteration
 * fallback, and the STT post-corrector on noisy Whisper-tiny Hinglish output.
 */
class HinglishCodecTest {

    companion object {
        /** The app always installs the CMU lexicon before the first transcript (VaaniRuntime). */
        @BeforeClass @JvmStatic fun installLexicon() {
            HinglishCodec.englishLexicon =
                EnglishLexicon.parse(File("src/main/assets/lexicon/vaani_en_cmudict.txt").readText())
        }
    }

    private fun hi(raw: String) = HinglishCodec.displayTranscript(raw, "hi")
    private fun en(raw: String) = HinglishCodec.displayTranscript(raw, "en")

    // ===== STT normalisation: Devanagari -> Roman Hinglish/English ==========

    @Test fun prompt1_weatherForecast() {
        val dev = "वेदर फोरकास्ट फॉर बैंगलोर"
        assertEquals("weather forecast for Bangalore", HinglishCodec.normalizeSttTranscript(dev))
    }

    @Test fun prompt2_reminderMeeting() {
        val dev = "रिमाइंडर फॉर माय मीटिंग एट 5 PM"
        assertEquals("reminder for my meeting at 5 PM", HinglishCodec.normalizeSttTranscript(dev))
    }

    @Test fun prompt3_alarmHinglish() {
        val dev = "मुझे कल सुबह 7 बजे का अलार्म सेट करना है"
        assertEquals(
            "Mujhe kal subah 7 baje ka alarm set karna hai",
            HinglishCodec.normalizeSttTranscript(dev),
        )
    }

    @Test fun prompt4_scheduleHinglishWithComma() {
        val dev = "आज का शेड्यूल क्या है, और मेरी नेक्स्ट मीटिंग कब है"
        assertEquals(
            "Aaj ka schedule kya hai, aur meri next meeting kab hai",
            HinglishCodec.normalizeSttTranscript(dev),
        )
    }

    // ===== TTS acoustic script: Roman/English + numerics -> Devanagari ======

    @Test fun tts_bareClockPm() {
        assertEquals("पाँच पी एम", HinglishCodec.toAcousticDevanagariScript("5 PM"))
    }

    @Test fun tts_colonTimeAm() {
        assertEquals("सात ए एम", HinglishCodec.toAcousticDevanagariScript("7:00 AM"))
    }

    @Test fun tts_temperatureCelsius() {
        assertEquals("अट्ठाईस डिग्री सेल्सियस", HinglishCodec.toAcousticDevanagariScript("28°C"))
    }

    @Test fun tts_sentenceWeather() {
        assertEquals(
            "वेदर फोरकास्ट फॉर बैंगलोर",
            HinglishCodec.toAcousticDevanagariScript("weather forecast for Bangalore"),
        )
    }

    @Test fun tts_sentenceAlarm() {
        assertEquals(
            "मुझे कल सुबह सात बजे का अलार्म सेट करना है",
            HinglishCodec.toAcousticDevanagariScript("Mujhe kal subah 7 baje ka alarm set karna hai"),
        )
    }

    // ===== Fallbacks & edge cases ===========================================

    @Test fun hunterianFallbackDropsFinalSchwa() {
        // "कमल" is not in the lexicon -> algorithmic transliteration, final schwa dropped.
        assertEquals("kamal", HinglishCodec.hunterianTransliterate("कमल"))
        assertEquals("kamal", HinglishCodec.normalizeSttTranscript("कमल"))
    }

    @Test fun emptyInputs() {
        assertEquals("", HinglishCodec.normalizeSttTranscript(""))
        assertEquals("", HinglishCodec.toAcousticDevanagariScript("   "))
    }

    @Test fun romanPassthroughUnchanged() {
        // Already-Roman STT output should survive normalisation intact.
        val roman = "set alarm for 7"
        assertTrue(HinglishCodec.normalizeSttTranscript(roman).contains("alarm"))
    }

    // ===== Whisper Roman-Hinglish post-correction ==========================

    @Test fun noisyAlarmPromptFromWhisperNormalisesToCleanHinglish() {
        assertEquals("Mujhe kal subah 7 baje ka alarm set karna hai",
            hi("Mohjee kaal subah, 7 bajyakal alarm sate karna hai"))
        assertEquals("mujhe kal subah 7 baje ka alarm set karna hai",              // Whisper's casing is kept
            hi("muji kalsuba, 7-7 baji ka alarm isaate karna hai"))
        assertEquals("Mujhe kal subah 7 baje ka alarm set karna hai", hi("Mohjee Kalsuba, Saad Bhajee ka alarm sate karna hai"))
    }

    @Test fun fusedWhisperCompoundsAreSplit() {
        val cases = linkedMapOf(
            "kalsubah" to "kal subah", "kalsuba" to "kal subah", "kalsooba" to "kal subah", "bajeka" to "baje ka",
            "aajka" to "aaj ka", "kyahai" to "kya hai", "kabhai" to "kab hai", "merinext" to "meri next",
            "karnaa" to "karna", "karnahai" to "karna hai", "hogaya" to "ho gaya", "kardiya" to "kar diya",
        )
        for ((noisy, clean) in cases) assertEquals(noisy, "mujhe $clean", hi("mujhe $noisy"))
        assertEquals("7 baje ka alarm", hi("7 bajyakal alarm"))
    }

    @Test fun phoneticSpellingVariantsResolveToTheCanonicalWord() {
        val cases = linkedMapOf(
            "mohjee" to "mujhe", "muje" to "mujhe", "mujhey" to "mujhe", "subha" to "subah", "suba" to "subah",
            "bajey" to "baje", "bhajee" to "baje", "baaje" to "baje", "kyaa" to "kya", "rahegaa" to "rahega",
            "sham" to "shaam", "aapkaa" to "aapka",
        )
        for ((noisy, clean) in cases) assertEquals(noisy, "aur $clean", hi("aur $noisy"))
        // Two canonical words share the key (mere/meri): left alone rather than guessed.
        assertEquals("aur meree", hi("aur meree"))
    }

    @Test fun contextDecidesHomophones() {
        assertEquals("kal subah", hi("kaal subah"))
        assertEquals("kal shaam 6 baje", hi("kaal sham 6 baje"))
        assertEquals("kaal chakra", hi("kaal chakra"))                       // no time word: untouched
        assertEquals("alarm set karna hai", hi("alarm sate karna hai"))
        assertEquals("reminder set kar do", hi("reminder seat kar do"))
        assertEquals("woh sat gaya", hi("woh sat gaya"))                     // no alarm/verb context
        assertEquals("7 baje ka alarm", hi("saat baje ka alarm"))            // number word before baje
        assertEquals("subah 9 baje", hi("subah nau baje"))
    }

    @Test fun englishDomainMisspellingsAreFixedInBothLanguages() {
        assertEquals("What is the weather forecast for Bangalore tomorrow?", en("What is the wether forcast for banglor tomorrow?"))
        assertEquals("weather forecast for Bangalore", en("weather fore cast for bengaluru"))
        assertEquals("aaj ka schedule kya hai", hi("aaj ka skedule kya hai"))
        assertEquals("meri next meeting kab hai", hi("meri next miting kab hai"))
    }

    @Test fun cleanTranscriptsAreLeftUntouched() {
        for (s in listOf(
            "Mujhe kal subah 7 baje ka alarm set karna hai.",
            "Aaj ka schedule kya hai, aur meri next meeting kab hai?",
            "Kal Bangalore mein baarish ki sambhavna hai, temperature 20 se 28 degree tak rahega.",
            "Theek hai, maine aapki meeting ke liye 5 PM ka reminder set kar diya hai.",
        )) assertEquals(s, hi(s))
        for (s in listOf(
            "What is the weather forecast for Bangalore tomorrow?",
            "Can you set a reminder for my meeting at 5 PM?",
            "I sat on the seat near the window.",
            "Do it now, the meeting is at 4:30 PM.",
        )) assertEquals(s, en(s))
        // English that Whisper produced for Hinglish audio is not "corrected" into Hindi.
        assertEquals("Today's schedule is our next meeting.", hi("Today's schedule is our next meeting."))
        assertEquals("I am going to say that you have made a mistake.", hi("I am going to say that you have made a mistake."))
    }

    @Test fun withoutTheEnglishLexiconOnlyExplicitTablesApply() {
        val lexicon = HinglishCodec.englishLexicon
        try {
            HinglishCodec.englishLexicon = null
            assertEquals("I am going to say that you have made a mistake.", hi("I am going to say that you have made a mistake."))
            assertEquals("aur mohjee", hi("aur mohjee"))                       // fuzzy key match needs the guard
            assertEquals("kal subah", hi("kaal suba"))                         // explicit table + context rule
        } finally {
            HinglishCodec.englishLexicon = lexicon
        }
    }

    @Test fun hinglishTextIsCorrectedEvenWhenLidSaidEnglish() {
        assertEquals("Mujhe kal subah 7 baje ka alarm set karna hai", en("Mohjee kal subah 7 baje ka alarm sate karna hai"))
    }

    @Test fun shoutedTranscriptsAreLowerCased() {
        assertEquals("Aapka alarm kal subah 7 baje set ho gaya", hi("AAPKA ALARM KAL SUBHA SAAT BAJE SEAT HOGAYA"))
    }

    @Test fun postCorrectorRunsWellUnderHalfAMillisecond() {
        val noisy = listOf(
            "Mohjee kaal subah, 7 bajyakal alarm sate karna hai", "Maji Kalsooba, 7-7 baji ka alarm isaate karna hai",
            "Today's schedule is our next meeting.", "AAPKA ALARM KALS BASAT MAJI KELI SATE HONGI AHE",
            "Can you set a remainder for my miting at 5 p.m.?", "aajka skedule kyahai aur merinext miting kabhai",
        )
        repeat(2_000) { for (s in noisy) hi(s) }                             // JIT warm-up
        val n = 5_000
        val t0 = System.nanoTime()
        repeat(n) { for (s in noisy) hi(s) }
        val usPerCall = (System.nanoTime() - t0) / 1_000.0 / (n * noisy.size)
        assertTrue("$usPerCall us per transcript", usPerCall < 500.0)
    }
}
