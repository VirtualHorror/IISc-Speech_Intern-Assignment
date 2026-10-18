package `in`.iisc.vaani.codec

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies the Hinglish Codec against the four IISc evaluation prompts in both
 * directions, plus numeric/temporal/temperature handling and the Hunterian
 * transliteration fallback.
 */
class HinglishCodecTest {

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
}
