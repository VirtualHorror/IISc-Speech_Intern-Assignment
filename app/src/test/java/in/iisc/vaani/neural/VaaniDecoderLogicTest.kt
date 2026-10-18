package `in`.iisc.vaani.neural

import `in`.iisc.vaani.agent.Metrics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** Pure decoder logic: the real vocab's logit masks, the loop guard, and WER. */
class VaaniDecoderLogicTest {

    private val vocab by lazy { VaaniVocab.fromJson(File("src/main/assets/models/vaani_vocab.json").readText()) }

    @Test fun stepZeroLanguageTokens() {
        assertEquals(50259, vocab.languageToken("en"))
        assertEquals(50276, vocab.languageToken("hi"))
        assertEquals(50258, vocab.sot)
        assertEquals(50363, vocab.noTimestamps)
    }

    @Test fun textMaskBlocksSpecialsNonSpeechAndScriptsButNotEot() {
        assertFalse(vocab.textMask[vocab.eot])
        assertTrue(vocab.firstStepMask[vocab.eot])                    // suppress_blank step
        assertTrue(vocab.firstStepMask[220])
        assertTrue(vocab.textMask[vocab.languageToken("hi")])         // no language token mid-text
        assertTrue(vocab.textMask[vocab.timestampBegin])
        assertTrue(vocab.textMask[3961])                              // "♪"
        assertTrue(vocab.textMask[58])                                // "["
        assertFalse(vocab.textMask[13])                               // "." stays allowed
        val masked = (0 until vocab.eot).count { vocab.textMask[it] }
        assertTrue("script + non-speech mask size $masked", masked in 7000..7400)
    }

    @Test fun detokenisesByteLevelBpe() {
        assertEquals("", vocab.decode(listOf(vocab.eot, vocab.sot)))
    }

    @Test fun repetitionGuard() {
        assertTrue(VaaniSttEngine.repeatingTail(listOf(1, 2, 7, 7, 7, 7)))
        assertTrue(VaaniSttEngine.repeatingTail(listOf(9, 4, 5, 4, 5, 4, 5, 4, 5)))
        assertFalse(VaaniSttEngine.repeatingTail(listOf(1, 2, 3, 4, 5, 6, 7, 8)))
    }

    @Test fun wordErrorRate() {
        assertEquals(0.0, Metrics.wordErrorRate("Can you set a reminder for my meeting at 5 PM?", "can you set a reminder for my meeting at 5 pm"), 1e-9)
        assertEquals(1.0 / 9, Metrics.wordErrorRate("Mujhe kal subah 7 baje ka alarm set karna", "Mujhe kal subah 7 baje ka alarm set krna"), 1e-9)
    }
}
