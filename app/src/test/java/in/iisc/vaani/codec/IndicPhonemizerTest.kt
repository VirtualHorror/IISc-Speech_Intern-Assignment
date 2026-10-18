package `in`.iisc.vaani.codec

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies Devanagari -> IPA G2P (including Hindi schwa deletion) and the
 * Piper/VITS phoneme-id layout (BOS ^, interspersed 0 blanks, EOS $).
 */
class IndicPhonemizerTest {

    private val schwa = "ə"   // ə

    @Test fun schwaDeletionWordFinal() {
        // कमल -> k ə m ə l  (the inherent final schwa after ल is deleted).
        assertEquals("k${schwa}m${schwa}l", IndicPhonemizer.textToPhonemes("कमल").joinToString(""))
    }

    @Test fun viramaSuppressesSchwa_consonantPlusMatra() {
        // नमस्ते -> n ə m ə s t̪ eː : virama kills the schwa on स, ते carries eː.
        val ph = IndicPhonemizer.textToPhonemes("नमस्ते")
        assertEquals(7, ph.size)
        assertEquals("n", ph[0])
        assertEquals(schwa, ph[1])
        assertEquals("m", ph[2])
        assertEquals(schwa, ph[3])
        assertEquals("s", ph[4])
        assertTrue("dental t expected", ph[5].startsWith("t"))   // t̪
        assertTrue("long e expected", ph[6].startsWith("e"))     // eː
    }

    @Test fun independentVowelAndMatra() {
        // आप -> aː p  (independent आ, then प with final schwa deleted).
        val ph = IndicPhonemizer.textToPhonemes("आप")
        assertEquals(listOf("aː", "p"), ph)
    }

    @Test fun phonemeIdLayout_bosBlankEos() {
        val idMap = mapOf(
            "^" to intArrayOf(1),
            "\$" to intArrayOf(2),
            "k" to intArrayOf(10),
            schwa to intArrayOf(11),
            "m" to intArrayOf(12),
            "l" to intArrayOf(13),
        )
        // कमल -> ^ , (k 0)(ə 0)(m 0)(ə 0)(l 0) , $
        val expected = intArrayOf(1, 10, 0, 11, 0, 12, 0, 11, 0, 13, 0, 2)
        assertArrayEquals(expected, IndicPhonemizer.textToPhonemeIds("कमल", idMap))
    }

    @Test fun unknownPhonemesAreSkippedGracefully() {
        // An empty map should yield an empty id stream, not a crash.
        assertEquals(0, IndicPhonemizer.textToPhonemeIds("कमल", emptyMap()).size)
    }
}
