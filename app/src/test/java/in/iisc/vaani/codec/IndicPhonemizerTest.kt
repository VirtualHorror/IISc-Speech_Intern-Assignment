package `in`.iisc.vaani.codec

import `in`.iisc.vaani.neural.VaaniTtsConfig
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.text.Normalizer

/**
 * Verifies the Devanagari -> IPA G2P against the conventions of the voice's training
 * phonemizer (espeak-ng `hi`; expected strings below are its verbatim output), the
 * Piper id layout (`^ _` + every code point followed by `_` + `$`), and 100% coverage
 * of the shipped `phoneme_id_map`.
 */
class IndicPhonemizerTest {

    /** Model ids are looked up per NFD code point, so compare in NFD. */
    private fun nfd(s: String) = Normalizer.normalize(s, Normalizer.Form.NFD)
    private fun ipa(w: String) = nfd(IndicPhonemizer.wordToIpa(w))
    private fun assertIpa(expected: String, actual: String) = assertEquals(nfd(expected), actual)

    @Test fun schwaDeletionWordFinal() {
        // कमल -> k ʌ m ə l : stressed inherent schwa is ʌ, final schwa deleted.
        assertIpa("kˈʌməl", ipa("कमल"))
    }

    @Test fun viramaSuppressesSchwa_consonantPlusMatra() {
        assertIpa("nəmˈʌsteː", ipa("नमस्ते"))
    }

    @Test fun independentVowelAndMatra() {
        assertIpa("ˌaːp", ipa("आप"))          // pronoun: secondary stress only
        assertIpa("pˈaːni", ipa("पानी"))     // word-final ी is short
    }

    @Test fun breathyConsonantsUseAspirationMarkPresentInIdMap() {
        assertIpa("ɡʰˈʌɾ", ipa("घर"))
        assertIpa("ɟʰˈʌɳɖaː", ipa("झंडा"))
        assertIpa("ɖʰˈoːl", ipa("ढोल"))
        assertIpa("dʰˈʌn", ipa("धन"))
        assertIpa("bʰˈaːɾət", ipa("भारत"))
        assertIpa("pˈʌr.hənˌaː", ipa("पढ़ना"))          // decomposed ढ + nukta
        assertIpa("pˈʌr.hənˌaː", ipa("पढ़ना"))     // precomposed ढ़
        for (w in listOf("घर", "झंडा", "ढोल", "धन", "भारत", "पढ़ना")) {
            assertTrue("no breathy ʱ in $w", 'ʱ' !in ipa(w))
        }
    }

    @Test fun medialSchwaDeletionRightToLeft() {
        assertIpa("kˈʌɾnˌaː", ipa("करना"))
        assertIpa("sˈʌməɟʰnaː", ipa("समझना"))
        assertIpa("bˈʌdəlnaː", ipa("बदलना"))
        assertIpa("sˌəɾkˈaːɾ", ipa("सरकार"))
        assertIpa("kˈʌhənˌaː", ipa("कहना"))               // never deleted after ह
        assertIpa("mˈɪtɾə", ipa("मित्र"))                // kept after C+र conjunct
    }

    @Test fun nasalsAndStress() {
        assertIpa("hˈã", ipa("हाँ"))
        assertIpa("nˈʌhĩ", ipa("नहीं"))
        assertIpa("miʈˈĩɡ", ipa("मीटिंग"))
        assertIpa("bˌɛ̃ɡlˈoːɾ", ipa("बैंगलोर"))
        assertIpa("səŋɡˈiːt", ipa("संगीत"))
        assertIpa("pəɲɟˈaːb", ipa("पंजाब"))
    }

    @Test fun geminationAndClusters() {
        assertIpa("ˈʊtːəɾ", ipa("उत्तर"))
        assertIpa("kːjaː", ipa("क्या"))
        assertIpa("ˈʌcʰcʰaː", ipa("अच्छा"))
        assertIpa("ʌlˈaːrm", ipa("अलार्म"))
        assertIpa("kʃˈʌmaː", ipa("क्षमा"))
    }

    @Test fun sentenceMatchesEspeakVerbatim() {
        val s = IndicPhonemizer.phonemizeSentences("आज का शेड्यूल क्या है, और मेरी नेक्स्ट मीटिंग कब है?")
        assertEquals(1, s.size)
        assertIpa("ˈaːɟ kaː ʃˈeːɖjuːl kːjaː hɛː, ɔːɾ mˌeːɾi nˈeːksʈ miʈˈĩɡ kˈʌb hɛː?", nfd(s[0].ipa))
    }

    @Test fun sentencesAreSplitAtTerminators() {
        val s = IndicPhonemizer.phonemizeSentences("ठीक है। अलार्म सेट हो गया है।")
        assertEquals(2, s.size)
        assertTrue(s[0].ipa.endsWith("."))
    }

    @Test fun phonemeIdLayout_piperBosPadEos() {
        val idMap = mapOf(
            "_" to intArrayOf(0), "^" to intArrayOf(1), "\$" to intArrayOf(2),
            "k" to intArrayOf(10), "ˈ" to intArrayOf(11), "ʌ" to intArrayOf(12),
            "m" to intArrayOf(13), "ə" to intArrayOf(14), "l" to intArrayOf(15),
        )
        // कमल -> ^ _ (k _)(ˈ _)(ʌ _)(m _)(ə _)(l _) $
        val expected = intArrayOf(1, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 2)
        assertArrayEquals(expected, IndicPhonemizer.textToPhonemeIds("कमल", idMap))
    }

    @Test fun unknownPhonemesAreSkippedGracefully() {
        // An empty map should yield an empty id stream, not a crash.
        assertEquals(0, IndicPhonemizer.textToPhonemeIds("कमल", emptyMap()).size)
    }

    @Test fun fullCoverageOfShippedPhonemeIdMap() {
        val cfg = VaaniTtsConfig.fromJson(File("src/main/assets/models/vaani_tts_config.json").readText())
        val text = "मुझे कल सुबह सात बजे का अलार्म सेट करना है। घर झंडा ढोल धन भारत पढ़ना ड़ ज़रूर फ़ोन " +
            "क़ायदा ख़ुश ग़लत ऋषि कृपया क्षमा ज्ञान दुःख हाँ में हैं नहीं हूँ क्यों पाँच?"
        val cov = IndicPhonemizer.coverage(text, cfg.phonemeIdMap)
        assertEquals("dropped symbols: ${cov.missing}", 0, cov.dropped)
        assertEquals(100.0, cov.percent, 0.0)
    }
}
