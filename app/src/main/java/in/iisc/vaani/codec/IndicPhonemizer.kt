package `in`.iisc.vaani.codec

/**
 * Pure-Kotlin Grapheme-to-Phoneme (G2P) engine for the Indic VITS voice.
 *
 * Converts phonetic Devanagari (the output of [HinglishCodec.toAcousticDevanagariScript])
 * — with a light English fallback — into eSpeak-style IPA phonemes, then into the
 * integer phoneme ids expected by `vaani_tts_vits.int8.onnx`.
 *
 * Implements Hindi **schwa deletion**: the inherent vowel `ə` that follows a bare
 * consonant is dropped word-finally (and suppressed by an explicit virama),
 * which is what makes `कमल` render as `/kəməl/` rather than `/kəmələ/`.
 *
 * The phoneme id layout matches Piper/VITS: a leading BOS (`^`), each phoneme
 * followed by an interspersed `0` blank, and a trailing EOS (`$`).
 *
 * No Android or native dependencies — fully unit-testable on the host JVM.
 */
object IndicPhonemizer {

    private const val VIRAMA = '्'
    private const val ANUSVARA = 'ं'
    private const val CHANDRABINDU = 'ँ'
    private const val VISARGA = 'ः'
    private const val NUKTA = '़'
    private const val SCHWA = "ə"
    private const val BLANK_ID = 0

    // --- Devanagari -> IPA tables (standard Hindi phonology) ------------------
    private val CONS: Map<Char, String> = mapOf(
        'क' to "k", 'ख' to "kʰ", 'ग' to "ɡ", 'घ' to "ɡʱ", 'ङ' to "ŋ",
        'च' to "tʃ", 'छ' to "tʃʰ", 'ज' to "dʒ", 'झ' to "dʒʱ", 'ञ' to "ɲ",
        'ट' to "ʈ", 'ठ' to "ʈʰ", 'ड' to "ɖ", 'ढ' to "ɖʱ", 'ण' to "ɳ",
        'त' to "t̪", 'थ' to "t̪ʰ", 'द' to "d̪", 'ध' to "d̪ʱ", 'न' to "n",
        'प' to "p", 'फ' to "pʰ", 'ब' to "b", 'भ' to "bʱ", 'म' to "m",
        'य' to "j", 'र' to "ɾ", 'ल' to "l", 'व' to "ʋ",
        'श' to "ʃ", 'ष' to "ʂ", 'स' to "s", 'ह' to "ɦ", 'ळ' to "ɭ",
        '\u0958' to "q", '\u0959' to "x", '\u095a' to "ɣ", '\u095b' to "z", '\u095c' to "ɽ", '\u095d' to "ɽʱ", '\u095f' to "f",
    )
    private val MATRA: Map<Char, String> = mapOf(
        'ा' to "aː", 'ि' to "ɪ", 'ी' to "iː", 'ु' to "ʊ",
        'ू' to "uː", 'ृ' to "ɾɪ", 'े' to "eː", 'ै' to "ɛː",
        'ो' to "oː", 'ौ' to "ɔː",
    )
    private val VOWEL: Map<Char, String> = mapOf(
        'अ' to "ə", 'आ' to "aː", 'इ' to "ɪ", 'ई' to "iː", 'उ' to "ʊ", 'ऊ' to "uː",
        'ऋ' to "ɾɪ", 'ए' to "eː", 'ऐ' to "ɛː", 'ओ' to "oː", 'औ' to "ɔː", 'ऑ' to "ɒ",
    )

    /** Minimal English letter -> IPA fallback (used only for stray Latin tokens). */
    private val EN_FALLBACK: Map<Char, String> = mapOf(
        'a' to "ə", 'b' to "b", 'c' to "k", 'd' to "d̪", 'e' to "eː", 'f' to "f",
        'g' to "ɡ", 'h' to "ɦ", 'i' to "ɪ", 'j' to "dʒ", 'k' to "k", 'l' to "l",
        'm' to "m", 'n' to "n", 'o' to "oː", 'p' to "p", 'q' to "k", 'r' to "ɾ",
        's' to "s", 't' to "t̪", 'u' to "ʊ", 'v' to "ʋ", 'w' to "ʋ", 'x' to "k",
        'y' to "j", 'z' to "z",
    )

    private fun isDevanagari(c: Char): Boolean = c in 'ऀ'..'ॿ'

    /** Converts a single word of Devanagari into IPA phonemes (with schwa deletion). */
    private fun devWordToPhonemes(word: String): List<String> {
        val res = ArrayList<String>()
        var pending: String? = null   // consonant awaiting a vowel

        fun flush(withSchwa: Boolean) {
            pending?.let {
                res.add(it)
                if (withSchwa) res.add(SCHWA)
            }
            pending = null
        }

        for (c in word) {
            when {
                c == NUKTA -> { /* folded into precomposed consonants already */ }
                CONS.containsKey(c) -> { flush(true); pending = CONS.getValue(c) }
                MATRA.containsKey(c) -> { pending?.let { res.add(it) }; pending = null; res.add(MATRA.getValue(c)) }
                c == VIRAMA -> { pending?.let { res.add(it) }; pending = null }
                VOWEL.containsKey(c) -> { flush(true); res.add(VOWEL.getValue(c)) }
                c == ANUSVARA || c == CHANDRABINDU -> { flush(true); res.add("n") }
                c == VISARGA -> { flush(true); res.add("h") }
            }
        }
        flush(withSchwa = false)   // word-final inherent schwa is deleted
        return res
    }

    private fun enWordToPhonemes(word: String): List<String> =
        word.lowercase().mapNotNull { EN_FALLBACK[it] }

    /**
     * Converts arbitrary mixed text into a flat IPA phoneme sequence.
     * Devanagari words use full Hindi G2P; Latin words use the light fallback.
     */
    fun textToPhonemes(text: String): List<String> {
        val out = ArrayList<String>()
        for (word in text.trim().split(Regex("\\s+"))) {
            if (word.isEmpty()) continue
            if (word.any { isDevanagari(it) }) out.addAll(devWordToPhonemes(word))
            else out.addAll(enWordToPhonemes(word))
        }
        return out
    }

    /** Looks up a phoneme in the model's map, splitting multi-glyph IPA if needed. */
    private fun idsForPhoneme(ph: String, idMap: Map<String, IntArray>): List<Int> {
        idMap[ph]?.let { return it.toList() }
        val acc = ArrayList<Int>()
        for (ch in ph) idMap[ch.toString()]?.let { acc.addAll(it.toList()) }
        return acc
    }

    /**
     * Full text -> VITS phoneme id sequence:
     *   [ ^ ]  then for each phoneme [ id(s), 0 ]  then [ $ ].
     *
     * @param phonemeIdMap the `phoneme_id_map` from `vaani_tts_config.json`.
     */
    fun textToPhonemeIds(text: String, phonemeIdMap: Map<String, IntArray>): IntArray {
        val ids = ArrayList<Int>()
        phonemeIdMap["^"]?.let { ids.addAll(it.toList()) }      // BOS
        for (ph in textToPhonemes(text)) {
            val mapped = idsForPhoneme(ph, phonemeIdMap)
            if (mapped.isEmpty()) continue                      // unknown phoneme: emit neither id nor blank
            ids.addAll(mapped)
            ids.add(BLANK_ID)                                   // interspersed blank
        }
        phonemeIdMap["\$"]?.let { ids.addAll(it.toList()) }     // EOS
        return ids.toIntArray()
    }
}
