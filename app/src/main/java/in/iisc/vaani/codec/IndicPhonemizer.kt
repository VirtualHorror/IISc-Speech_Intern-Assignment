package `in`.iisc.vaani.codec

import java.text.Normalizer

/**
 * Pure-Kotlin Grapheme-to-Phoneme (G2P) engine for the Indic VITS voice.
 *
 * Converts phonetic Devanagari (the output of [HinglishCodec.toAcousticDevanagariScript])
 * into the exact IPA conventions the Piper `hi_IN` voice was trained on, then into the
 * integer phoneme ids expected by `vaani_tts_vits.fp16.onnx`.
 *
 * The training front-end of that voice was espeak-ng (`hi`), whose output looks like
 * `ˈaːɟ kaː ʃˈeːɖjuːl kːjaː hɛː, ɔːɾ mˌeːɾi nˈeːksʈ miʈˈĩɡ kˈʌb hɛː?`. Matching it matters:
 * every code point is a separate model token, so a wrong symbol (`dʒ` for `ɟ`, the
 * breathy `ʱ` that is *absent* from `phoneme_id_map`, a dental `t̪` the voice never saw)
 * is either silently dropped or fed to the network out of distribution. The rules below
 * were derived from — and are scored against — espeak-ng by
 * `speech_lab/verify_closed_loop.py` (espeak-ng is a host-side measuring stick only;
 * nothing from it ships). Implemented from first principles:
 *
 *  - consonants: aspiration as `ʰ` (घ → ɡʰ, भ → bʰ, ढ़ → r.h), palatal stops c/ɟ, plain
 *    dentals t/d, `h` for ह, nukta letters (क़ q, ख़ x, ग़ ɣ, ज़ z, फ़ f, ड़ r.);
 *  - **schwa deletion**: word-final (kept after a C+र / C+य conjunct) and medial
 *    VC_CV deletion scanned right-to-left (करना → kˈʌɾnˌaː, समझना → sˈʌməɟʰnaː),
 *    never after ह or ढ़ (कहना → kˈʌhənˌaː);
 *  - **stress**: espeak's Hindi rule — the last heaviest non-final syllable, computed on
 *    the underlying (pre-deletion) syllables; light-only words stress the first; ˌ on
 *    alternate syllables; pronouns demoted to ˌ, clitics (है, का, को, में …) unstressed;
 *    stressed inherent schwa is `ʌ`, ी is long (`iː`) only under primary stress;
 *  - **nasals**: anusvara after schwa → homorganic nasal (ŋ ɲ ɳ n m), after a matra →
 *    nasalised vowel with U+0303 (हाँ → hˈã, में → mẽː);
 *  - **gemination** exactly as espeak spells it (उत्तर ʊtːəɾ, क्या kːjaː, अच्छा ʌcʰcʰaː).
 *
 * The id layout matches Piper's `phonemes_to_ids`: `^`, `_`, then every NFD code point
 * followed by `_`, then `$`; one sequence per sentence; words separated by ` `, commas and
 * full stops kept as tokens. No Android or native dependencies — fully unit-testable.
 */
object IndicPhonemizer {

    private const val VIRAMA = '्'
    private const val ANUSVARA = 'ं'
    private const val CHANDRABINDU = 'ँ'
    private const val VISARGA = 'ः'
    private const val NUKTA = '़'
    private const val TILDE = "̃"
    private const val PRIMARY = "ˈ"
    private const val SECONDARY = "ˌ"

    // --- Consonants -----------------------------------------------------------
    private val CONS: Map<Char, String> = mapOf(
        'क' to "k", 'ख' to "kʰ", 'ग' to "ɡ", 'घ' to "ɡʰ", 'ङ' to "ŋ",
        'च' to "c", 'छ' to "cʰ", 'ज' to "ɟ", 'झ' to "ɟʰ", 'ञ' to "ɲ",
        'ट' to "ʈ", 'ठ' to "ʈʰ", 'ड' to "ɖ", 'ढ' to "ɖʰ", 'ण' to "ɳ",
        'त' to "t", 'थ' to "tʰ", 'द' to "d", 'ध' to "dʰ", 'न' to "n",
        'प' to "p", 'फ' to "pʰ", 'ब' to "b", 'भ' to "bʰ", 'म' to "m",
        'य' to "j", 'र' to "ɾ", 'ल' to "l", 'व' to "ʋ", 'ळ' to "ɭ",
        'श' to "ʃ", 'ष' to "ʂ", 'स' to "s", 'ह' to "h",
        // precomposed nukta letters (U+0958..U+095F)
        'क़' to "q", 'ख़' to "x", 'ग़' to "ɣ", 'ज़' to "z",
        'ड़' to "r.", 'ढ़' to "r.h", 'फ़' to "f", 'य़' to "j",
    )

    /** Base consonant + U+093C (decomposed nukta) -> nukta variant. */
    private val NUKTA_FORM: Map<String, String> = mapOf(
        "k" to "q", "kʰ" to "x", "ɡ" to "ɣ", "ɟ" to "z", "ɖ" to "r.", "ɖʰ" to "r.h", "pʰ" to "f",
    )

    private val MATRA: Map<Char, String> = mapOf(
        'ा' to "aː", 'ि' to "ɪ", 'ी' to "iː", 'ु' to "ʊ", 'ू' to "uː", 'ृ' to "ɾɪ",
        'े' to "eː", 'ै' to "ɛː", 'ो' to "oː", 'ौ' to "ɔː", 'ॉ' to "ɔ", 'ॅ' to "ɛ",
    )
    private val VOWEL: Map<Char, String> = mapOf(
        'अ' to "ʌ", 'आ' to "aː", 'इ' to "ɪ", 'ई' to "iː", 'उ' to "ʊ", 'ऊ' to "uː", 'ऋ' to "ɾɪ",
        'ए' to "eː", 'ऐ' to "ɛː", 'ओ' to "oː", 'औ' to "ɔː", 'ऑ' to "ɔ", 'ऍ' to "ɛ",
    )
    private val LONG = setOf("aː", "iː", "uː", "eː", "oː", "ɛː", "ɔː")
    private val NASALISED: Map<String, String> = mapOf(
        "aː" to "a$TILDE", "iː" to "i$TILDE", "uː" to "u$TILDE", "eː" to "e${TILDE}ː",
        "oː" to "o$TILDE", "ɛː" to "ɛ$TILDE", "ɔː" to "ɔ$TILDE", "ɔ" to "ɔ$TILDE",
        "ɪ" to "i$TILDE", "ʊ" to "u$TILDE", "ɛ" to "ɛ$TILDE", "ʌ" to "ʌ$TILDE", "ə" to "ə$TILDE",
        "i" to "i$TILDE",
    )

    // Place of articulation for homorganic anusvara.
    private val VELAR = setOf("k", "kʰ", "ɡ", "ɡʰ", "q", "x", "ɣ")
    private val PALATAL = setOf("c", "cʰ", "ɟ", "ɟʰ", "j")
    private val RETROFLEX = setOf("ʈ", "ʈʰ", "ɖ", "ɖʰ", "ɳ")
    private val LABIAL = setOf("p", "pʰ", "b", "bʰ", "m")

    /** Geminate spellings, exactly as espeak-ng renders them. */
    private val GEMINATE: Map<Pair<String, String>, String> = mapOf(
        ("ʈ" to "ʈʰ") to "ʈʰʈʰ", ("c" to "cʰ") to "cʰcʰ", ("k" to "kʰ") to "kkʰ",
        ("t" to "tʰ") to "tʰː", ("d" to "dʰ") to "dʰː", ("p" to "pʰ") to "pʰː", ("b" to "bʰ") to "bʰː",
        ("ɡ" to "ɡʰ") to "ɡʰː", ("ɟ" to "ɟʰ") to "ɟʰː", ("ɖ" to "ɖʰ") to "ɖʰː", ("ɖ" to "ɖ") to "ɖɖ",
    )
    private val GEMINATE_BEFORE_Y = setOf("k", "c", "t", "p")
    /** Sonorant / fricative geminates are spelled doubled (दिल्ली -> dˈɪlli), stops with ː. */
    private val DOUBLED_GEMINATE = setOf("l", "n", "m", "ɾ", "s", "ʃ", "ʋ", "j", "ɳ", "ɖ")
    /** र before these is the trill `r` (अलार्म ʌlˈaːrm, मॉर्निंग mɔrnˈĩɡ), elsewhere the tap `ɾ`. */
    private val TRILL_CONTEXT = setOf("m", "n", "f", "s", "ʃ", "ʂ", "z")

    /** Clitics espeak-ng leaves unstressed. */
    private val UNSTRESSED = setOf(
        "है", "हैं", "हूँ", "हूं", "और", "का", "की", "के", "को", "में", "से", "था", "थी", "थे",
        "हो", "क्या", "मैं",
    )
    /** Pronouns espeak-ng demotes to secondary stress. */
    private val SECONDARY_ONLY = setOf(
        "यह", "वह", "ये", "वो", "मेरा", "मेरी", "मेरे", "मुझे", "मुझको", "तुम", "तुम्हें", "तुम्हारा",
        "तुम्हारी", "तुम्हारे", "आप", "आपका", "आपकी", "आपके", "आपको", "हम", "हमें", "हमारा",
        "हमारी", "हमारे", "उसका", "उसकी", "उसके", "इसका", "इसकी", "इसके", "तक",
    )

    // --- Word model ----------------------------------------------------------
    private enum class Kind { INHERENT, MATRA, INDEPENDENT }

    private class Nucleus(val ipa: String, val kind: Kind) {
        var nasal = false
        var chandrabindu = false
        var visarga = false
        var deleted = false
        var stress = 0          // 0 none, 1 secondary, 2 primary
        val isSchwa get() = kind == Kind.INHERENT || ipa == "ʌ" && kind == Kind.INDEPENDENT
        val isLong get() = ipa in LONG
    }

    /** A word is an alternation of consonant clusters and vowel nuclei. */
    private class Seg(val cons: String?, val nucleus: Nucleus?)

    private fun isDevanagari(c: Char): Boolean = c in 'ऀ'..'ॿ'

    private fun parse(word: String): List<Seg> {
        val segs = ArrayList<Seg>()
        var pending = false          // last segment is a consonant still owed a vowel
        fun close() {
            if (pending) segs.add(Seg(null, Nucleus("ə", Kind.INHERENT)))
            pending = false
        }
        fun lastNucleus(): Nucleus? = segs.lastOrNull { it.nucleus != null }?.nucleus
        for (c in word) {
            when {
                CONS.containsKey(c) -> { close(); segs.add(Seg(CONS.getValue(c), null)); pending = true }
                c == NUKTA -> {
                    val last = segs.lastOrNull()
                    if (last?.cons != null) NUKTA_FORM[last.cons]?.let { segs[segs.size - 1] = Seg(it, null) }
                }
                MATRA.containsKey(c) -> {
                    val v = MATRA.getValue(c)
                    if (v == "ɾɪ") segs.add(Seg("ɾ", null))                 // ृ = r + short i
                    segs.add(Seg(null, Nucleus(if (v == "ɾɪ") "ɪ" else v, Kind.MATRA)))
                    pending = false
                }
                c == VIRAMA -> pending = false
                VOWEL.containsKey(c) -> {
                    close()
                    val v = VOWEL.getValue(c)
                    if (v == "ɾɪ") segs.add(Seg("ɾ", null))
                    segs.add(Seg(null, Nucleus(if (v == "ɾɪ") "ɪ" else v, Kind.INDEPENDENT)))
                }
                c == ANUSVARA || c == CHANDRABINDU -> {
                    close()
                    lastNucleus()?.let { it.nasal = true; it.chandrabindu = c == CHANDRABINDU }
                }
                c == VISARGA -> { close(); lastNucleus()?.visarga = true }
                else -> { /* ZWJ/ZWNJ, avagraha, stray marks */ }
            }
        }
        close()                       // word-final consonant carries a (deletable) schwa
        return segs
    }

    /** Consonants strictly between nucleus i and the next nucleus (or word end). */
    private fun clusterAfter(segs: List<Seg>, nucIdx: List<Int>, i: Int): List<String> {
        val from = nucIdx[i] + 1
        val to = if (i + 1 < nucIdx.size) nucIdx[i + 1] else segs.size
        return (from until to).mapNotNull { segs[it].cons }
    }

    private fun clusterBefore(segs: List<Seg>, nucIdx: List<Int>, i: Int): List<String> {
        val from = if (i == 0) 0 else nucIdx[i - 1] + 1
        return (from until nucIdx[i]).mapNotNull { segs[it].cons }
    }

    /** Anusvara after a short vowel behaves like a coda consonant. */
    private fun consonantalNasal(n: Nucleus): Boolean = n.nasal && !n.isLong

    private fun applySchwaDeletion(segs: List<Seg>, nucIdx: List<Int>) {
        val nucs = nucIdx.map { segs[it].nucleus!! }
        val last = nucs.size - 1
        if (last < 0) return
        fun deletable(n: Nucleus) = n.kind == Kind.INHERENT && !n.nasal && !n.visarga

        // 1) word-final schwa (kept after C+र / C+य conjuncts and in one-vowel words)
        val fin = nucs[last]
        if (last > 0 && deletable(fin) && nucIdx[last] == segs.size - 1) {
            val onset = clusterBefore(segs, nucIdx, last)
            val keep = onset.size >= 2 && (onset.last() == "ɾ" || onset.last() == "j")
            if (!keep) fin.deleted = true
        }
        // 2) medial VC_CV deletion, right to left, never in the first syllable or after ह
        for (i in last - 1 downTo 1) {
            val n = nucs[i]
            if (!deletable(n)) continue
            val before = clusterBefore(segs, nucIdx, i)
            val after = clusterAfter(segs, nucIdx, i)
            val prevNasal = nucs[i - 1].nasal && nucs[i - 1].isSchwa
            val next = nucs[i + 1]
            if (before.size == 1 && !prevNasal && !before[0].endsWith("h") &&
                after.size == 1 && !next.deleted && next.kind != Kind.INHERENT
            ) n.deleted = true
        }
    }

    private fun applyStress(word: String, segs: List<Seg>, nucIdx: List<Int>) {
        val nucs = nucIdx.map { segs[it].nucleus!! }
        if (nucs.isEmpty() || word in UNSTRESSED) return
        val n = nucs.size
        // espeak-ng weight: +1 for a long vowel, +1 if two or more consonants follow
        // (a consonantal anusvara / visarga counts as one). The underlying word-final
        // schwa still counts as the final syllable, which is never a candidate.
        val weights = IntArray(n) { i ->
            val follow = clusterAfter(segs, nucIdx, i).size +
                (if (consonantalNasal(nucs[i])) 1 else 0) + (if (nucs[i].visarga) 1 else 0)
            (if (nucs[i].isLong) 1 else 0) + (if (follow >= 2) 1 else 0)
        }
        val primary = if (n == 1) 0 else {
            val cands = 0 until n - 1
            val maxW = cands.maxOf { weights[it] }
            when {
                weights[n - 1] == 2 && maxW < 2 -> n - 1        // only the final syllable is superheavy
                maxW == 0 -> 0                                  // all light: first syllable
                else -> cands.last { weights[it] == maxW }      // last heaviest non-final
            }
        }
        nucs[primary].stress = 2
        var j = primary + 2
        while (j < n) { nucs[j].stress = 1; j += 2 }
        j = primary - 2
        while (j >= 0) { nucs[j].stress = 1; j -= 2 }
        if (word in SECONDARY_ONLY) nucs.forEach { if (it.stress == 2) it.stress = 1 }
    }

    private fun homorganic(next: String): String = when {
        next in VELAR && next != "j" -> "ŋ"
        next == "j" || next in PALATAL -> "ɲ"
        next in RETROFLEX -> "ɳ"
        next in LABIAL -> "m"
        else -> "n"
    }

    /** Converts one Devanagari word to espeak-style IPA (stress marks included). */
    fun wordToIpa(word: String): String {
        val segs = parse(word)
        val nucIdx = segs.indices.filter { segs[it].nucleus != null }
        if (nucIdx.isEmpty()) return segs.joinToString("") { it.cons ?: "" }
        applySchwaDeletion(segs, nucIdx)
        applyStress(word, segs, nucIdx)

        val sb = StringBuilder()
        val lastLive = nucIdx.lastOrNull { !segs[it].nucleus!!.deleted } ?: -1
        var k = 0
        while (k < segs.size) {
            val s = segs[k]
            val c = s.cons
            if (c != null) {
                val next = segs.getOrNull(k + 1)?.cons
                when {
                    // geminates (same or unaspirated+aspirated pair) inside one cluster
                    next != null && (GEMINATE.containsKey(c to next) || c == next) -> {
                        sb.append(GEMINATE[c to next] ?: if (c in DOUBLED_GEMINATE) c + c else c + "ː")
                        k += 2; continue
                    }
                    next == "j" && c in GEMINATE_BEFORE_Y -> sb.append(c).append("ː")
                    next == "j" && c == "ʈ" -> sb.append("ʈʈ")
                    c == "k" && next == "ʂ" -> { sb.append("kʃ"); k += 2; continue }        // क्ष
                    c == "ɟ" && next == "ɲ" -> { sb.append("ɡj"); k += 2; continue }        // ज्ञ
                    c == "ɾ" && next != null && next in TRILL_CONTEXT -> sb.append("r")
                    else -> sb.append(c)
                }
                k++
                continue
            }
            val n = s.nucleus!!
            if (!n.deleted) {
                val isFinal = k == lastLive && segs.drop(k + 1).none { it.cons != null }
                var v = when {
                    n.kind == Kind.INHERENT -> if (n.stress == 2 && !isFinal) "ʌ" else "ə"
                    n.ipa == "iː" -> if (n.stress >= 1 && !isFinal) "iː" else "i"
                    else -> n.ipa
                }
                val nextCons = segs.drop(k + 1).firstOrNull { it.cons != null || it.nucleus != null }?.cons
                var nasalCoda = ""
                if (n.nasal) {
                    when {
                        n.isSchwa && nextCons != null -> nasalCoda = homorganic(nextCons)
                        n.kind == Kind.INDEPENDENT && v == "aː" && nextCons != null -> nasalCoda = "n"
                        else -> {
                            v = NASALISED[v] ?: v
                            if (n.chandrabindu && n.ipa == "aː" && nextCons != null) nasalCoda = "n"
                        }
                    }
                }
                when (n.stress) {
                    2 -> sb.append(PRIMARY)
                    1 -> sb.append(SECONDARY)
                }
                sb.append(v).append(nasalCoda)
                if (n.visarga) sb.append("h")
            }
            k++
        }
        return sb.toString()
    }

    /** Minimal Latin letter -> IPA fallback (only reached if the codec left Latin text). */
    private val LATIN_FALLBACK: Map<Char, String> = mapOf(
        'a' to "ʌ", 'b' to "b", 'c' to "k", 'd' to "ɖ", 'e' to "eː", 'f' to "f",
        'g' to "ɡ", 'h' to "h", 'i' to "ɪ", 'j' to "ɟ", 'k' to "k", 'l' to "l",
        'm' to "m", 'n' to "n", 'o' to "oː", 'p' to "p", 'q' to "k", 'r' to "ɾ",
        's' to "s", 't' to "ʈ", 'u' to "ʊ", 'v' to "ʋ", 'w' to "ʋ", 'x' to "ks",
        'y' to "j", 'z' to "z",
    )

    private val TOKEN = Regex("""[,.?!;:।॥]|[^\s,.?!;:।॥]+""")
    private val SENTENCE_END = setOf(".", "?", "!", "।", "॥")

    /** One synthesis unit (Piper synthesises each sentence separately). */
    data class Sentence(val text: String, val ipa: String)

    /** Splits text into sentences and phonemises each to an espeak-style IPA string. */
    fun phonemizeSentences(text: String): List<Sentence> {
        val out = ArrayList<Sentence>()
        val ipa = StringBuilder()
        val raw = StringBuilder()
        fun flush() {
            val s = ipa.toString().trim()
            if (s.any { it.isLetter() }) out.add(Sentence(raw.toString().trim(), s))
            ipa.setLength(0); raw.setLength(0)
        }
        for (tok in TOKEN.findAll(text).map { it.value }) {
            raw.append(tok).append(' ')
            when {
                tok in SENTENCE_END -> {
                    if (ipa.isNotEmpty()) {
                        while (ipa.endsWith(" ")) ipa.setLength(ipa.length - 1)
                        ipa.append(if (tok == "?" || tok == "!") tok else ".")
                    }
                    flush()
                }
                tok == "," || tok == ";" || tok == ":" -> {
                    while (ipa.endsWith(" ")) ipa.setLength(ipa.length - 1)
                    if (ipa.isNotEmpty()) ipa.append(tok).append(' ')
                }
                else -> {
                    val w = if (tok.any { isDevanagari(it) }) wordToIpa(tok)
                    else tok.lowercase().mapNotNull { LATIN_FALLBACK[it] }.joinToString("")
                    if (w.isNotEmpty()) {
                        if (ipa.isNotEmpty() && !ipa.endsWith(" ")) ipa.append(' ')
                        ipa.append(w)
                    }
                }
            }
        }
        flush()
        return out
    }

    /** Whole text as one IPA string (sentences joined by a space). */
    fun textToIpa(text: String): String = phonemizeSentences(text).joinToString(" ") { it.ipa }

    /**
     * Flat list of model phonemes for [text]: one entry per NFD code point, i.e. exactly
     * the units Piper looks up in `phoneme_id_map` (spaces and punctuation included).
     */
    fun textToPhonemes(text: String): List<String> =
        phonemizeSentences(text).flatMapIndexed { i, s ->
            val cps = codePoints(s.ipa)
            if (i == 0) cps else listOf(" ") + cps
        }

    private fun codePoints(ipa: String): List<String> {
        val nfd = Normalizer.normalize(ipa, Normalizer.Form.NFD)
        val res = ArrayList<String>(nfd.length)
        var i = 0
        while (i < nfd.length) {
            val cp = nfd.codePointAt(i)
            res.add(String(Character.toChars(cp)))
            i += Character.charCount(cp)
        }
        return res
    }

    /**
     * Piper `phonemes_to_ids` for one sentence:
     *   `^ _` then for every NFD code point `id _`, then `$`.
     * Code points absent from [phonemeIdMap] are skipped (and reported by [coverage]).
     */
    fun ipaToIds(ipa: String, phonemeIdMap: Map<String, IntArray>): IntArray {
        val pad = phonemeIdMap["_"] ?: return IntArray(0)
        val ids = ArrayList<Int>()
        phonemeIdMap["^"]?.let { ids.addAll(it.toList()) }
        ids.addAll(pad.toList())
        for (cp in codePoints(ipa)) {
            val mapped = phonemeIdMap[cp] ?: continue
            ids.addAll(mapped.toList())
            ids.addAll(pad.toList())
        }
        phonemeIdMap["\$"]?.let { ids.addAll(it.toList()) }
        return ids.toIntArray()
    }

    /** Text -> one Piper id sequence per sentence (synthesised separately, Piper-style). */
    fun textToSentenceIds(text: String, phonemeIdMap: Map<String, IntArray>): List<IntArray> =
        phonemizeSentences(text).map { ipaToIds(it.ipa, phonemeIdMap) }.filter { it.size > 3 }

    /**
     * Full text -> a single VITS id sequence (all sentences in one utterance). Kept for
     * callers that synthesise in one pass; [textToSentenceIds] is preferred.
     */
    fun textToPhonemeIds(text: String, phonemeIdMap: Map<String, IntArray>): IntArray {
        val ipa = textToIpa(text)
        return if (ipa.isEmpty()) IntArray(0) else ipaToIds(ipa, phonemeIdMap)
    }

    /** Phoneme coverage of [text] against [phonemeIdMap]: every code point must have an id. */
    data class Coverage(val total: Int, val missing: Map<String, Int>) {
        val dropped: Int get() = missing.values.sum()
        val percent: Double get() = if (total == 0) 100.0 else 100.0 * (total - dropped) / total
    }

    fun coverage(text: String, phonemeIdMap: Map<String, IntArray>): Coverage {
        val missing = sortedMapOf<String, Int>()
        val cps = textToPhonemes(text)
        for (cp in cps) if (!phonemeIdMap.containsKey(cp)) missing[cp] = (missing[cp] ?: 0) + 1
        return Coverage(cps.size, missing)
    }
}
