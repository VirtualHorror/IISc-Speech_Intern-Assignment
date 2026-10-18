package `in`.iisc.vaani.codec

/**
 * On-device English pronunciation lexicon: the first pronunciation of ~125k words from the
 * CMU Pronouncing Dictionary (`assets/lexicon/vaani_en_cmudict.txt`, BSD-style licence in
 * its header), produced by `speech_lab/prepare_models.py`.
 *
 * Kept as one String plus sorted line offsets, so a lookup is an O(log n) binary search
 * with no per-entry objects (~3.3 MB resident instead of a ~20 MB HashMap).
 */
class EnglishLexicon private constructor(
    private val data: String,
    private val starts: IntArray,
) {
    val size: Int get() = starts.size

    /** ARPAbet phones for [word] (stress digits kept, e.g. `W EH1 DH ER0`), or null. */
    fun phones(word: String): List<String>? {
        val key = word.lowercase()
        if (key.isEmpty() || key.length > 20) return null
        var lo = 0
        var hi = starts.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            val c = compareAt(starts[mid], key)
            when {
                c < 0 -> lo = mid + 1
                c > 0 -> hi = mid - 1
                else -> {
                    val from = starts[mid] + key.length + 1
                    val end = data.indexOf('\n', from).let { if (it < 0) data.length else it }
                    return data.substring(from, end).split(' ')
                }
            }
        }
        return null
    }

    fun contains(word: String): Boolean = phones(word) != null

    /** Compares the entry word starting at [start] (terminated by ' ') with [key]. */
    private fun compareAt(start: Int, key: String): Int {
        var i = 0
        while (true) {
            val endD = start + i >= data.length || data[start + i] == ' '
            val endK = i == key.length
            if (endD && endK) return 0
            if (endD) return -1
            if (endK) return 1
            val d = data[start + i] - key[i]
            if (d != 0) return d
            i++
        }
    }

    companion object {
        /** Parses the sorted `word PH PH ...` text format; `;;;` lines are comments. */
        fun parse(text: String): EnglishLexicon {
            var starts = IntArray(1 shl 17)
            var n = 0
            var i = 0
            while (i < text.length) {
                val end = text.indexOf('\n', i).let { if (it < 0) text.length else it }
                if (end > i && !text.startsWith(";;;", i)) {
                    if (n == starts.size) starts = starts.copyOf(n * 2)
                    starts[n++] = i
                }
                i = end + 1
            }
            return EnglishLexicon(text, starts.copyOf(n))
        }
    }
}

/**
 * ARPAbet -> phonetic Devanagari, tuned for an Indian-English rendering that the Hindi
 * VITS voice pronounces naturally (T/D -> retroflex ट/ड, TH/DH -> थ/द, final schwa -> ा,
 * nasal + stop -> anusvara). The Devanagari is then phonemised by [IndicPhonemizer].
 */
internal object ArpabetDevanagari {
    private val CONS: Map<String, String> = mapOf(
        "B" to "ब", "CH" to "च", "D" to "ड", "DH" to "द", "F" to "फ़", "G" to "ग", "HH" to "ह",
        "JH" to "ज", "K" to "क", "L" to "ल", "M" to "म", "N" to "न", "NG" to "ङ", "P" to "प",
        "R" to "र", "S" to "स", "SH" to "श", "T" to "ट", "TH" to "थ", "V" to "व", "W" to "व",
        "Y" to "य", "Z" to "ज़", "ZH" to "ज़",
    )

    /** vowel -> (matra after a consonant, independent letter). */
    private val VOW: Map<String, Pair<String, String>> = mapOf(
        "AA" to ("ा" to "आ"), "AE" to ("ै" to "ऐ"), "AH" to ("" to "अ"), "AO" to ("ॉ" to "ऑ"),
        "AW" to ("ाउ" to "आउ"), "AY" to ("ाइ" to "आइ"), "EH" to ("े" to "ए"), "EY" to ("े" to "ए"),
        "IH" to ("ि" to "इ"), "IY" to ("ी" to "ई"), "OW" to ("ो" to "ओ"), "OY" to ("ॉय" to "ऑय"),
        "UH" to ("ु" to "उ"), "UW" to ("ू" to "ऊ"),
    )
    private val STOPS = setOf("P", "B", "T", "D", "K", "G", "CH", "JH", "TH", "DH")

    fun convert(phones: List<String>, spelling: String): String {
        val ph = phones.map { it.trimEnd { c -> c.isDigit() } }.filter { it.isNotEmpty() }
        val lower = spelling.lowercase()
        val roundAA = 'o' in lower && 'a' !in lower        // "hot", "clock" -> ॉ ; "father" -> ा
        val sb = StringBuilder()
        var openConsonant = false                          // last letter is a consonant without vowel
        for ((i, p) in ph.withIndex()) {
            val next = ph.getOrNull(i + 1)
            val nextIsCons = next != null && next !in VOW && next != "ER"
            when {
                p == "ER" -> {
                    if (!openConsonant) sb.append("अ")
                    sb.append("र")
                    openConsonant = true
                }
                p in VOW -> {
                    val key = if (p == "AA" && roundAA) "AO" else p
                    var (matra, indep) = VOW.getValue(key)
                    if (key == "AH" && next == null) { matra = "ा"; indep = "आ" }   // final schwa
                    sb.append(if (openConsonant) matra else indep)
                    openConsonant = false
                }
                (p == "N" || p == "M" && next in setOf("P", "B")) && !openConsonant && sb.isNotEmpty() &&
                    nextIsCons && next in STOPS -> sb.append("ं")            // center -> सेंटर
                p == "NG" -> {
                    if (openConsonant) sb.append("्")
                    sb.append(if (sb.isEmpty() || openConsonant) "ङ" else "ं")
                    if (next != "K" && next != "G") { sb.append("ग"); openConsonant = true }
                    else openConsonant = false
                }
                else -> {
                    if (openConsonant) sb.append("्")
                    sb.append(CONS[p] ?: "")
                    openConsonant = true
                }
            }
        }
        return sb.toString()
    }
}
