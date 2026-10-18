package `in`.iisc.vaani.neural

import org.json.JSONObject

/**
 * Whisper vocabulary + script-calibration table, parsed from
 * `assets/models/vaani_vocab.json` (produced by `speech_lab/prepare_models.py`).
 *
 * Holds everything the on-device greedy decoder needs:
 *   - the id -> byte-level-BPE token strings (for detokenisation),
 *   - the GPT-2 byte<->unicode decoder (rebuilt here, not stored),
 *   - the special-token ids (SOT / EOT / transcribe / notimestamps / ...),
 *   - the per-language SOT ids, and
 *   - the suppressed-token-id set used for **logit-level script masking** so the
 *     decoder can never drift into Perso-Arabic/Urdu, Cyrillic or CJK output.
 */
class VaaniVocab private constructor(
    val vocabSize: Int,
    private val tokens: Array<String>,
    val sot: Int,
    val eot: Int,
    val transcribe: Int,
    val translate: Int,
    val noTimestamps: Int,
    val timestampBegin: Int,
    val languageTokens: Map<String, Int>,
    private val suppressed: BooleanArray,
) {
    /** True if token [id] must be masked out (suppressed script or timestamp). */
    fun isSuppressed(id: Int): Boolean = id in suppressed.indices && suppressed[id]

    fun languageToken(language: String): Int =
        languageTokens[language.lowercase()] ?: languageTokens["hi"] ?: sot + 1

    /**
     * Detokenises text-range ids (< [eot]) from byte-level BPE back to UTF-8.
     * Special tokens are skipped.
     */
    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id < 0 || id >= eot) continue          // skip specials (eot == first special id)
            if (id < tokens.size) sb.append(tokens[id])
        }
        val bytes = ArrayList<Byte>(sb.length)
        for (ch in sb) BYTE_DECODER[ch]?.let { bytes.add(it.toByte()) }
        return String(bytes.toByteArray(), Charsets.UTF_8)
    }

    companion object {
        /** GPT-2 / Whisper byte<->unicode tables, rebuilt deterministically. */
        private val BYTE_DECODER: Map<Char, Int> = buildByteDecoder()

        private fun buildByteDecoder(): Map<Char, Int> {
            val bs = ArrayList<Int>()
            ('!'.code..'~'.code).forEach { bs.add(it) }
            ('¡'.code..'¬'.code).forEach { bs.add(it) }
            ('®'.code..'ÿ'.code).forEach { bs.add(it) }
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0..255) {
                if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
            }
            // char (from cs codepoint) -> original byte value
            return bs.indices.associate { cs[it].toChar() to bs[it] }
        }

        fun fromJson(json: String): VaaniVocab {
            val root = JSONObject(json)
            val size = root.optInt("vocab_size", 51865)

            val tokArr = root.optJSONArray("tokens")
            val tokens = Array(tokArr?.length() ?: 0) { tokArr!!.optString(it, "") }

            val st = root.optJSONObject("special_tokens") ?: JSONObject()
            val eot = st.optInt("endoftext", 50257)
            val sot = st.optInt("startoftranscript", 50258)
            val transcribe = st.optInt("transcribe", 50359)
            val translate = st.optInt("translate", 50358)
            val noTs = st.optInt("notimestamps", 50363)
            val tsBegin = st.optInt("timestamp_begin", 50364)

            val langs = HashMap<String, Int>()
            root.optJSONObject("language_tokens")?.let { lt ->
                for (k in lt.keys()) langs[k] = lt.getInt(k)
            }

            val suppressed = BooleanArray(size)
            root.optJSONArray("suppressed_token_ids")?.let { arr ->
                for (i in 0 until arr.length()) {
                    val id = arr.getInt(i)
                    if (id in 0 until size) suppressed[id] = true
                }
            }

            return VaaniVocab(size, tokens, sot, eot, transcribe, translate, noTs, tsBegin, langs, suppressed)
        }
    }
}
