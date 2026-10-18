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
 *   - the per-language SOT ids (step-0 language identification reads `en`/`hi`),
 *   - the **logit mask** for text steps: suppressed-script tokens (Perso-Arabic/Urdu,
 *     Cyrillic, CJK, ...), Whisper's non-speech symbols (♪, [, ( ...), and every
 *     special/timestamp token except EOT; plus `suppress_blank` for the first step.
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
    val blank: Int,
    val languageTokens: Map<String, Int>,
    private val scriptSuppressed: BooleanArray,
    private val nonSpeech: BooleanArray,
) {
    /** True if token [id] is a suppressed-script token (Session-1 calibration). */
    fun isSuppressed(id: Int): Boolean = id in scriptSuppressed.indices && scriptSuppressed[id]

    /** Mask for every text step: true = logit forced to -inf. */
    val textMask: BooleanArray = BooleanArray(vocabSize) { id ->
        (id in scriptSuppressed.indices && scriptSuppressed[id]) ||
            (id in nonSpeech.indices && nonSpeech[id]) || id > eot
    }

    /** First text step additionally suppresses EOT and the bare space (Whisper suppress_blank). */
    val firstStepMask: BooleanArray = textMask.copyOf().also {
        if (eot in it.indices) it[eot] = true
        if (blank in it.indices) it[blank] = true
    }

    fun languageToken(language: String): Int =
        languageTokens[language.lowercase()] ?: languageTokens["hi"] ?: sot + 1

    /**
     * Detokenises text-range ids (< [eot]) from byte-level BPE back to UTF-8.
     * Special tokens are skipped; a dangling partial UTF-8 sequence is dropped.
     */
    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id < 0 || id >= eot) continue          // skip specials (eot == first special id)
            if (id < tokens.size) sb.append(tokens[id])
        }
        val bytes = ArrayList<Byte>(sb.length)
        for (ch in sb) BYTE_DECODER[ch]?.let { bytes.add(it.toByte()) }
        return String(bytes.toByteArray(), Charsets.UTF_8).replace("�", "").trim()
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

            fun idSet(key: String): BooleanArray {
                val mask = BooleanArray(size)
                root.optJSONArray(key)?.let { arr ->
                    for (i in 0 until arr.length()) {
                        val id = arr.getInt(i)
                        if (id in 0 until size) mask[id] = true
                    }
                }
                return mask
            }

            return VaaniVocab(
                size, tokens, sot, eot, transcribe, translate, noTs, tsBegin,
                root.optInt("blank_token_id", 220), langs,
                idSet("suppressed_token_ids"), idSet("non_speech_token_ids"),
            )
        }
    }
}
