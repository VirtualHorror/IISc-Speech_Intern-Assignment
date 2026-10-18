package `in`.iisc.vaani.neural

import org.json.JSONObject

/**
 * Parsed `vaani_tts_config.json` for the Indic VITS voice: the eSpeak-IPA
 * `phoneme_id_map` and the output audio sample rate.
 */
class VaaniTtsConfig private constructor(
    val phonemeIdMap: Map<String, IntArray>,
    val sampleRate: Int,
) {
    companion object {
        fun fromJson(json: String): VaaniTtsConfig {
            val root = JSONObject(json)
            val sampleRate = root.optJSONObject("audio")?.optInt("sample_rate", 22_050) ?: 22_050

            val map = HashMap<String, IntArray>()
            root.optJSONObject("phoneme_id_map")?.let { pim ->
                for (key in pim.keys()) {
                    val arr = pim.getJSONArray(key)
                    map[key] = IntArray(arr.length()) { arr.getInt(it) }
                }
            }
            return VaaniTtsConfig(map, sampleRate)
        }
    }
}
