package `in`.iisc.vaani.lab

import `in`.iisc.vaani.codec.EnglishLexicon
import `in`.iisc.vaani.codec.HinglishCodec
import `in`.iisc.vaani.codec.IndicPhonemizer
import `in`.iisc.vaani.neural.VaaniTtsConfig
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Bridge between the shipped Kotlin front-end and the host-side acoustic lab.
 *
 * Runs the *exact* on-device text front-end (HinglishCodec -> IndicPhonemizer) over
 * `speech_lab/closed_loop_corpus.json`, checks every emitted IPA code point against the
 * real `phoneme_id_map` of `vaani_tts_config.json`, and writes
 * `app/build/vaani/closed_loop_fixture.json`, which `speech_lab/verify_closed_loop.py`
 * feeds to the VITS -> Whisper round trip (and scores against espeak-ng). The Python lab
 * therefore never re-implements the G2P: the closed loop always measures the code that
 * ships in the APK.
 */
class ClosedLoopFixtureTest {

    private val moduleDir = File(".").absoluteFile
    private val corpusFile = File(moduleDir, "../speech_lab/closed_loop_corpus.json")
    private val ttsConfigFile = File(moduleDir, "src/main/assets/models/vaani_tts_config.json")
    private val outFile = File(moduleDir, "build/vaani/closed_loop_fixture.json")
    private val lexiconFile = File(moduleDir, "src/main/assets/lexicon/vaani_en_cmudict.txt")

    /** Extra Devanagari probe words covering every rule family of the G2P. */
    private val probeWords = listOf(
        "घर", "झंडा", "ढोल", "धन", "भारत", "पढ़ना", "पढ़ाई", "कपड़ा", "लड़का", "सड़क", "बड़ा",
        "खाना", "छत", "ठंडा", "थाली", "फल", "कमल", "नमस्ते", "हाँ", "में", "मैं", "हैं", "नहीं", "हूँ",
        "क्यों", "माँ", "पाँच", "आँख", "गाँव", "आँसू", "ऋषि", "कृपया", "क्षमा", "ज्ञान", "श्री",
        "ज़रूर", "फ़ोन", "क़ायदा", "ख़ुश", "ग़लत", "पानी", "नीला", "दीदी", "ईद", "इमली", "ठीक", "सीधा",
        "करना", "समझना", "बदलना", "कमला", "अपना", "सबसे", "हमको", "बोलना", "नमकीन", "सरकार",
        "अमरूद", "कमरा", "बहन", "पहला", "कहना", "रहना", "महल", "मित्र", "पत्र", "प्यार", "स्कूल",
        "स्टेशन", "ट्रेन", "प्रकाश", "विद्या", "शब्द", "दर्द", "धर्म", "अर्थ", "सत्य", "उत्तर",
        "दक्षिण", "अंदर", "संगीत", "पंप", "गंदा", "बंद", "ठंड", "संसार", "हंस", "अंश", "संयोग",
        "संवाद", "सिंह", "हँसना", "लंबा", "चंपा", "पंजाब", "घंटा", "मंच", "अच्छा", "पत्थर", "बुद्धि",
        "मक्खन", "चिट्ठी", "अड्डा", "गद्दा", "ग्यारह", "ध्यान", "ज्यादा", "न्याय", "व्यस्त",
        "स्याही", "म्यूजिक", "ब्याज", "ट्यूशन", "प्यास", "गर्म", "शर्ट", "कार्ड", "मार्च", "पार्क",
        "बर्फ़", "दर्जा", "सर्दी", "पर्दा", "अर्जुन", "दुःख", "ऐसा", "ओर", "औरत", "ऋतु", "आयोग",
        "सवाल", "नया", "गया", "किया", "हुआ", "दिया", "मदद", "सकते", "लिए", "मैंने",
    )

    @Test fun exportFixtureAndAssertFullPhonemeCoverage() {
        assertTrue("missing corpus ${corpusFile.path}", corpusFile.exists())
        HinglishCodec.englishLexicon = EnglishLexicon.parse(lexiconFile.readText())
        val config = VaaniTtsConfig.fromJson(ttsConfigFile.readText())
        val idMap = config.phonemeIdMap
        assertTrue("phoneme_id_map failed to parse", idMap.size > 100)

        val utterances = JSONObject(corpusFile.readText()).getJSONArray("utterances")
        val out = JSONArray()
        var total = 0
        val missing = sortedMapOf<String, Int>()
        val words = LinkedHashSet<String>()

        for (i in 0 until utterances.length()) {
            val u = utterances.getJSONObject(i)
            val text = u.getString("text")
            val lang = u.getString("lang")
            val acoustic = HinglishCodec.toAcousticDevanagariScript(text, lang)
            val sentences = IndicPhonemizer.phonemizeSentences(acoustic)
            val cov = IndicPhonemizer.coverage(acoustic, idMap)
            total += cov.total
            cov.missing.forEach { (k, v) -> missing[k] = (missing[k] ?: 0) + v }
            acoustic.split(Regex("[\\s,.?!;:।॥]+")).filter { w -> w.any { it in 'ऀ'..'ॿ' } }.forEach { words.add(it) }

            out.put(JSONObject().apply {
                put("id", u.getString("id"))
                put("kind", u.optString("kind"))
                put("lang", lang)
                put("text", text)
                put("acoustic_devanagari", acoustic)
                put("ipa_sentences", JSONArray(sentences.map { it.ipa }))
                put("phonemes", JSONArray(IndicPhonemizer.textToPhonemes(acoustic)))
                put("sentence_ids", JSONArray(sentences.map { JSONArray(IndicPhonemizer.ipaToIds(it.ipa, idMap).toList()) }))
                put("codepoints_total", cov.total)
                put("codepoints_missing", JSONObject(cov.missing as Map<*, *>))
            })
        }

        // Word-level probe set for espeak-ng parity: lexicon + corpus + rule probes.
        HinglishCodec.DEV_TO_ROMAN.keys.forEach { words.add(it) }
        HinglishCodec.ROMAN_TO_DEV.values.forEach { v -> v.split(' ').filter { it.isNotBlank() }.forEach { words.add(it) } }
        words.addAll(probeWords)
        val probe = JSONArray()
        for (w in words) {
            val ipa = IndicPhonemizer.wordToIpa(w)
            val cov = IndicPhonemizer.coverage(w, idMap)
            total += cov.total
            cov.missing.forEach { (k, v) -> missing[k] = (missing[k] ?: 0) + v }
            probe.put(JSONObject().put("word", w).put("ipa", ipa))
        }

        val dropped = missing.values.sum()
        val coverage = if (total == 0) 0.0 else 100.0 * (total - dropped) / total
        outFile.parentFile?.mkdirs()
        outFile.writeText(JSONObject().apply {
            put("generator", "ClosedLoopFixtureTest (Kotlin HinglishCodec + IndicPhonemizer)")
            put("phoneme_codepoints_total", total)
            put("phoneme_codepoints_dropped", dropped)
            put("phoneme_coverage_pct", coverage)
            put("missing_symbols", JSONObject(missing as Map<*, *>))
            put("utterances", out)
            put("word_probe", probe)
        }.toString(2))

        assertTrue(
            "phoneme coverage ${"%.2f".format(coverage)}% < 100%: dropped $missing",
            dropped == 0,
        )
    }
}
