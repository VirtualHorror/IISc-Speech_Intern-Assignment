package `in`.iisc.vaani.codec

/**
 * VaaniEdge Hinglish Codec.
 *
 * Two deterministic, allocation-light transforms that sit on either side of the
 * on-device neural models. Both run in well under 0.5 ms for typical utterances
 * because they are pure table lookups + a single linear scan (no regex
 * backtracking on the hot path, no model inference).
 *
 *  1. [normalizeSttTranscript]  — STT post-processing.
 *     Whisper, decoding Hindi/Hinglish speech, emits Devanagari. This converts
 *     that Devanagari back into crisp Roman **Hinglish** (for native Hindi words)
 *     and restored **English** spellings (for loanwords like "meeting",
 *     "schedule"), using a large phonetic loanword lexicon plus an algorithmic
 *     Hunterian transliterator for anything unseen, plus a small homophone
 *     corrector for frequent ASR confusions.
 *
 *  2. [toAcousticDevanagariScript] — TTS pre-processing.
 *     The Indic VITS voice pronounces Devanagari. This converts Roman Hinglish,
 *     English words, and numeric/temporal/temperature expressions into natural
 *     phonetic Devanagari (e.g. `5 PM -> पाँच पी एम`, `28°C -> अट्ठाईस डिग्री
 *     सेल्सियस`) so both Hindi and English words are spoken in one Indian voice.
 *
 * Nothing in this file depends on Android or any native library, so it is fully
 * unit-testable on the host JVM.
 */
object HinglishCodec {

    private const val DEV_BLOCK_START = 'ऀ'
    private const val DEV_BLOCK_END = 'ॿ'

    /** Punctuation that is kept verbatim and never prefixed with a space. */
    private val PUNCT = setOf(",", ".", "?", "!", ";", ":", "।", "॥")

    // =========================================================================
    //  Lexicon 1: Devanagari -> Roman (STT normalisation).
    //  Loanwords are restored to their English spelling; native Hindi words map
    //  to conventional Roman Hinglish. Casing is intentional (proper nouns and
    //  the sentence-opening pronouns the IISc prompts use are capitalised).
    // =========================================================================
    val DEV_TO_ROMAN: Map<String, String> = buildMap {
        // --- English loanwords (assistant / scheduling / weather domain) ------
        put("अलार्म", "alarm"); put("मीटिंग", "meeting"); put("मीटींग", "meeting")
        put("शेड्यूल", "schedule"); put("शैड्यूल", "schedule"); put("रिमाइंडर", "reminder")
        put("रिमाइन्डर", "reminder"); put("वेदर", "weather"); put("वैदर", "weather")
        put("फोरकास्ट", "forecast"); put("फॉरकास्ट", "forecast"); put("कैलेंडर", "calendar")
        put("कैलेण्डर", "calendar"); put("नोटिफिकेशन", "notification"); put("अपॉइंटमेंट", "appointment")
        put("टास्क", "task"); put("इवेंट", "event"); put("नोट", "note"); put("लिस्ट", "list")
        put("टाइमर", "timer"); put("क्लॉक", "clock"); put("डेट", "date"); put("टाइम", "time")
        put("मैसेज", "message"); put("कॉल", "call"); put("ईमेल", "email"); put("फोन", "phone")
        put("ऐप", "app"); put("ऐप्लिकेशन", "application"); put("सेटिंग", "setting")
        put("सेटिंग्स", "settings"); put("वॉल्यूम", "volume"); put("ब्राइटनेस", "brightness")
        put("बैटरी", "battery"); put("वाईफाई", "wifi"); put("ब्लूटूथ", "bluetooth")
        put("इंटरनेट", "internet"); put("डाउनलोड", "download"); put("अपलोड", "upload")
        put("सर्च", "search"); put("ओपन", "open"); put("क्लोज", "close"); put("स्टार्ट", "start")
        put("स्टॉप", "stop"); put("प्ले", "play"); put("पॉज", "pause"); put("नेक्स्ट", "next")
        put("प्रीवियस", "previous"); put("रिपीट", "repeat"); put("कैंसिल", "cancel")
        put("कन्फर्म", "confirm"); put("सेव", "save"); put("डिलीट", "delete"); put("एडिट", "edit")
        put("अपडेट", "update"); put("रिफ्रेश", "refresh"); put("लॉगिन", "login")
        put("लॉगआउट", "logout"); put("पासवर्ड", "password"); put("यूजर", "user")
        put("प्रोफाइल", "profile"); put("अकाउंट", "account"); put("होम", "home")
        put("स्क्रीन", "screen"); put("कैमरा", "camera"); put("फोटो", "photo"); put("वीडियो", "video")
        put("म्यूजिक", "music"); put("सॉन्ग", "song"); put("न्यूज", "news"); put("स्पोर्ट्स", "sports")
        put("ट्रैफिक", "traffic"); put("मैप", "map"); put("लोकेशन", "location")
        put("डायरेक्शन", "direction"); put("रूट", "route"); put("वेब", "web"); put("साइट", "site")

        // --- Weather vocabulary ----------------------------------------------
        put("टेम्परेचर", "temperature"); put("तापमान", "temperature"); put("रेन", "rain")
        put("बारिश", "baarish"); put("धूप", "dhoop"); put("क्लाउड", "cloud")
        put("हवा", "hawa"); put("तूफान", "toofan"); put("ह्यूमिडिटी", "humidity")
        put("डिग्री", "degree"); put("सेल्सियस", "celsius"); put("फारेनहाइट", "fahrenheit")

        // --- Days / months / time words --------------------------------------
        put("सोमवार", "Monday"); put("मंगलवार", "Tuesday"); put("बुधवार", "Wednesday")
        put("गुरुवार", "Thursday"); put("शुक्रवार", "Friday"); put("शनिवार", "Saturday")
        put("रविवार", "Sunday"); put("जनवरी", "January"); put("फरवरी", "February")
        put("मार्च", "March"); put("अप्रैल", "April"); put("मई", "May"); put("जून", "June")
        put("जुलाई", "July"); put("अगस्त", "August"); put("सितंबर", "September")
        put("अक्टूबर", "October"); put("नवंबर", "November"); put("दिसंबर", "December")
        put("सुबह", "subah"); put("शाम", "shaam"); put("दोपहर", "dopahar"); put("रात", "raat")
        put("बजे", "baje"); put("मिनट", "minute"); put("घंटा", "ghanta"); put("घंटे", "ghante")
        put("सेकंड", "second"); put("आज", "Aaj"); put("कल", "kal"); put("परसों", "parson")
        put("अभी", "abhi"); put("बाद", "baad"); put("पहले", "pehle"); put("रोज", "roz")

        // --- Places (Indian cities; proper-noun casing) ----------------------
        put("बैंगलोर", "Bangalore"); put("बेंगलुरु", "Bangalore"); put("दिल्ली", "Delhi")
        put("मुंबई", "Mumbai"); put("चेन्नई", "Chennai"); put("कोलकाता", "Kolkata")
        put("हैदराबाद", "Hyderabad"); put("पुणे", "Pune"); put("जयपुर", "Jaipur")
        put("लखनऊ", "Lucknow"); put("भारत", "Bharat"); put("इंडिया", "India")

        // --- Pronouns / question words / function words ----------------------
        put("मैं", "main"); put("मुझे", "Mujhe"); put("मेरा", "mera"); put("मेरी", "meri")
        put("मेरे", "mere"); put("तुम", "tum"); put("तुम्हें", "tumhe"); put("आप", "aap")
        put("आपको", "aapko"); put("हम", "hum"); put("वह", "wah"); put("यह", "yah")
        put("क्या", "kya"); put("कब", "kab"); put("कहाँ", "kahan"); put("कौन", "kaun")
        put("कैसे", "kaise"); put("क्यों", "kyun"); put("कितना", "kitna"); put("कितने", "kitne")
        put("और", "aur"); put("या", "ya"); put("है", "hai"); put("हैं", "hain"); put("था", "tha")
        put("थी", "thi"); put("का", "ka"); put("के", "ke"); put("की", "ki"); put("को", "ko")
        put("में", "mein"); put("से", "se"); put("पर", "par"); put("तक", "tak"); put("भी", "bhi")
        put("नहीं", "nahi"); put("हाँ", "haan"); put("ठीक", "theek"); put("अच्छा", "accha")

        // --- Common verbs (imperative / infinitive) --------------------------
        put("सेट", "set"); put("करना", "karna"); put("करो", "karo"); put("कर", "kar")
        put("बताओ", "batao"); put("बता", "bata"); put("दिखाओ", "dikhao"); put("दिखा", "dikha")
        put("खोलो", "kholo"); put("बंद", "band"); put("चालू", "chalu"); put("लगाओ", "lagao")
        put("भेजो", "bhejo"); put("बनाओ", "banao"); put("याद", "yaad"); put("दिला", "dila")
        put("चलाओ", "chalao"); put("रोको", "roko"); put("सुनाओ", "sunao"); put("पढ़ो", "padho")

        // --- Misc frequent words --------------------------------------------
        put("फॉर", "for"); put("फोर", "for"); put("माय", "my"); put("एट", "at"); put("इन", "in")
        put("ऑन", "on"); put("ऑफ", "off"); put("टू", "to"); put("नाउ", "now"); put("टुडे", "today")
        put("टुमारो", "tomorrow"); put("यस", "yes"); put("नो", "no"); put("ओके", "okay")
        put("प्लीज", "please"); put("थैंक्स", "thanks"); put("हैलो", "hello"); put("हाय", "hi")
    }

    // =========================================================================
    //  Lexicon 2: Roman/English -> Devanagari (TTS acoustic script).
    //  Keys are lower-cased on lookup. Values are phonetic Devanagari tuned for
    //  the Indic VITS voice.
    // =========================================================================
    val ROMAN_TO_DEV: Map<String, String> = buildMap {
        put("weather", "वेदर"); put("forecast", "फोरकास्ट"); put("for", "फॉर")
        put("bangalore", "बैंगलोर"); put("bengaluru", "बैंगलोर"); put("reminder", "रिमाइंडर")
        put("meeting", "मीटिंग"); put("schedule", "शेड्यूल"); put("alarm", "अलार्म")
        put("my", "माय"); put("at", "एट"); put("set", "सेट"); put("next", "नेक्स्ट")
        put("calendar", "कैलेंडर"); put("task", "टास्क"); put("event", "इवेंट"); put("note", "नोट")
        put("timer", "टाइमर"); put("message", "मैसेज"); put("call", "कॉल"); put("email", "ईमेल")
        put("today", "टुडे"); put("tomorrow", "टुमारो"); put("now", "नाउ"); put("please", "प्लीज")
        put("temperature", "टेम्परेचर"); put("degree", "डिग्री"); put("celsius", "सेल्सियस")
        put("fahrenheit", "फ़ारेनहाइट"); put("rain", "रेन"); put("humidity", "ह्यूमिडिटी")
        put("monday", "मंडे"); put("tuesday", "ट्यूज़डे"); put("wednesday", "वेंसडे")
        put("thursday", "थर्सडे"); put("friday", "फ्राइडे"); put("saturday", "सैटरडे")
        put("sunday", "संडे"); put("am", "ए एम"); put("pm", "पी एम"); put("ok", "ओके")
        put("yes", "यस"); put("no", "नो"); put("hello", "हैलो")
        // Hindi words written in Roman -> Devanagari
        put("mujhe", "मुझे"); put("kal", "कल"); put("subah", "सुबह"); put("shaam", "शाम")
        put("baje", "बजे"); put("ka", "का"); put("ke", "के"); put("ki", "की"); put("karna", "करना")
        put("hai", "है"); put("hain", "हैं"); put("aaj", "आज"); put("kya", "क्या"); put("aur", "और")
        put("meri", "मेरी"); put("mera", "मेरा"); put("mere", "मेरे"); put("kab", "कब")
        put("main", "मैं"); put("aap", "आप"); put("nahi", "नहीं"); put("haan", "हाँ")
    }

    // =========================================================================
    //  Numbers -> Hindi words (0–40 exact, tens, 100) with a digit-by-digit
    //  fallback for anything outside the table.
    // =========================================================================
    private val NUM_TO_HINDI: Map<Int, String> = mapOf(
        0 to "शून्य", 1 to "एक", 2 to "दो", 3 to "तीन", 4 to "चार", 5 to "पाँच",
        6 to "छह", 7 to "सात", 8 to "आठ", 9 to "नौ", 10 to "दस", 11 to "ग्यारह",
        12 to "बारह", 13 to "तेरह", 14 to "चौदह", 15 to "पंद्रह", 16 to "सोलह",
        17 to "सत्रह", 18 to "अठारह", 19 to "उन्नीस", 20 to "बीस", 21 to "इक्कीस",
        22 to "बाईस", 23 to "तेईस", 24 to "चौबीस", 25 to "पच्चीस", 26 to "छब्बीस",
        27 to "सत्ताईस", 28 to "अट्ठाईस", 29 to "उनतीस", 30 to "तीस", 31 to "इकतीस",
        32 to "बत्तीस", 33 to "तैंतीस", 34 to "चौंतीस", 35 to "पैंतीस", 36 to "छत्तीस",
        37 to "सैंतीस", 38 to "अड़तीस", 39 to "उनतालीस", 40 to "चालीस", 50 to "पचास",
        60 to "साठ", 70 to "सत्तर", 80 to "अस्सी", 90 to "नब्बे", 100 to "सौ",
    )

    private val DEV_DIGIT = mapOf(
        '०' to '0', '१' to '1', '२' to '2', '३' to '3', '४' to '4',
        '५' to '5', '६' to '6', '७' to '7', '८' to '8', '९' to '9',
    )

    /** Frequent Roman-side ASR homophone corrections (domain-scoped; no-ops on correct text). */
    private val HOMOPHONE_FIX: Map<String, String> = mapOf(
        "wether" to "weather", "wheather" to "weather", "fore cast" to "forecast",
        "bengaluru" to "Bangalore", "banglore" to "Bangalore", "metting" to "meeting",
        "schedual" to "schedule", "remainder" to "reminder", "alaram" to "alarm",
    )

    // -------------------------------------------------------------------------
    //  Tokeniser: splits into word tokens and single punctuation marks, dropping
    //  whitespace. Punctuation is preserved so we can re-attach it without a
    //  leading space on output.
    // -------------------------------------------------------------------------
    private val TOKEN = Regex("""[,.?!;:।॥]|[^\s,.?!;:।॥]+""")

    private fun tokenize(text: String): List<String> =
        TOKEN.findAll(text).map { it.value }.toList()

    private fun join(tokens: List<String>): String {
        val sb = StringBuilder()
        for (t in tokens) {
            if (sb.isEmpty() || t in PUNCT) sb.append(t)
            else sb.append(' ').append(t)
        }
        return sb.toString()
    }

    private fun isDevanagari(s: String): Boolean =
        s.any { it in DEV_BLOCK_START..DEV_BLOCK_END }

    private fun normalizeDigits(s: String): String =
        if (s.any { it in DEV_DIGIT }) s.map { DEV_DIGIT[it] ?: it }.joinToString("") else s

    // =========================================================================
    //  1. STT transcript normalisation: Devanagari -> Roman Hinglish/English.
    // =========================================================================
    @Suppress("UNUSED_PARAMETER")
    fun normalizeSttTranscript(rawText: String, languageHint: String = "hi"): String {
        if (rawText.isBlank()) return ""
        val out = ArrayList<String>()
        for (raw in tokenize(rawText)) {
            if (raw in PUNCT) { out.add(raw); continue }
            val tok = normalizeDigits(raw)
            when {
                !isDevanagari(tok) -> out.add(applyHomophoneFix(tok))            // already Roman / numeric
                DEV_TO_ROMAN.containsKey(tok) -> out.add(DEV_TO_ROMAN.getValue(tok))
                else -> out.add(applyHomophoneFix(hunterianTransliterate(tok)))  // unseen Hindi word
            }
        }
        return join(out)
    }

    private fun applyHomophoneFix(token: String): String =
        HOMOPHONE_FIX[token.lowercase()] ?: token

    /**
     * Algorithmic Devanagari -> Roman (Hunterian-style) fallback for words not in
     * the lexicon. Handles consonant clusters (virama), dependent/independent
     * vowels, and drops the word-final inherent schwa for natural Hinglish.
     */
    fun hunterianTransliterate(word: String): String {
        val sb = StringBuilder()
        var pending: String? = null      // consonant awaiting its vowel
        var inherent = false             // does `pending` still carry inherent 'a'?

        fun flush(withA: Boolean) {
            pending?.let {
                sb.append(it)
                if (withA && inherent) sb.append('a')
            }
            pending = null; inherent = false
        }

        for (c in word) {
            when (c) {
                in H_CONS -> { flush(true); pending = H_CONS.getValue(c); inherent = true }
                in H_MATRA -> { pending?.let { sb.append(it) }; sb.append(H_MATRA.getValue(c)); pending = null; inherent = false }
                '्' -> { pending?.let { sb.append(it) }; pending = null; inherent = false } // virama
                in H_VOWEL -> { flush(true); sb.append(H_VOWEL.getValue(c)) }
                'ं', 'ँ' -> { flush(true); sb.append('n') }                            // anusvara/chandrabindu
                'ः' -> { flush(true); sb.append("h") }                                      // visarga
                else -> { flush(true); if (c.code in 0x20..0x7E) sb.append(c) }
            }
        }
        flush(false)   // word-final schwa deletion
        return sb.toString()
    }

    // =========================================================================
    //  2. TTS acoustic script: Roman/English + numerics -> phonetic Devanagari.
    // =========================================================================
    private val RE_TEMP = Regex("""(\d+)\s*°?\s*([CF])\b""")
    private val RE_TIME = Regex("""(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?""")
    private val RE_AMPM = Regex("""(\d+)\s*(AM|PM|am|pm)""")
    private val RE_NUM = Regex("""\d+""")

    fun toAcousticDevanagariScript(displayText: String): String {
        if (displayText.isBlank()) return ""
        var s = normalizeDigits(displayText)

        s = RE_TEMP.replace(s) { m ->
            val unit = if (m.groupValues[2].uppercase() == "C") "सेल्सियस" else "फ़ारेनहाइट"
            "${hindiNumber(m.groupValues[1])} डिग्री $unit"
        }
        s = RE_TIME.replace(s) { m ->
            val hour = hindiNumber(m.groupValues[1])
            val min = m.groupValues[2]
            val minPart = if (min == "00") "" else " ${hindiNumber(min)}"
            val ap = ampmWords(m.groupValues[3])
            "$hour$minPart$ap"
        }
        s = RE_AMPM.replace(s) { m -> "${hindiNumber(m.groupValues[1])}${ampmWords(m.groupValues[2])}" }
        s = RE_NUM.replace(s) { m -> hindiNumber(m.value) }

        // Word-level Roman/English -> Devanagari; Devanagari & punctuation pass through.
        val out = ArrayList<String>()
        for (tok in tokenize(s)) {
            when {
                tok in PUNCT -> out.add(tok)
                isDevanagari(tok) -> out.add(tok)
                else -> out.add(ROMAN_TO_DEV[tok.lowercase()] ?: romanToDevFallback(tok))
            }
        }
        return join(out)
    }

    private fun ampmWords(raw: String): String = when (raw.uppercase()) {
        "AM" -> " ए एम"
        "PM" -> " पी एम"
        else -> ""
    }

    private fun hindiNumber(numStr: String): String {
        val n = numStr.toIntOrNull() ?: return numStr
        NUM_TO_HINDI[n]?.let { return it }
        // Digit-by-digit fallback for out-of-table numbers.
        return numStr.map { NUM_TO_HINDI[it.digitToIntOrNull() ?: -1] ?: it.toString() }
            .joinToString(" ")
    }

    /** Lightweight Roman -> Devanagari phonetic fallback for unseen Latin words. */
    private fun romanToDevFallback(word: String): String {
        if (word.isEmpty()) return word
        val lower = word.lowercase()
        val sb = StringBuilder()
        var i = 0
        while (i < lower.length) {
            val two = if (i + 1 < lower.length) lower.substring(i, i + 2) else ""
            val di = R2D_DIGRAPH[two]
            if (di != null) { sb.append(di); i += 2; continue }
            sb.append(R2D_CHAR[lower[i]] ?: lower[i].toString())
            i++
        }
        return sb.toString()
    }

    // --- Transliteration tables ---------------------------------------------
    private val H_CONS: Map<Char, String> = mapOf(
        'क' to "k", 'ख' to "kh", 'ग' to "g", 'घ' to "gh", 'ङ' to "ng",
        'च' to "ch", 'छ' to "chh", 'ज' to "j", 'झ' to "jh", 'ञ' to "ny",
        'ट' to "t", 'ठ' to "th", 'ड' to "d", 'ढ' to "dh", 'ण' to "n",
        'त' to "t", 'थ' to "th", 'द' to "d", 'ध' to "dh", 'न' to "n",
        'प' to "p", 'फ' to "ph", 'ब' to "b", 'भ' to "bh", 'म' to "m",
        'य' to "y", 'र' to "r", 'ल' to "l", 'व' to "v", 'श' to "sh",
        'ष' to "sh", 'स' to "s", 'ह' to "h", 'ळ' to "l", '\u0958' to "q",
        '\u0959' to "kh", '\u095a' to "gh", '\u095b' to "z", '\u095c' to "r", '\u095d' to "rh", '\u095f' to "f",
    )
    private val H_MATRA: Map<Char, String> = mapOf(
        'ा' to "aa", 'ि' to "i", 'ी' to "ee", 'ु' to "u",
        'ू' to "oo", 'ृ' to "ri", 'े' to "e", 'ै' to "ai",
        'ो' to "o", 'ौ' to "au",
    )
    private val H_VOWEL: Map<Char, String> = mapOf(
        'अ' to "a", 'आ' to "aa", 'इ' to "i", 'ई' to "ee", 'उ' to "u", 'ऊ' to "oo",
        'ऋ' to "ri", 'ए' to "e", 'ऐ' to "ai", 'ओ' to "o", 'औ' to "au", 'ऑ' to "o",
    )
    private val R2D_DIGRAPH: Map<String, String> = mapOf(
        "ch" to "च", "sh" to "श", "th" to "थ", "ph" to "फ", "kh" to "ख", "gh" to "घ",
        "oo" to "ू", "ee" to "ी", "aa" to "ा",
    )
    private val R2D_CHAR: Map<Char, String> = mapOf(
        'a' to "अ", 'b' to "ब", 'c' to "क", 'd' to "ड", 'e' to "े", 'f' to "फ़",
        'g' to "ग", 'h' to "ह", 'i' to "ि", 'j' to "ज", 'k' to "क", 'l' to "ल",
        'm' to "म", 'n' to "न", 'o' to "ो", 'p' to "प", 'q' to "क", 'r' to "र",
        's' to "स", 't' to "ट", 'u' to "ु", 'v' to "व", 'w' to "व", 'x' to "क्स",
        'y' to "य", 'z' to "ज़",
    )
}
