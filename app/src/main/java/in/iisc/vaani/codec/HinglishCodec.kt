package `in`.iisc.vaani.codec

/**
 * VaaniEdge Hinglish Codec.
 *
 * Two deterministic, allocation-light transforms that sit on either side of the
 * on-device neural models. Both run in well under a millisecond for typical utterances
 * because they are table lookups + linear scans (no model inference).
 *
 *  1. [normalizeSttTranscript]  — STT post-processing.
 *     Whisper, decoding Hindi/Hinglish speech, emits Devanagari. This converts
 *     that Devanagari back into crisp Roman **Hinglish** (for native Hindi words)
 *     and restored **English** spellings (for loanwords like "meeting",
 *     "schedule"), using a phonetic loanword lexicon plus an algorithmic
 *     Hunterian transliterator for anything unseen, plus a small homophone
 *     corrector for frequent ASR confusions.
 *
 *  2. [toAcousticDevanagariScript] — TTS pre-processing.
 *     The Indic VITS voice pronounces Devanagari. This converts Roman Hinglish,
 *     English words, and numeric/temporal/temperature expressions into natural
 *     phonetic Devanagari, language-aware:
 *       Hinglish  `4:30 PM, 28°C`  ->  `साढ़े चार पी एम, अट्ठाईस डिग्री सेल्सियस`
 *       English   `4:30 PM, 28°C`  ->  `फ़ोर थर्टी पी एम, ट्वेंटी एट डिग्रीज़ सेल्सियस`
 *     Each Latin word is routed: curated Indian-English lexicon -> Hinglish lexicon ->
 *     CMU English lexicon ([englishLexicon], ARPAbet -> Devanagari) -> rule-based
 *     Roman-Hindi transliteration; ALL-CAPS tokens are spelled as letters (DSP -> डी एस पी).
 *     Both Hindi and English words are then spoken by the one Indian voice.
 *
 * Nothing in this file depends on Android or any native library, so it is fully
 * unit-testable on the host JVM. The English lexicon is injected at runtime.
 */
object HinglishCodec {

    private const val DEV_BLOCK_START = 'ऀ'
    private const val DEV_BLOCK_END = 'ॿ'

    /** Punctuation that is kept verbatim and never prefixed with a space. */
    private val PUNCT = setOf(",", ".", "?", "!", ";", ":", "।", "॥")

    /**
     * CMU English pronunciation lexicon, installed at startup from
     * `assets/lexicon/vaani_en_cmudict.txt` (or by tests from the same file).
     */
    @Volatile var englishLexicon: EnglishLexicon? = null

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
        put("मीटिंग्स", "meetings"); put("रिव्यू", "review"); put("लैब", "lab"); put("सेट", "set")

        // --- Weather vocabulary ----------------------------------------------
        put("टेम्परेचर", "temperature"); put("तापमान", "temperature"); put("रेन", "rain")
        put("बारिश", "baarish"); put("धूप", "dhoop"); put("क्लाउड", "cloud")
        put("हवा", "hawa"); put("तूफान", "toofan"); put("ह्यूमिडिटी", "humidity")
        put("डिग्री", "degree"); put("सेल्सियस", "celsius"); put("फारेनहाइट", "fahrenheit")
        put("मौसम", "mausam"); put("बादल", "baadal")

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
        put("करना", "karna"); put("करो", "karo"); put("कर", "kar")
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
    //  Lexicon 2a: curated Indian-English pronunciations (Roman -> Devanagari).
    //  Takes priority over CMUdict for words whose Indian pronunciation differs
    //  from the American dictionary entry, and for the assistant's core vocabulary.
    // =========================================================================
    val ROMAN_TO_DEV: Map<String, String> = buildMap {
        put("weather", "वेदर"); put("forecast", "फोरकास्ट"); put("for", "फॉर")
        put("bangalore", "बैंगलोर"); put("bengaluru", "बेंगलुरु"); put("reminder", "रिमाइंडर")
        put("reminders", "रिमाइंडर्स"); put("meeting", "मीटिंग"); put("meetings", "मीटिंग्स")
        put("schedule", "शेड्यूल"); put("scheduled", "शेड्यूल्ड"); put("alarm", "अलार्म"); put("alarms", "अलार्म्स")
        put("my", "माय"); put("at", "एट"); put("set", "सेट"); put("next", "नेक्स्ट")
        put("calendar", "कैलेंडर"); put("task", "टास्क"); put("event", "इवेंट"); put("note", "नोट")
        put("timer", "टाइमर"); put("message", "मैसेज"); put("call", "कॉल"); put("email", "ईमेल")
        put("today", "टुडे"); put("tomorrow", "टुमारो"); put("now", "नाउ"); put("please", "प्लीज")
        put("temperature", "टेम्परेचर"); put("degree", "डिग्री"); put("degrees", "डिग्रीज़")
        put("celsius", "सेल्सियस"); put("fahrenheit", "फ़ारेनहाइट"); put("rain", "रेन")
        put("humidity", "ह्यूमिडिटी"); put("monday", "मंडे"); put("tuesday", "ट्यूज़डे")
        put("wednesday", "वेंसडे"); put("thursday", "थर्सडे"); put("friday", "फ्राइडे")
        put("saturday", "सैटरडे"); put("sunday", "संडे"); put("ok", "ओके"); put("okay", "ओके")
        put("yes", "यस"); put("no", "नो"); put("hello", "हैलो")
        put("the", "द"); put("a", "अ"); put("an", "ऐन"); put("i", "आई"); put("is", "इज़")
        put("are", "आर"); put("and", "ऐंड"); put("of", "ऑफ़"); put("to", "टू"); put("in", "इन")
        put("on", "ऑन"); put("with", "विद"); put("your", "योर"); put("you", "यू"); put("have", "हैव")
        put("has", "हैज़"); put("will", "विल"); put("be", "बी"); put("it", "इट"); put("this", "दिस")
        put("that", "दैट"); put("what", "व्हाट"); put("when", "वेन"); put("where", "वेयर")
        put("how", "हाउ"); put("can", "कैन"); put("high", "हाई"); put("low", "लो")
        put("morning", "मॉर्निंग"); put("evening", "ईवनिंग"); put("afternoon", "आफ़्टरनून")
        put("night", "नाइट"); put("done", "डन"); put("sure", "श्योर"); put("lab", "लैब")
        put("review", "रिव्यू"); put("sync", "सिंक"); put("speech", "स्पीच"); put("edge", "एज")
        put("benchmark", "बेंचमार्क"); put("telemetry", "टेलीमेट्री"); put("quantization", "क्वांटाइज़ेशन")
        put("computational", "कम्प्यूटेशनल"); put("intelligence", "इंटेलिजेंस")
        put("expect", "एक्सपेक्ट"); put("thunderstorm", "थंडरस्टॉर्म"); put("thunderstorms", "थंडरस्टॉर्म्स")
        put("cloudy", "क्लाउडी"); put("sunny", "सनी"); put("chance", "चांस"); put("showers", "शावर्स")
        put("drizzle", "ड्रिज़ल"); put("partly", "पार्टली"); put("clear", "क्लियर"); put("umbrella", "अम्ब्रेला")
        put("gemini", "जेमिनी"); put("llama", "लामा"); put("groq", "ग्रॉक"); put("minutes", "मिनट्स")
        put("minute", "मिनट"); put("hour", "आवर"); put("hours", "आवर्स"); put("percent", "परसेंट")
        put("point", "पॉइंट"); put("zero", "ज़ीरो"); put("one", "वन"); put("two", "टू"); put("three", "थ्री")
        put("four", "फ़ोर"); put("five", "फ़ाइव"); put("six", "सिक्स"); put("seven", "सेवन")
        put("eight", "एट"); put("nine", "नाइन"); put("ten", "टेन"); put("eleven", "इलेवन")
        put("twelve", "ट्वेल्व"); put("twenty", "ट्वेंटी"); put("thirty", "थर्टी"); put("forty", "फ़ोर्टी")
        put("fifty", "फ़िफ़्टी"); put("hundred", "हंड्रेड"); put("thousand", "थाउज़ंड")
        put("int", "इंट"); put("iisc", "आई आई एस सी")
    }

    // =========================================================================
    //  Lexicon 2b: Hindi words written in Roman (Hinglish) -> Devanagari.
    // =========================================================================
    val HINGLISH_TO_DEV: Map<String, String> = buildMap {
        put("mujhe", "मुझे"); put("kal", "कल"); put("subah", "सुबह"); put("shaam", "शाम")
        put("raat", "रात"); put("dopahar", "दोपहर"); put("din", "दिन"); put("baje", "बजे")
        put("ka", "का"); put("ke", "के"); put("ki", "की"); put("ko", "को"); put("se", "से")
        put("karna", "करना"); put("karo", "करो"); put("kar", "कर"); put("karenge", "करेंगे")
        put("hai", "है"); put("hain", "हैं"); put("ho", "हो"); put("hoga", "होगा"); put("hogi", "होगी")
        put("honge", "होंगे"); put("hua", "हुआ"); put("hui", "हुई"); put("hue", "हुए"); put("tha", "था")
        put("thi", "थी"); put("the", "थे"); put("aaj", "आज"); put("kya", "क्या"); put("aur", "और")
        put("meri", "मेरी"); put("mera", "मेरा"); put("mere", "मेरे"); put("kab", "कब")
        put("main", "मैं"); put("mai", "मैं"); put("aap", "आप"); put("aapka", "आपका"); put("aapki", "आपकी")
        put("aapke", "आपके"); put("aapko", "आपको"); put("nahi", "नहीं"); put("nahin", "नहीं")
        put("haan", "हाँ"); put("mein", "में"); put("me", "में"); put("par", "पर"); put("pe", "पे")
        put("ya", "या"); put("bhi", "भी"); put("hi", "ही"); put("to", "तो"); put("toh", "तो")
        put("na", "ना"); put("kahan", "कहाँ"); put("kaise", "कैसे"); put("kyun", "क्यों"); put("kyon", "क्यों")
        put("kaun", "कौन"); put("kitna", "कितना"); put("kitne", "कितने"); put("kitni", "कितनी")
        put("ye", "ये"); put("yeh", "यह"); put("wo", "वो"); put("woh", "वह"); put("vo", "वो")
        put("is", "इस"); put("us", "उस"); put("iska", "इसका"); put("uska", "उसका"); put("hum", "हम")
        put("hamara", "हमारा"); put("humara", "हमारा"); put("tum", "तुम"); put("tumhara", "तुम्हारा")
        put("abhi", "अभी"); put("parson", "परसों"); put("ghanta", "घंटा"); put("ghante", "घंटे")
        put("samay", "समय"); put("waqt", "वक़्त"); put("kiya", "किया"); put("kiye", "किए")
        put("diya", "दिया"); put("diye", "दिए"); put("liya", "लिया"); put("liye", "लिए"); put("lie", "लिए")
        put("gaya", "गया"); put("gayi", "गई"); put("gaye", "गए"); put("raha", "रहा"); put("rahi", "रही")
        put("rahe", "रहे"); put("rahega", "रहेगा"); put("rahegi", "रहेगी"); put("sakte", "सकते")
        put("sakta", "सकता"); put("sakti", "सकती"); put("chahiye", "चाहिए"); put("ji", "जी")
        put("theek", "ठीक"); put("thik", "ठीक"); put("accha", "अच्छा"); put("achha", "अच्छा")
        put("bahut", "बहुत"); put("thoda", "थोड़ा"); put("zyada", "ज़्यादा"); put("kam", "कम")
        put("sab", "सब"); put("kuch", "कुछ"); put("koi", "कोई"); put("ek", "एक"); put("do", "दो")
        put("teen", "तीन"); put("char", "चार"); put("chaar", "चार"); put("paanch", "पाँच")
        put("saat", "सात"); put("aath", "आठ"); put("nau", "नौ"); put("das", "दस")
        put("pehle", "पहले"); put("baad", "बाद"); put("phir", "फिर"); put("fir", "फिर")
        put("jab", "जब"); put("tab", "तब"); put("agar", "अगर"); put("lekin", "लेकिन")
        put("kyunki", "क्योंकि"); put("isliye", "इसलिए"); put("mausam", "मौसम"); put("baarish", "बारिश")
        put("barish", "बारिश"); put("dhoop", "धूप"); put("garmi", "गर्मी"); put("sardi", "सर्दी")
        put("thand", "ठंड"); put("badal", "बादल"); put("baadal", "बादल"); put("tapman", "तापमान")
        put("sambhavna", "संभावना"); put("ummeed", "उम्मीद"); put("apni", "अपनी"); put("apna", "अपना")
        put("apne", "अपने"); put("yaad", "याद"); put("dila", "दिला"); put("dilana", "दिलाना")
        put("bata", "बता"); put("batao", "बताओ"); put("bataiye", "बताइए"); put("bataya", "बताया")
        put("maine", "मैंने"); put("humne", "हमने"); put("aapne", "आपने"); put("hoon", "हूँ")
        put("hu", "हूँ"); put("shukriya", "शुक्रिया"); put("dhanyavaad", "धन्यवाद"); put("namaste", "नमस्ते")
        put("aage", "आगे"); put("wala", "वाला"); put("wali", "वाली"); put("wale", "वाले")
        put("sabse", "सबसे"); put("pehli", "पहली"); put("doosri", "दूसरी"); put("teesri", "तीसरी")
        put("saadhe", "साढ़े"); put("sawa", "सवा"); put("paune", "पौने"); put("dedh", "डेढ़"); put("dhai", "ढाई")
        put("degree", "डिग्री"); put("bangalore", "बैंगलोर"); put("bengaluru", "बेंगलुरु")
        put("tak", "तक"); put("bas", "बस"); put("sirf", "सिर्फ़"); put("ab", "अब"); put("kahin", "कहीं")
        put("yahan", "यहाँ"); put("wahan", "वहाँ"); put("idhar", "इधर"); put("udhar", "उधर")
        put("upar", "ऊपर"); put("neeche", "नीचे"); put("andar", "अंदर"); put("bahar", "बाहर")
        put("saath", "साथ"); put("paas", "पास"); put("door", "दूर"); put("aapse", "आपसे")
        put("hamesha", "हमेशा"); put("kabhi", "कभी"); put("jaldi", "जल्दी"); put("der", "देर")
        put("dekh", "देख"); put("dekho", "देखो"); put("suno", "सुनो"); put("chalo", "चलो")
        put("lagta", "लगता"); put("lagti", "लगती"); put("lagega", "लगेगा"); put("milega", "मिलेगा")
        put("milegi", "मिलेगी"); put("jayega", "जाएगा"); put("jayegi", "जाएगी"); put("jaega", "जाएगा")
        put("aayega", "आएगा"); put("aayegi", "आएगी"); put("hogaya", "हो गया"); put("karta", "करता")
        put("karti", "करती"); put("karte", "करते"); put("karke", "करके"); put("kijiye", "कीजिए")
        put("dijiye", "दीजिए"); put("rakhiye", "रखिए"); put("rakh", "रख"); put("rakha", "रखा")
        put("tayyar", "तैयार"); put("taiyaar", "तैयार"); put("zaroor", "ज़रूर"); put("bilkul", "बिल्कुल")
        put("garam", "गरम"); put("thanda", "ठंडा"); put("halki", "हल्की"); put("halka", "हल्का")
        put("tez", "तेज़"); put("hawa", "हवा"); put("aasman", "आसमान"); put("saaf", "साफ़")
        put("dhundh", "धुंध"); put("toofan", "तूफ़ान"); put("bijli", "बिजली"); put("chhata", "छाता")
        put("ghar", "घर"); put("kaam", "काम"); put("din", "दिन"); put("hafte", "हफ़्ते"); put("mahine", "महीने")
        put("agla", "अगला"); put("agli", "अगली"); put("agle", "अगले"); put("pichla", "पिछला")
        put("baaki", "बाकी"); put("khatam", "ख़त्म"); put("shuru", "शुरू"); put("baje", "बजे")
        put("bajkar", "बजकर"); put("aadha", "आधा"); put("pune", "पुणे"); put("dilli", "दिल्ली")
        put("unka", "उनका"); put("unki", "उनकी"); put("unke", "उनके"); put("inka", "इनका")
        put("sath", "साथ"); put("se", "से"); put("tha", "था"); put("rahenge", "रहेंगे")
    }

    /** Hinglish markers used by [detectLanguage] (unambiguous, never English words). */
    private val HINGLISH_MARKERS = setOf(
        "hai", "hain", "kya", "kal", "aaj", "mujhe", "mein", "aur", "nahi", "nahin", "ka", "ki", "ke",
        "ko", "se", "karna", "kar", "baje", "subah", "shaam", "meri", "mera", "mere", "aap", "aapka",
        "aapki", "aapke", "aapko", "kab", "hoga", "hogi", "rahega", "rahegi", "gaya", "gayi", "diya",
        "liye", "theek", "accha", "haan", "ji", "bhi", "tha", "thi", "kaise", "kyun", "baarish", "mausam",
        "maine", "sambhavna", "tak", "wala", "wali", "abhi", "sakte", "chahiye", "hum", "yeh", "woh",
    )
    private val ENGLISH_MARKERS = setOf(
        "the", "is", "are", "you", "your", "what", "will", "with", "of", "and", "for", "have", "has",
        "it", "this", "that", "be", "can", "set", "at", "tomorrow", "today", "an", "i", "my", "there",
        "here", "expect", "high", "low", "degrees", "o'clock", "please", "next", "done",
    )

    // =========================================================================
    //  Numbers
    // =========================================================================
    private val HI_NUM: Array<String> = arrayOf(
        "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
        "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
        "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
        "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
        "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
        "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
        "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
        "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
        "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
        "नब्बे", "इक्यानबे", "बानबे", "तिरानबे", "चौरानबे", "पचानबे", "छियानबे", "सत्तानबे", "अट्ठानबे", "निन्यानबे",
    )
    private val EN_ONES = arrayOf(
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    )
    private val EN_TENS = arrayOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

    /** Hindi number words, 0..99,99,999 (Indian grouping: सौ, हज़ार, लाख). */
    fun hindiNumber(n: Long): String = when {
        n < 0 -> "माइनस " + hindiNumber(-n)
        n < 100 -> HI_NUM[n.toInt()]
        n < 1000 -> {
            val h = HI_NUM[(n / 100).toInt()] + " सौ"
            if (n % 100 == 0L) h else h + " " + hindiNumber(n % 100)
        }
        n < 100_000 -> {
            val t = hindiNumber(n / 1000) + " हज़ार"
            if (n % 1000 == 0L) t else t + " " + hindiNumber(n % 1000)
        }
        n < 10_000_000 -> {
            val l = hindiNumber(n / 100_000) + " लाख"
            if (n % 100_000 == 0L) l else l + " " + hindiNumber(n % 100_000)
        }
        else -> n.toString().map { HI_NUM[it - '0'] }.joinToString(" ")
    }

    /** English number words (Roman), 0..999,999. */
    fun englishNumber(n: Long): String = when {
        n < 0 -> "minus " + englishNumber(-n)
        n < 20 -> EN_ONES[n.toInt()]
        n < 100 -> EN_TENS[(n / 10).toInt()] + if (n % 10 != 0L) " " + EN_ONES[(n % 10).toInt()] else ""
        n < 1000 -> EN_ONES[(n / 100).toInt()] + " hundred" + if (n % 100 != 0L) " " + englishNumber(n % 100) else ""
        n < 1_000_000 -> englishNumber(n / 1000) + " thousand" + if (n % 1000 != 0L) " " + englishNumber(n % 1000) else ""
        else -> n.toString().map { EN_ONES[it - '0'] }.joinToString(" ")
    }

    private val DEV_DIGIT = mapOf(
        '०' to '0', '१' to '1', '२' to '2', '३' to '3', '४' to '4',
        '५' to '5', '६' to '6', '७' to '7', '८' to '8', '९' to '9',
    )

    // =========================================================================
    //  STT post-correction tables (Whisper-tiny's Roman Hinglish -> canonical).
    // =========================================================================

    /** Assistant-domain spellings Whisper gets wrong in either language (no-ops on correct text). */
    private val EN_SPELLING_FIX: Map<String, String> = mapOf(
        "wether" to "weather", "wheather" to "weather", "weathar" to "weather",
        "forcast" to "forecast", "forcaste" to "forecast", "forecaste" to "forecast", "fourcast" to "forecast",
        "banglor" to "Bangalore", "banglore" to "Bangalore", "bangalor" to "Bangalore", "bengalore" to "Bangalore",
        "bengaluru" to "Bangalore", "bangaluru" to "Bangalore", "bengluru" to "Bangalore", "bangalure" to "Bangalore",
        "metting" to "meeting", "miting" to "meeting", "meting" to "meeting", "meating" to "meeting", "mitting" to "meeting",
        "schedual" to "schedule", "skedule" to "schedule", "shedule" to "schedule", "shedual" to "schedule",
        "schedul" to "schedule", "skejul" to "schedule", "remainder" to "reminder", "reminer" to "reminder",
        "remindar" to "reminder", "alaram" to "alarm", "alarum" to "alarm", "allarm" to "alarm", "calender" to "calendar",
        "temprature" to "temperature", "tempreture" to "temperature",
    )

    /** Split English domain words Whisper sometimes emits ("fore cast"), joined back. */
    private val MERGE_TARGETS: Map<String, String> = listOf(
        "forecast", "reminder", "schedule", "meeting", "tomorrow", "weather", "calendar", "temperature", "humidity",
    ).associateWith { it } + ("bangalore" to "Bangalore")

    /**
     * Curated Hinglish spellings: fused verb compounds Whisper writes as one token, plus
     * variants that collide with CMU English words (so the phonetic key may not touch them).
     */
    private val HI_VARIANTS: Map<String, String> = mapOf(
        "hogaya" to "ho gaya", "hogya" to "ho gaya", "hongaya" to "ho gaya", "hogayi" to "ho gayi",
        "hogai" to "ho gayi", "hogaye" to "ho gaye", "kardiya" to "kar diya", "kardia" to "kar diya",
        "kardo" to "kar do", "kardi" to "kar di", "karde" to "kar de", "kardijiye" to "kar dijiye",
        "bajya" to "baje", "suba" to "subah", "sham" to "shaam",
    )

    /** Spoken clock numbers, rewritten as digits only directly before "baje" ("saat baje" -> "7 baje"). */
    private val HI_CLOCK_NUMBERS: Map<String, Int> = mapOf(
        "ek" to 1, "do" to 2, "teen" to 3, "tin" to 3, "char" to 4, "chaar" to 4, "paanch" to 5, "panch" to 5,
        "chhe" to 6, "chhah" to 6, "che" to 6, "chah" to 6, "saat" to 7, "sat" to 7, "saath" to 7, "sath" to 7,
        "saad" to 7, "aath" to 8, "aat" to 8, "nau" to 9, "das" to 10, "dus" to 10, "gyarah" to 11, "gyara" to 11,
        "barah" to 12, "bara" to 12, "baarah" to 12,
    )
    private val BAJE_FORMS = setOf("baje", "bajkar", "bajke")
    private val TIME_WORDS = setOf("subah", "shaam", "raat", "dopahar", "savere", "baje", "bajkar", "bajke")
    private val KAL_NEXT = TIME_WORDS + setOf("ka", "ki", "ke", "ko", "se", "tak", "mausam", "weather", "meeting", "alarm", "reminder")
    private val KAL_PREV = setOf("aaj", "aur", "mujhe", "parson", "ya")
    private val ACTION_NOUNS = setOf("alarm", "alarms", "reminder", "reminders", "timer", "meeting", "call", "notification")
    private val SET_NEXT = setOf("karna", "karo", "kar", "karde", "kardo", "kijiye", "karke", "karenge", "kiya", "dijiye", "ho")
    private val COMMA_TIME_WORDS = TIME_WORDS + setOf("kal", "aaj", "parson")

    /** "set" as Whisper hears it in Hinglish: sate, seat, sat, saat, sett, isaate ... */
    private val SET_LIKE = Regex("^[iea]?s(?:a|aa|e|ee|ai|ae|ea|ei|i)tt?[ea]?$")
    private val LATIN_WORD = Regex("^[a-z]+$")

    /** Hinglish canonical spellings (the keys of [HINGLISH_TO_DEV]) plus a few common words. */
    private val HINGLISH_CANON: Set<String> by lazy {
        HINGLISH_TO_DEV.keys + setOf("kaisa", "kaisi", "laga", "lagana", "savere", "bajke", "uthana", "jagana", "ghante")
    }

    /** English domain words that may stand as parts of a fused token ("merinext" -> "meri next"). */
    private val EN_SPLIT_WORDS = setOf(
        "alarm", "reminder", "meeting", "meetings", "schedule", "next", "set", "weather", "forecast", "today",
        "tomorrow", "timer", "call", "lab", "review",
    )

    /**
     * Hinglish phonetic key: folds what varies when Hinglish is written by ear — aspiration
     * (bh/b, jh/j), vowel length (aa/a, ee/i), e~i and o~u quality, a post-vocalic h
     * (subah/subha), -ay/-ey endings and gemination — so `mohjee`, `muje` and `mujhey` all
     * meet `mujhe` at "muji".
     */
    internal fun phoneticKey(word: String): String {
        val sb = StringBuilder(word.length)
        for (c in word.lowercase()) if (c in 'a'..'z') sb.append(c)
        var s = sb.toString()
        for ((a, b) in KEY_DIGRAPHS) s = s.replace(a, b)
        s = POST_VOCALIC_H.replace(s, "")
        s = AY_END.replace(s, "e")
        s = s.replace('o', 'u').replace('e', 'i')
        val out = StringBuilder(s.length)
        for (c in s) if (out.isEmpty() || out[out.length - 1] != c) out.append(c)
        return out.toString()
    }

    private val KEY_DIGRAPHS = listOf(
        "chh" to "c", "ch" to "c", "sh" to "s", "kh" to "k", "gh" to "g", "jh" to "j", "bh" to "b", "dh" to "d",
        "th" to "t", "ph" to "f", "z" to "j", "w" to "v", "q" to "k", "x" to "ks", "ck" to "k", "c" to "k",
    )
    private val POST_VOCALIC_H = Regex("(?<=[aeiou])h(?=[^aeiou]|$)")
    private val AY_END = Regex("(ay|ey)$")

    /** phonetic key -> the canonical Hinglish word, only where exactly one canonical word has it. */
    private val KEY_INDEX: Map<String, String> by lazy {
        HINGLISH_CANON.groupBy { phoneticKey(it) }.filterValues { it.size == 1 }.mapValues { it.value.single() }
    }

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

    /**
     * STT tokeniser: whitespace chunks with trailing punctuation peeled off, so `4:30`,
     * `p.m.`, `7-7` and `today's` stay whole while `subah,` -> `subah` `,`.
     */
    private fun sttTokens(text: String): List<String> {
        val out = ArrayList<String>()
        for (chunk in text.trim().split(WHITESPACE)) {
            var end = chunk.length
            while (end > 0 && chunk[end - 1].toString() in PUNCT) end--
            if (end > 0) out.add(chunk.substring(0, end))
            for (k in end until chunk.length) out.add(chunk[k].toString())
        }
        return out
    }

    private val WHITESPACE = Regex("""\s+""")

    // =========================================================================
    //  1. STT transcript normalisation: Devanagari -> Roman Hinglish/English,
    //     then the Hinglish phonetic post-corrector.
    // =========================================================================

    /**
     * Whisper-tiny writes Hindi either in Devanagari or, very often, as Roman Hinglish spelled
     * by ear and fused across word boundaries ("Mohjee kaal subah, 7 bajyakal alarm sate
     * karna hai"). This maps both to the app's canonical Roman Hinglish/English
     * ("Mujhe kal subah 7 baje ka alarm set karna hai") with deterministic table passes:
     *
     *  1. script: Devanagari words via [DEV_TO_ROMAN], else [hunterianTransliterate];
     *  2. split-word merge ("fore cast" -> "forecast") and assistant-domain spellings
     *     ([EN_SPELLING_FIX]: skedule, miting, forcast, banglor, bengaluru ...);
     *  3. Hinglish phonetic lexicon: curated variants, then [phoneticKey] — an unseen spelling
     *     is rewritten only if exactly one canonical word shares its key and it is not a
     *     real English (CMU) word;
     *  4. compound splitter: fused tokens segmented into 2-3 known words by dynamic
     *     programming ("kalsuba" -> "kal subah", "kyahai" -> "kya hai");
     *  5. context rules for homophones: "kaal" -> "kal" next to a time word, "sate"/"seat"
     *     -> "set" after alarm/reminder or before karna, number words before "baje" ->
     *     digits, "baje kal alarm" -> "baje ka alarm", "subah, 7" -> "subah 7".
     *
     * With [languageHint] "en" only step 2 runs, unless the text itself is clearly Hinglish.
     * Words already canonical (and English sentences Whisper produced) pass through untouched.
     */
    fun normalizeSttTranscript(rawText: String, languageHint: String = "hi"): String {
        if (rawText.isBlank()) return ""
        val letters = rawText.count { it.isLetter() && it.code < 0x0900 }
        val shouting = letters >= 6 && rawText.count { it.isUpperCase() } >= letters * 0.7
        val words = ArrayList<String>()
        for (raw in sttTokens(rawText)) {
            if (raw in PUNCT) { words.add(if (raw == "।" || raw == "॥") "." else raw); continue }
            val tok = normalizeDigits(raw)
            words.add(when {
                !isDevanagari(tok) -> if (shouting) tok.lowercase() else tok
                else -> DEV_TO_ROMAN[tok] ?: hunterianTransliterate(tok)
            })
        }
        val merged = mergeSplitWords(words)
        val out = if (languageHint != "en" || looksHinglish(merged)) correctHinglish(merged) else correctEnglish(merged)
        if (shouting && out.isNotEmpty()) out[0] = out[0].replaceFirstChar { it.uppercaseChar() }
        return join(out)
    }

    private fun looksHinglish(words: List<String>): Boolean {
        var hi = 0
        var en = 0
        for (w in words) {
            val lw = w.lowercase()
            if (lw in HINGLISH_MARKERS) hi++
            if (lw in ENGLISH_MARKERS) en++
        }
        return hi >= 2 && hi > en
    }

    /** Keeps [original]'s leading capital on a corrected word at the start of a sentence. */
    private fun recase(original: String, fixed: String, sentenceStart: Boolean): String =
        if (sentenceStart && original.firstOrNull()?.isUpperCase() == true) fixed.replaceFirstChar { it.uppercaseChar() } else fixed

    private fun isSentenceStart(out: List<String>): Boolean =
        out.isEmpty() || out.last() == "." || out.last() == "?" || out.last() == "!"

    private fun mergeSplitWords(words: List<String>): List<String> {
        val out = ArrayList<String>(words.size)
        var i = 0
        while (i < words.size) {
            if (i + 1 < words.size && words[i].all { it.isLetter() } && words[i + 1].all { it.isLetter() }) {
                val joined = (words[i] + words[i + 1]).lowercase()
                val target = MERGE_TARGETS[joined] ?: EN_SPELLING_FIX[joined]?.takeIf { joined.length >= 6 }
                if (target != null) { out.add(recase(words[i], target, isSentenceStart(out))); i += 2; continue }
            }
            out.add(words[i]); i++
        }
        return out
    }

    private fun correctEnglish(words: List<String>): MutableList<String> {
        val out = ArrayList<String>(words.size)
        for (w in words) {
            val fix = EN_SPELLING_FIX[w.lowercase()]
            out.add(if (fix == null) w else recase(w, fix, isSentenceStart(out)))
        }
        return out
    }

    private fun correctHinglish(words: List<String>): MutableList<String> {
        val out = ArrayList<String>(words.size + 4)
        for (w in words) {
            val lw = w.lowercase()
            val fix = if (LATIN_WORD.matches(lw) && lw !in HI_CLOCK_NUMBERS && lw != "kaal" && !SET_LIKE.matches(lw)) {
                resolveHinglish(lw)
            } else null
            if (fix == null) { out.add(w); continue }
            val start = isSentenceStart(out)
            fix.split(' ').forEachIndexed { k, part -> out.add(if (k == 0) recase(w, part, start) else part) }
        }
        applyContextRules(out)
        return out
    }

    /**
     * Canonical replacement for one lower-case token, or null to keep it as written. The fuzzy
     * steps (phonetic key, compound split) need the CMU lexicon to recognise real English words
     * ("say" must not become "se"), so without it only the explicit tables apply.
     */
    private fun resolveHinglish(lw: String): String? {
        HI_VARIANTS[lw]?.let { return it }
        EN_SPELLING_FIX[lw]?.let { return it }
        if (lw in HINGLISH_CANON || lw in ROMAN_TO_DEV) return null
        val english = englishLexicon ?: return null
        if (english.contains(lw)) return null                                    // a real English word
        if (lw.length >= 3) KEY_INDEX[phoneticKey(lw)]?.let { return it }
        return if (lw.length >= 5) segment(lw) else null
    }

    /** One part of a fused token: (canonical spelling, cost), exact words cheaper than key matches. */
    private fun splitPart(p: String): Pair<String, Int>? = when {
        p in HINGLISH_CANON || p in EN_SPLIT_WORDS -> p to 10
        HI_VARIANTS[p]?.let { ' ' !in it } == true -> HI_VARIANTS.getValue(p) to 10
        p.length >= 3 && englishLexicon?.contains(p) != true -> KEY_INDEX[phoneticKey(p)]?.let { it to 13 }
        else -> null
    }

    /** Fewest-parts segmentation of a fused token into 2-3 known words (parts >= 2 letters). */
    private fun segment(w: String): String? {
        val n = w.length
        val cost = IntArray(n + 1) { Int.MAX_VALUE }
        val parts = IntArray(n + 1)
        val back = IntArray(n + 1) { -1 }
        val word = arrayOfNulls<String>(n + 1)
        cost[0] = 0
        for (i in 2..n) {
            for (j in maxOf(0, i - 12)..i - 2) {
                if (cost[j] == Int.MAX_VALUE || parts[j] >= 3) continue
                val (canon, c) = splitPart(w.substring(j, i)) ?: continue
                if (cost[j] + c < cost[i]) { cost[i] = cost[j] + c; parts[i] = parts[j] + 1; back[i] = j; word[i] = canon }
            }
        }
        if (cost[n] == Int.MAX_VALUE || parts[n] < 2) return null
        val result = ArrayList<String>(3)
        var i = n
        while (i > 0) { result.add(word[i]!!); i = back[i] }
        return result.asReversed().joinToString(" ")
    }

    /** Homophones resolved only by their neighbours (see [normalizeSttTranscript] step 5). */
    private fun applyContextRules(w: MutableList<String>) {
        fun lw(i: Int): String? = w.getOrNull(i)?.lowercase()
        fun start(i: Int) = i == 0 || w[i - 1] == "." || w[i - 1] == "?" || w[i - 1] == "!"
        for (i in 0 until w.size - 1) {
            val n = HI_CLOCK_NUMBERS[lw(i)] ?: continue
            if (lw(i + 1) in BAJE_FORMS) w[i] = n.toString()
        }
        for (i in w.indices) {
            if (lw(i) != "kaal") continue
            val next = lw(i + 1)
            if (next in KAL_NEXT || next?.firstOrNull()?.isDigit() == true || lw(i - 1) in KAL_PREV) {
                w[i] = recase(w[i], "kal", start(i))
            }
        }
        for (i in w.indices) {
            val t = lw(i) ?: continue
            if (t == "set" || !SET_LIKE.matches(t) || lw(i + 1) in BAJE_FORMS) continue
            val nounBefore = (maxOf(0, i - 6) until i).any { lw(it) in ACTION_NOUNS }
            val canonical = t in HINGLISH_CANON                                    // "saat" is also "seven"
            if (nounBefore || (!canonical && lw(i + 1) in SET_NEXT)) w[i] = recase(w[i], "set", start(i))
        }
        for (i in 0 until w.size - 2) {
            if (lw(i) == "baje" && lw(i + 1) == "kal" && lw(i + 2) in ACTION_NOUNS) w[i + 1] = "ka"
        }
        var i = 1
        while (i < w.size - 1) {
            val next = w[i + 1]
            if (w[i] == "," && lw(i - 1) in COMMA_TIME_WORDS && (next.first().isDigit() || next.lowercase() in HI_CLOCK_NUMBERS)) {
                w.removeAt(i)
            } else i++
        }
    }

    /**
     * Raw on-device STT text -> the transcript the feed shows and the LLM receives:
     * [normalizeSttTranscript] (script + Hinglish phonetic correction) then [cleanAsrText].
     * [language] is the step-0 LID decision ("en" / "hi").
     */
    fun displayTranscript(raw: String, language: String): String =
        cleanAsrText(normalizeSttTranscript(raw, language))

    private val RE_NUM_STUTTER = Regex("""\b(\d{1,2})(?:\s*-\s*\1\b)+""")
    private val RE_WORD_STUTTER = Regex("""\b(\p{L}+)(?:\s+\1\b)+""", RegexOption.IGNORE_CASE)

    /**
     * Removes Whisper-tiny decoding stutters before the transcript reaches the LLM:
     * `7-7 baji` -> `7 baji` (which an LLM would otherwise read as 7:07), `the the` -> `the`.
     */
    fun cleanAsrText(text: String): String =
        RE_WORD_STUTTER.replace(RE_NUM_STUTTER.replace(text, "$1"), "$1").replace(Regex("""\s{2,}"""), " ").trim()

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
                '़' -> { /* nukta: keep base consonant */ }
                else -> { flush(true); if (c.code in 0x20..0x7E) sb.append(c) }
            }
        }
        flush(false)   // word-final schwa deletion
        return sb.toString()
    }

    // =========================================================================
    //  2. TTS acoustic script: Roman/English + numerics -> phonetic Devanagari.
    // =========================================================================
    private val RE_TEMP = Regex("""(-?\d+(?:\.\d+)?)\s*°\s*([CF])\b|(-?\d+(?:\.\d+)?)\s*([CF])\b(?![a-zA-Z])""")
    private val RE_TIME = Regex("""\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\b\.?|\b(\d{1,2}):(\d{2})\b""")
    private val RE_AMPM = Regex("""\b(\d{1,2})\s*([AaPp])\.?\s*[Mm]\b\.?""")
    private val RE_PERCENT = Regex("""(\d+(?:\.\d+)?)\s*%""")
    private val RE_DECIMAL = Regex("""\b(\d+)\.(\d+)\b""")
    private val RE_ORDINAL = Regex("""\b(\d+)(st|nd|rd|th)\b""")
    private val RE_NUM = Regex("""\d+""")

    private val EN_ORDINAL = mapOf(
        1 to "first", 2 to "second", 3 to "third", 4 to "fourth", 5 to "fifth", 6 to "sixth", 7 to "seventh",
        8 to "eighth", 9 to "ninth", 10 to "tenth", 11 to "eleventh", 12 to "twelfth", 13 to "thirteenth",
        14 to "fourteenth", 15 to "fifteenth", 16 to "sixteenth", 17 to "seventeenth", 18 to "eighteenth",
        19 to "nineteenth", 20 to "twentieth", 30 to "thirtieth",
    )

    /** "en" if [text] reads as English, else "hi" (Hinglish/Hindi, also the default). */
    fun detectLanguage(text: String): String {
        if (isDevanagari(text)) return "hi"
        var hi = 0
        var en = 0
        for (w in Regex("[A-Za-z']+").findAll(text)) {
            val lw = w.value.lowercase()
            if (lw in HINGLISH_MARKERS) hi++
            if (lw in ENGLISH_MARKERS) en++
        }
        return if (en > hi) "en" else "hi"
    }

    /**
     * Display text (Roman Hinglish / English / Devanagari) -> phonetic Devanagari for the
     * Indic VITS voice. [language] is "en", "hi", or "auto" ([detectLanguage]).
     */
    fun toAcousticDevanagariScript(displayText: String, language: String = "auto"): String {
        if (displayText.isBlank()) return ""
        val lang = if (language == "en" || language == "hi") language else detectLanguage(displayText)
        val en = lang == "en"
        var s = normalizeDigits(displayText)
        s = s.replace(Regex("""\[ACTION:[^\]]*]"""), " ")
            .replace(Regex("[–—]"), ", ").replace('“', ' ').replace('”', ' ').replace('"', ' ')

        s = RE_TEMP.replace(s) { m ->
            val num = m.groupValues[1].ifEmpty { m.groupValues[3] }
            val unitC = (m.groupValues[2].ifEmpty { m.groupValues[4] }).uppercase() == "C"
            val unit = if (unitC) "celsius" else "fahrenheit"
            " " + number(num, en) + (if (en) " degrees " else " डिग्री ") + unit + " "
        }
        s = RE_TIME.replace(s) { m ->
            if (m.groupValues[1].isNotEmpty()) timePhrase(m.groupValues[1].toInt(), m.groupValues[2].toInt(), m.groupValues[3], en)
            else timePhrase(m.groupValues[4].toInt(), m.groupValues[5].toInt(), "", en)
        }
        s = RE_AMPM.replace(s) { m -> " " + number(m.groupValues[1], en) + ampm(m.groupValues[2]) + " " }
        s = RE_PERCENT.replace(s) { m -> " " + number(m.groupValues[1], en) + (if (en) " percent " else " प्रतिशत ") }
        s = RE_ORDINAL.replace(s) { m ->
            val n = m.groupValues[1].toInt()
            " " + (if (en) EN_ORDINAL[n] ?: englishNumber(n.toLong()) else hindiNumber(n.toLong())) + " "
        }
        s = RE_DECIMAL.replace(s) { m -> " " + number(m.value, en) + " " }
        s = RE_NUM.replace(s) { m -> " " + number(m.value, en) + " " }

        // Word-level Roman/English -> Devanagari; Devanagari & punctuation pass through.
        val out = ArrayList<String>()
        for (tok in tokenize(s)) {
            when {
                tok in PUNCT -> out.add(tok)
                isDevanagari(tok) -> out.add(tok)
                else -> {
                    val parts = tok.split('-', '/', '_').filter { it.isNotEmpty() }
                    for (p in parts) wordToDevanagari(p, en).takeIf { it.isNotBlank() }?.let { out.add(it) }
                }
            }
        }
        return join(out)
    }

    /** Spoken number (integer or decimal) in the requested language; Roman words for EN. */
    private fun number(raw: String, en: Boolean): String {
        val parts = raw.split('.')
        val whole = parts[0].toLongOrNull() ?: return raw
        val base = if (en) englishNumber(whole) else hindiNumber(whole)
        if (parts.size < 2 || parts[1].isEmpty()) return base
        val frac = parts[1].map { d -> if (en) EN_ONES[d - '0'] else HI_NUM[d - '0'] }.joinToString(" ")
        return base + (if (en) " point " else " दशमलव ") + frac
    }

    private fun ampm(letter: String): String = when (letter.uppercase()) {
        "A" -> " ए एम"
        "P" -> " पी एम"
        else -> ""
    }

    /** Spoken clock time: Hindi uses सवा / साढ़े / पौने, English "four thirty". */
    private fun timePhrase(h: Int, m: Int, ap: String, en: Boolean): String {
        val suffix = ampm(ap)
        val phrase = if (en) {
            when {
                m == 0 -> englishNumber(h.toLong())
                m < 10 -> englishNumber(h.toLong()) + " o " + englishNumber(m.toLong())
                else -> englishNumber(h.toLong()) + " " + englishNumber(m.toLong())
            }
        } else {
            val h12 = if (h % 12 == 0) 12 else h % 12
            val next = if (h12 == 12) 1 else h12 + 1
            when (m) {
                0 -> hindiNumber(h.toLong())
                15 -> "सवा " + hindiNumber(h12.toLong())
                30 -> when (h12) { 1 -> "डेढ़"; 2 -> "ढाई"; else -> "साढ़े " + hindiNumber(h12.toLong()) }
                45 -> "पौने " + hindiNumber(next.toLong())
                else -> hindiNumber(h.toLong()) + " " + hindiNumber(m.toLong())
            }
        }
        return " $phrase$suffix "
    }

    /** Routes one Latin token to Devanagari (see class doc for the priority order). */
    private fun wordToDevanagari(token: String, en: Boolean): String {
        val word = token.trim('\'', '’', '(', ')', '[', ']', '{', '}', '*', '#', '"')
        if (word.isEmpty()) return ""
        if (word.none { it.isLetter() }) return ""
        val lw = word.lowercase().replace('’', '\'')
        ROMAN_TO_DEV[lw]?.let { if (en || lw !in HINGLISH_TO_DEV) return it }
        if (isAcronym(word)) return spellLetters(word)
        if (!en) HINGLISH_TO_DEV[lw]?.let { return it }
        englishLexicon?.phones(lw)?.let { if (en || !looksHinglish(lw)) return ArpabetDevanagari.convert(it, lw) }
        if (en) HINGLISH_TO_DEV[lw]?.let { return it }
        return romanHindiToDevanagari(lw)
    }

    private fun isAcronym(w: String): Boolean {
        val letters = w.filter { it.isLetter() }
        return letters.length >= 2 && letters.count { it.isUpperCase() } >= 2 &&
            letters.count { it.isUpperCase() } >= letters.length - 1
    }

    private val LETTER_DEV = mapOf(
        'a' to "ए", 'b' to "बी", 'c' to "सी", 'd' to "डी", 'e' to "ई", 'f' to "एफ़", 'g' to "जी",
        'h' to "एच", 'i' to "आई", 'j' to "जे", 'k' to "के", 'l' to "एल", 'm' to "एम", 'n' to "एन",
        'o' to "ओ", 'p' to "पी", 'q' to "क्यू", 'r' to "आर", 's' to "एस", 't' to "टी", 'u' to "यू",
        'v' to "वी", 'w' to "डब्ल्यू", 'x' to "एक्स", 'y' to "वाई", 'z' to "ज़ेड",
    )

    private fun spellLetters(w: String): String =
        w.lowercase().mapNotNull { LETTER_DEV[it] }.joinToString(" ")

    /** Roman spellings that are clearly Hindi (long-vowel digraphs, retroflex/aspirate clusters). */
    private fun looksHinglish(w: String): Boolean =
        Regex("aa|ii|uu|bh|dh|gh|jh|kh|chh").containsMatchIn(w) && !w.endsWith("ing") && !w.endsWith("tion")

    // --- Roman Hinglish -> Devanagari transliteration ------------------------
    private val R_CONS: List<Pair<String, String>> = listOf(
        "chh" to "छ", "kh" to "ख", "gh" to "घ", "ch" to "च", "jh" to "झ", "th" to "थ", "dh" to "ध",
        "ph" to "फ", "bh" to "भ", "sh" to "श", "k" to "क", "g" to "ग", "c" to "क", "j" to "ज",
        "t" to "त", "d" to "द", "n" to "न", "p" to "प", "b" to "ब", "m" to "म", "y" to "य", "r" to "र",
        "l" to "ल", "v" to "व", "w" to "व", "s" to "स", "h" to "ह", "z" to "ज़", "f" to "फ़", "q" to "क़",
        "x" to "क्स",
    )
    private val R_VOW: List<Triple<String, String, String>> = listOf(   // roman, matra, independent
        Triple("aa", "ा", "आ"), Triple("ai", "ै", "ऐ"), Triple("au", "ौ", "औ"), Triple("ee", "ी", "ई"),
        Triple("ii", "ी", "ई"), Triple("oo", "ू", "ऊ"), Triple("uu", "ू", "ऊ"), Triple("ei", "ै", "ऐ"),
        Triple("ou", "ौ", "औ"), Triple("a", "", "अ"), Triple("i", "ि", "इ"), Triple("u", "ु", "उ"),
        Triple("e", "े", "ए"), Triple("o", "ो", "ओ"),
    )

    /** ITRANS-like Roman Hindi -> Devanagari (final -a -> ा, final -i -> ी, n+C -> anusvara). */
    fun romanHindiToDevanagari(word: String): String {
        val w = word.lowercase().filter { it in 'a'..'z' }
        val sb = StringBuilder()
        var open = false            // last emitted letter is a consonant without a vowel sign
        var i = 0
        while (i < w.length) {
            val v = R_VOW.firstOrNull { w.startsWith(it.first, i) }
            if (v != null) {
                val end = i + v.first.length >= w.length
                val matra = when {
                    end && v.first == "a" && open -> "ा"
                    end && v.first == "i" && open -> "ी"
                    else -> v.second
                }
                sb.append(if (open) matra else v.third)
                open = false
                i += v.first.length
                continue
            }
            val c = R_CONS.firstOrNull { w.startsWith(it.first, i) }
            if (c == null) { i++; continue }
            val after = i + c.first.length
            val nextIsCons = after < w.length && R_VOW.none { w.startsWith(it.first, after) }
            if (c.first == "n" && !open && sb.isNotEmpty() && nextIsCons) {
                sb.append("ं")                                     // n + consonant -> anusvara
            } else {
                if (open) sb.append("्")
                sb.append(c.second)
                open = true
            }
            i = after
        }
        return sb.toString()
    }

    // --- Transliteration tables (Devanagari -> Roman) -------------------------
    private val H_CONS: Map<Char, String> = mapOf(
        'क' to "k", 'ख' to "kh", 'ग' to "g", 'घ' to "gh", 'ङ' to "ng",
        'च' to "ch", 'छ' to "chh", 'ज' to "j", 'झ' to "jh", 'ञ' to "ny",
        'ट' to "t", 'ठ' to "th", 'ड' to "d", 'ढ' to "dh", 'ण' to "n",
        'त' to "t", 'थ' to "th", 'द' to "d", 'ध' to "dh", 'न' to "n",
        'प' to "p", 'फ' to "ph", 'ब' to "b", 'भ' to "bh", 'म' to "m",
        'य' to "y", 'र' to "r", 'ल' to "l", 'व' to "v", 'श' to "sh",
        'ष' to "sh", 'स' to "s", 'ह' to "h", 'ळ' to "l", 'क़' to "q",
        'ख़' to "kh", 'ग़' to "gh", 'ज़' to "z", 'ड़' to "r", 'ढ़' to "rh", 'फ़' to "f",
        'य़' to "y",
    )
    private val H_MATRA: Map<Char, String> = mapOf(
        'ा' to "aa", 'ि' to "i", 'ी' to "ee", 'ु' to "u",
        'ू' to "oo", 'ृ' to "ri", 'े' to "e", 'ै' to "ai",
        'ो' to "o", 'ौ' to "au", 'ॉ' to "o", 'ॅ' to "e",
    )
    private val H_VOWEL: Map<Char, String> = mapOf(
        'अ' to "a", 'आ' to "aa", 'इ' to "i", 'ई' to "ee", 'उ' to "u", 'ऊ' to "oo",
        'ऋ' to "ri", 'ए' to "e", 'ऐ' to "ai", 'ओ' to "o", 'औ' to "au", 'ऑ' to "o",
    )
}
