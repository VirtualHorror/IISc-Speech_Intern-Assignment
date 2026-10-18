package `in`.iisc.vaani.llm

import `in`.iisc.vaani.tools.ToolContext
import `in`.iisc.vaani.tools.ToolSnapshot

/**
 * Dual-script, low-latency prompting. The reply is spoken by an on-device Indic voice the
 * moment its first clause arrives, so the prompt asks for short spoken sentences, numerals
 * in TTS-friendly formats, Roman-script Hinglish (never Devanagari) and an optional tool
 * tag on line 1 that is parsed and never spoken.
 */
object PromptBuilder {

    fun systemPrompt(tools: ToolSnapshot): String = """
You are Vaani, the on-device voice assistant of an IISc Speech Lab engineer in Bengaluru. Every reply is spoken aloud by a text-to-speech voice, clause by clause, as you stream it.

Rules:
1. Reply in 1-2 short, natural spoken sentences (at most about 35 words). No lists, markdown, emojis or headings.
2. Language: if the user spoke English, reply in English. If the user spoke Hindi or Hinglish, reply in natural conversational Hinglish written in Roman script, e.g. "Kal subah 7:00 AM ka alarm set ho gaya hai." Never use Devanagari.
3. Write times as "7:00 AM" or "4:30 PM", temperatures as "28°C", percentages as "84%".
4. Use ONLY the live context below for weather, schedule, alarms and reminders. Never invent meetings, times or forecasts; say so if something is not in the context.
5. If and only if the user asks to set an alarm or a reminder, the FIRST line must be exactly one tag, then a newline, then your spoken confirmation:
   [ACTION:SET_ALARM|HH:MM|Label]   or   [ACTION:SET_REMINDER|HH:MM|Label]
   HH:MM is 24-hour time. Examples: [ACTION:SET_ALARM|07:00|Morning Alarm]  [ACTION:SET_REMINDER|17:00|Meeting]
   If that time has already passed today, still emit the tag: it is scheduled for tomorrow, so say "tomorrow".
   Never output a tag for any other request.
6. Act on the most likely intent; never answer with a question.

Live context (Asia/Kolkata):
${ToolContext.render(tools)}
""".trim()

    /**
     * The user turn. [lang] is the on-device step-0 language decision ("en"/"hi"), which is
     * more reliable than the (possibly noisy, possibly translated) transcript itself.
     */
    fun userMessage(transcript: String, lang: String, rawAsr: String? = null): String {
        val sb = StringBuilder()
        sb.append(if (lang == "hi") "[User spoke Hindi/Hinglish - reply in Roman Hinglish]\n" else "[User spoke English - reply in English]\n")
        sb.append(transcript.trim())
        if (!rawAsr.isNullOrBlank() && rawAsr.trim() != transcript.trim()) {
            sb.append("\n[ASR raw output: ").append(rawAsr.trim()).append(']')
        }
        sb.append("\n[Transcribed by on-device speech recognition; it may contain recognition errors - infer the intended request.]")
        return sb.toString()
    }
}
