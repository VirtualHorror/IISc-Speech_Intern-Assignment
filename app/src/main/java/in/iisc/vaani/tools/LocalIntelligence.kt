package `in`.iisc.vaani.tools

import java.time.LocalTime
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

/** Immutable view of every grounded tool, rendered into the LLM prompt and the cards. */
data class ToolSnapshot(
    val now: ZonedDateTime,
    val forecast: DailyForecast,
    val schedule: List<ScheduleEvent>,
    val alarms: List<AlarmItem>,
    val reminders: List<ReminderItem>,
)

enum class AgentIntent { WEATHER, ALARM, REMINDER, SCHEDULE }

/**
 * Deterministic, on-device understanding of the four IISc scenarios: intent keywords
 * (Roman + Devanagari, ASR-misspelling tolerant) and English/Hinglish clock times.
 * Used to pick the action cards, as a backstop when the LLM forgets a tool tag, and
 * by [LocalResponder] when no cloud LLM is reachable.
 */
object LocalIntents {
    private val WEATHER = listOf("weather", "wether", "forecast", "temperature", "rain", "mausam", "mosam",
        "barish", "baarish", "baris", "dhoop", "umbrella", "chhata", "वेदर", "मौसम", "बारिश", "तापमान", "फोरकास्ट")
    private val ALARM = listOf("alarm", "alaram", "wake me", "jaga", "uthana", "utha dena", "अलार्म")
    private val REMINDER = listOf("remind", "reminder", "remainder", "yaad dila", "yaad", "रिमाइंडर", "याद")
    private val SCHEDULE = listOf("schedule", "shedule", "schedual", "agenda", "calendar", "meeting", "meetings",
        "शेड्यूल", "मीटिंग", "plan for today", "aaj ka plan")

    fun detect(text: String): Set<AgentIntent> {
        val t = text.lowercase(Locale.ROOT)
        val out = LinkedHashSet<AgentIntent>()
        if (ALARM.any { it in t }) out.add(AgentIntent.ALARM)
        if (REMINDER.any { it in t }) out.add(AgentIntent.REMINDER)
        if (WEATHER.any { it in t }) out.add(AgentIntent.WEATHER)
        // "reminder for my meeting at 5" is a reminder, not a schedule query.
        if (SCHEDULE.any { it in t } && AgentIntent.REMINDER !in out && AgentIntent.ALARM !in out) out.add(AgentIntent.SCHEDULE)
        return out
    }

    fun mentionsTomorrow(text: String): Boolean {
        val t = text.lowercase(Locale.ROOT)
        // prefix match: Whisper often fuses "kal subah" into "kalsuba"
        return Regex("""\b(tomorrow|kal)""").containsMatchIn(t) || "कल" in t || "टुमारो" in t
    }

    private val NUM_WORDS = mapOf(
        "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6, "seven" to 7, "eight" to 8,
        "nine" to 9, "ten" to 10, "eleven" to 11, "twelve" to 12,
        "ek" to 1, "do" to 2, "teen" to 3, "char" to 4, "chaar" to 4, "paanch" to 5, "panch" to 5, "chhe" to 6,
        "che" to 6, "chheh" to 6, "saat" to 7, "sat" to 7, "aath" to 8, "nau" to 9, "das" to 10, "gyarah" to 11,
        "barah" to 12,
    )
    private val RE_HM_AMPM = Regex("""\b(\d{1,2})[:.](\d{2})\s*([ap])\.?\s*m\b""")
    private val RE_H_AMPM = Regex("""\b(\d{1,2})\s*([ap])\.?\s*m\b""")
    private val RE_BAJE = Regex("""\b(\d{1,2})(?:[:.](\d{2}))?\s*(?:-\s*\d{1,2}\s*)?baj""")
    private val RE_24H = Regex("""\b([01]?\d|2[0-3]):([0-5]\d)\b""")
    private val RE_WORD_TIME = Regex("""\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|ek|do|teen|chaar|char|paanch|panch|chhe|che|saat|sat|aath|nau|das|gyarah|barah)\s+(o'clock|am|pm|a\.m\.|p\.m\.|baje|baj)""")

    /**
     * Extracts an HH:MM (24 h) from English or Hinglish, e.g. "5 PM", "4:30 pm",
     * "subah 7 baje", "shaam paanch baje", "17:00". [preferMorning] decides bare hours
     * (alarms default to AM, meetings/reminders to PM).
     */
    fun parseTime(text: String, preferMorning: Boolean): Pair<Int, Int>? {
        val t = text.lowercase(Locale.ROOT)
        val morning = listOf("subah", "morning", "suba", "सुबह").any { it in t }
        val evening = listOf("shaam", "sham", "evening", "raat", "night", "dopahar", "afternoon", "शाम", "रात").any { it in t }
        fun ampm(h: Int, pm: Boolean?): Int = when {
            pm == true -> if (h % 12 == 0) 12 else (h % 12) + 12
            pm == false -> h % 12
            morning -> h % 12
            evening -> if (h % 12 == 0) 12 else (h % 12) + 12
            h >= 13 -> h
            preferMorning -> if (h in 4..11) h else if (h == 12) 12 else (h % 12) + 12
            else -> if (h in 1..7) h + 12 else h
        }
        RE_HM_AMPM.find(t)?.let { m ->
            return ampm(m.groupValues[1].toInt(), m.groupValues[3] == "p") to m.groupValues[2].toInt()
        }
        RE_H_AMPM.find(t)?.let { m -> return ampm(m.groupValues[1].toInt(), m.groupValues[2] == "p") to 0 }
        RE_BAJE.find(t)?.let { m ->
            return ampm(m.groupValues[1].toInt(), null) to (m.groupValues[2].toIntOrNull() ?: 0)
        }
        RE_WORD_TIME.find(t)?.let { m ->
            val h = NUM_WORDS.getValue(m.groupValues[1])
            val suffix = m.groupValues[2]
            val pm = when { suffix.startsWith("p") -> true; suffix.startsWith("a") -> false; else -> null }
            return ampm(h, pm) to 0
        }
        RE_24H.find(t)?.let { m -> return m.groupValues[1].toInt() to m.groupValues[2].toInt() }
        return null
    }
}

/**
 * Offline brain: answers the four IISc scenarios from the grounded tools when no cloud
 * LLM is configured/reachable, in the user's language, with the same action semantics.
 */
object LocalResponder {
    data class Reply(val text: String, val actions: List<ToolAction>, val intents: Set<AgentIntent>)

    fun respond(transcript: String, lang: String, s: ToolSnapshot): Reply {
        val hi = lang == "hi"
        val intents = LocalIntents.detect(transcript)
        val actions = ArrayList<ToolAction>()
        val text = when {
            AgentIntent.ALARM in intents -> {
                val (h, m) = LocalIntents.parseTime(transcript, preferMorning = true) ?: (7 to 0)
                actions.add(ToolAction(ActionType.SET_ALARM, h, m, if (h < 12) "Morning Alarm" else "Alarm", "local"))
                val date = nextOccurrence(s.now, h, m, LocalIntents.mentionsTomorrow(transcript))
                val day = dayName(date, s.now.toLocalDate())
                if (hi) "Theek hai, aapka alarm ${if (day == "Tomorrow") "kal" else "aaj"} ${clock12(h, m)} ke liye set ho gaya hai."
                else "Done, your alarm is set for ${clock12(h, m)} ${day.lowercase()}."
            }
            AgentIntent.REMINDER in intents -> {
                val (h, m) = LocalIntents.parseTime(transcript, preferMorning = false) ?: (17 to 0)
                val label = if ("meeting" in transcript.lowercase() || "मीटिंग" in transcript) "Meeting" else "Reminder"
                actions.add(ToolAction(ActionType.SET_REMINDER, h, m, label, "local"))
                if (hi) "Theek hai, maine ${clock12(h, m)} ka ${label.lowercase()} reminder set kar diya hai."
                else "Done, I've set a reminder for your ${label.lowercase()} at ${clock12(h, m)}."
            }
            AgentIntent.WEATHER in intents -> {
                val f = s.forecast
                if (hi) "Kal Bangalore mein ${f.descriptionHi} ki sambhavna hai, temperature ${f.minRounded} se ${f.maxRounded} degree tak rahega, aur baarish ka chance ${f.precipProbPct} percent hai."
                else "Tomorrow in Bangalore expect ${f.descriptionEn}, with a high of ${f.maxRounded}°C, a low of ${f.minRounded}°C and a ${f.precipProbPct}% chance of rain."
            }
            AgentIntent.SCHEDULE in intents -> {
                val now = s.now.toLocalTime()
                val next = ScheduleGenerator.next(s.schedule, now)
                val n = s.schedule.size
                when {
                    next == null && hi -> "Aaj aapki $n meetings thi, aur ab koi meeting baaki nahi hai."
                    next == null -> "You had $n meetings today and none are left."
                    hi -> "Aaj aapki $n meetings hain, aur aapki next meeting ${next.startLabel} par ${next.title} hai."
                    else -> "Today you have $n meetings, and your next one is ${next.title} at ${next.startLabel}."
                }
            }
            hi -> "Maaf kijiye, main abhi sirf mausam, alarm, reminder aur aaj ke schedule mein madad kar sakti hoon."
            else -> "Sorry, offline I can help with the weather, alarms, reminders and today's schedule."
        }
        return Reply(text, actions, intents)
    }
}

/** Renders the grounded context block injected into the LLM system prompt. */
object ToolContext {
    private val DAY_FMT = DateTimeFormatter.ofPattern("EEEE, d MMM yyyy", Locale.US)
    private val SHORT_DAY = DateTimeFormatter.ofPattern("EEE d MMM", Locale.US)

    fun render(s: ToolSnapshot): String {
        val now = s.now
        val today = now.toLocalDate()
        val nowT = now.toLocalTime()
        val f = s.forecast
        val sb = StringBuilder()
        sb.append("- Now: ").append(now.format(DAY_FMT)).append(", ").append(clock12(now.hour, now.minute)).append(" IST\n")
        sb.append("- Bengaluru weather tomorrow (").append(f.date.format(SHORT_DAY)).append("): ")
            .append(f.descriptionEn).append(", ").append(fmt(f.tMinC)).append("–").append(fmt(f.tMaxC))
            .append("°C, ").append(f.precipProbPct).append("% chance of rain [").append(f.source.label).append("]\n")
        sb.append("- Today's schedule: ")
        sb.append(ScheduleGenerator.withStatus(s.schedule, nowT).joinToString("; ") { (e, st) ->
            "${e.startLabel} ${e.title}" + when (st) {
                EventStatus.DONE -> " (done)"
                EventStatus.ONGOING -> " (in progress)"
                EventStatus.NEXT -> " (NEXT, in ${minutesUntil(nowT, e.start)} min)"
                EventStatus.UPCOMING -> ""
            }
        }).append('\n')
        val next = ScheduleGenerator.next(s.schedule, nowT)
        sb.append("- Next meeting: ").append(next?.let { "${it.startLabel} ${it.title}" } ?: "none left today").append('\n')
        sb.append("- Active alarms: ")
            .append(s.alarms.joinToString("; ") { "${it.timeLabel} ${it.dayLabel(today)} (${it.label})" }.ifEmpty { "none" })
            .append('\n')
        sb.append("- Active reminders: ")
            .append(s.reminders.joinToString("; ") { "${it.timeLabel} ${it.dayLabel(today)} (${it.label})" }.ifEmpty { "none" })
        return sb.toString()
    }

    private fun fmt(v: Double) = if (v == Math.floor(v)) v.toInt().toString() else String.format(Locale.US, "%.1f", v)

    private fun minutesUntil(now: LocalTime, t: LocalTime) = (t.toSecondOfDay() - now.toSecondOfDay()) / 60
}
