package `in`.iisc.vaani.tools

/** Grounded side-effects the assistant can request (see [ActionTagParser]). */
enum class ActionType { SET_ALARM, SET_REMINDER }

data class ToolAction(
    val type: ActionType,
    val hour: Int,
    val minute: Int,
    val label: String,
    /** Where the action came from: the LLM's line-1 tag, or the local intent fallback. */
    val source: String = "llm-tag",
)

/**
 * Parses the tool tags the system prompt asks the LLM to emit on line 1, e.g.
 * `[ACTION:SET_ALARM|07:00|Morning Alarm]` or `[ACTION:SET_REMINDER|17:00|Meeting]`.
 * Tolerant of spacing/case and `7.00`, strict about ranges (24-hour HH:MM).
 */
object ActionTagParser {
    private val TAG = Regex(
        """\[\s*ACTION\s*:\s*(SET_ALARM|SET_REMINDER)\s*\|\s*(\d{1,2})\s*[:.]\s*(\d{2})\s*(?:\|\s*([^\]]*))?]""",
        RegexOption.IGNORE_CASE,
    )

    fun parse(text: String): List<ToolAction> = TAG.findAll(text).mapNotNull { m ->
        val type = if (m.groupValues[1].uppercase() == "SET_ALARM") ActionType.SET_ALARM else ActionType.SET_REMINDER
        val h = m.groupValues[2].toInt()
        val min = m.groupValues[3].toInt()
        if (h !in 0..23 || min !in 0..59) return@mapNotNull null
        val label = m.groupValues[4].trim().ifEmpty { if (type == ActionType.SET_ALARM) "Alarm" else "Reminder" }
        ToolAction(type, h, min, label)
    }.toList()

    fun strip(text: String): String = TAG.replace(text, " ").replace(Regex("[ \t]{2,}"), " ").trim()
}
