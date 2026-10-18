package `in`.iisc.vaani.llm

import `in`.iisc.vaani.tools.ActionTagParser
import `in`.iisc.vaani.tools.ToolAction

/**
 * Cuts a token stream into speakable clauses as early as linguistically safe:
 *  - at `?`, `!`, `।` and newlines immediately;
 *  - at `.` only when followed by whitespace (so `28.4`, `4.30` and `p.m.` never split);
 *  - at `,` / `;` once the clause has at least [minCommaWords] words;
 *  - at a space once a clause runs past [maxChars] (no unbounded waits).
 * A boundary at the very end of the buffer is held until the next token (or [flush]).
 */
class ClauseSegmenter(private val minCommaWords: Int = 5, private val maxChars: Int = 160) {
    private val buf = StringBuilder()

    fun push(text: String): List<String> {
        buf.append(text)
        return drain(final = false)
    }

    fun flush(): List<String> = drain(final = true)

    private fun drain(final: Boolean): List<String> {
        val out = ArrayList<String>()
        var start = 0
        var i = 0
        while (i < buf.length) {
            val c = buf[i]
            val next: Char? = if (i + 1 < buf.length) buf[i + 1] else null
            val cut = when (c) {
                '?', '!', '।', '\n' -> true
                '.' -> when {
                    next == null -> final
                    !next.isWhitespace() -> false
                    // "p.m. tonight" continues, "p.m. Your..." ends: decide on the next word
                    isAbbreviation(start, i) -> {
                        var k = i + 1
                        while (k < buf.length && buf[k].isWhitespace()) k++
                        if (k >= buf.length) final else buf[k].isUpperCase()
                    }
                    else -> true
                }
                ',', ';' -> words(start, i) >= minCommaWords && (if (next == null) final else next.isWhitespace())
                ' ' -> i - start >= maxChars
                else -> false
            }
            if (cut) {
                emit(buf.substring(start, i + 1), out)
                start = i + 1
            }
            i++
        }
        if (final) {
            emit(buf.substring(start), out)
            buf.setLength(0)
        } else {
            buf.delete(0, start)
        }
        return out
    }

    private fun emit(raw: String, out: MutableList<String>) {
        val clause = raw.trim()
        if (clause.any { it.isLetterOrDigit() }) out.add(clause)
    }

    /** The word ending just before the '.' at [dot] is `p.m` / `e.g` / a title like `Dr`. */
    private fun isAbbreviation(from: Int, dot: Int): Boolean {
        var s = dot
        while (s > from && !buf[s - 1].isWhitespace()) s--
        val word = buf.substring(s, dot)
        return DOTTED.matches(word) || word.lowercase() in TITLES
    }

    private companion object {
        val DOTTED = Regex("""(?:[A-Za-z]\.)+[A-Za-z]""")
        val TITLES = setOf("dr", "mr", "mrs", "ms", "st", "vs", "prof", "approx")
    }

    private fun words(from: Int, to: Int): Int {
        var n = 0
        var inWord = false
        for (k in from until to) {
            val w = !buf[k].isWhitespace()
            if (w && !inWord) n++
            inWord = w
        }
        return n
    }
}

/**
 * Streams LLM tokens into (a) speakable clauses for clause-level TTS pipelining,
 * (b) grounded [ToolAction]s parsed from `[ACTION:...]` tags — which are never spoken
 * or displayed, even when a tag arrives split across several tokens — and (c) the clean
 * display text for the conversation feed.
 */
class ResponseStreamProcessor(
    private val onClause: (String) -> Unit,
    private val onAction: (ToolAction) -> Unit = {},
    private val onDisplay: (String) -> Unit = {},
    minCommaWords: Int = 5,
) {
    private val segmenter = ClauseSegmenter(minCommaWords)
    private val display = StringBuilder()
    private val tag = StringBuilder()
    private var inTag = false

    private val _actions = ArrayList<ToolAction>()
    val actions: List<ToolAction> get() = _actions
    val displayText: String get() = display.toString().trim()

    fun push(token: String) {
        val clean = StringBuilder()
        for (c in token) {
            if (inTag) {
                tag.append(c)
                if (c == ']') {
                    val parsed = ActionTagParser.parse(tag.toString())
                    if (parsed.isEmpty()) clean.append(tag)      // ordinary bracketed text
                    parsed.forEach { _actions.add(it); onAction(it) }
                    tag.setLength(0); inTag = false
                } else if (tag.length > 96) {                   // not a tag after all
                    clean.append(tag); tag.setLength(0); inTag = false
                }
            } else if (c == '[') {
                inTag = true; tag.setLength(0); tag.append(c)
            } else {
                clean.append(c)
            }
        }
        emit(clean.toString())
    }

    fun finish() {
        if (inTag) { val t = tag.toString(); tag.setLength(0); inTag = false; emit(t) }
        segmenter.flush().forEach(onClause)
    }

    private fun emit(text: String) {
        if (text.isEmpty()) return
        val wasEmpty = display.isBlank()
        display.append(text)
        if (!wasEmpty || display.isNotBlank()) onDisplay(displayText)
        segmenter.push(text).forEach(onClause)
    }
}
