package `in`.iisc.vaani.llm

import `in`.iisc.vaani.tools.ActionTagParser
import `in`.iisc.vaani.tools.ActionType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Clause-level TTS pipelining + tool-tag handling on realistic streamed token splits. */
class ResponseStreamProcessorTest {

    private fun run(tokens: List<String>): Triple<List<String>, ResponseStreamProcessor, List<String>> {
        val clauses = ArrayList<String>()
        val displays = ArrayList<String>()
        val p = ResponseStreamProcessor(onClause = { clauses.add(it) }, onDisplay = { displays.add(it) })
        tokens.forEach { p.push(it) }
        p.finish()
        return Triple(clauses, p, displays)
    }

    @Test fun firstClauseIsReleasedBeforeTheStreamEnds() {
        val clauses = ArrayList<String>()
        val p = ResponseStreamProcessor(onClause = { clauses.add(it) })
        p.push("Tomorrow in Bangalore expect thunderstorms with a high of 28°C.")
        assertTrue("clause must wait for the char after '.'", clauses.isEmpty())
        p.push(" Carry an")
        assertEquals(listOf("Tomorrow in Bangalore expect thunderstorms with a high of 28°C."), clauses)
        p.push(" umbrella!")
        assertEquals(2, clauses.size)
        p.finish()
        assertEquals("Carry an umbrella!", clauses[1])
    }

    @Test fun decimalsTimesAndAbbreviationsDoNotSplit() {
        val (clauses, _, _) = run(listOf("Low of 19", ".7°C at 4", ".30 p.m. tonight", ". Done"))
        assertEquals(listOf("Low of 19.7°C at 4.30 p.m. tonight.", "Done"), clauses)
    }

    @Test fun commaCutsOnlyAfterFiveWords() {
        val (clauses, _, _) = run(listOf("Kal Bangalore mein, ", "baarish ki sambhavna kaafi zyada hai, ", "chhata le jaana."))
        assertEquals(listOf("Kal Bangalore mein, baarish ki sambhavna kaafi zyada hai,", "chhata le jaana."), clauses)
    }

    @Test fun devanagariDandaIsABoundary() {
        val (clauses, _, _) = run(listOf("ठीक है। अलार्म सेट हो गया"))
        assertEquals(listOf("ठीक है।", "अलार्म सेट हो गया"), clauses)
    }

    @Test fun actionTagSplitAcrossTokensIsParsedAndNeverSpoken() {
        val (clauses, p, displays) = run(listOf("[ACT", "ION:SET_ALA", "RM|07:", "00|Morning Alarm]\n", "Aapka alarm kal subah ", "7:00 AM ke liye set ho gaya hai."))
        assertEquals(1, p.actions.size)
        val a = p.actions[0]
        assertEquals(ActionType.SET_ALARM, a.type)
        assertEquals(7, a.hour); assertEquals(0, a.minute); assertEquals("Morning Alarm", a.label)
        assertEquals(listOf("Aapka alarm kal subah 7:00 AM ke liye set ho gaya hai."), clauses)
        assertTrue(clauses.none { "ACTION" in it })
        assertTrue(displays.none { "ACTION" in it })
        assertEquals("Aapka alarm kal subah 7:00 AM ke liye set ho gaya hai.", p.displayText)
    }

    @Test fun reminderTag() {
        val (_, p, _) = run(listOf("[ACTION:SET_REMINDER|17:00|Meeting]\nDone, I've set a reminder for your meeting at 5:00 PM."))
        assertEquals(ActionType.SET_REMINDER, p.actions.single().type)
        assertEquals(17, p.actions.single().hour)
    }

    @Test fun ordinaryBracketsSurvive() {
        val (clauses, p, _) = run(listOf("Next is the review [lab 2] at 4:30 PM."))
        assertTrue(p.actions.isEmpty())
        assertEquals(listOf("Next is the review [lab 2] at 4:30 PM."), clauses)
    }

    @Test fun tagParserValidatesRanges() {
        assertTrue(ActionTagParser.parse("[ACTION:SET_ALARM|25:00|x]").isEmpty())
        assertEquals(1, ActionTagParser.parse("[action:set_alarm|7.30|]").size)
        assertEquals("Alarm", ActionTagParser.parse("[ACTION:SET_ALARM|07:30]").single().label)
        assertEquals("Done.", ActionTagParser.strip("[ACTION:SET_REMINDER|17:00|Meeting] Done."))
    }

    @Test fun runawayClauseIsCut() {
        val long = (1..40).joinToString(" ") { "word$it" }
        val clauses = ArrayList<String>()
        val p = ResponseStreamProcessor(onClause = { clauses.add(it) })
        p.push(long)
        assertTrue("a 40-word clause must not block TTS", clauses.isNotEmpty())
    }
}
