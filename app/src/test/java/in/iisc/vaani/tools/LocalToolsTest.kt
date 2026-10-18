package `in`.iisc.vaani.tools

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZonedDateTime

/** Grounded tools for the four IISc scenarios: weather, reminder, alarm, schedule. */
class LocalToolsTest {

    private fun at(h: Int, m: Int) = ZonedDateTime.of(2026, 10, 18, h, m, 0, 0, IST)

    // Verbatim Open-Meteo response for Bengaluru (2026-10-18).
    private val openMeteo = """{"latitude":12.970123,"longitude":77.56364,"timezone":"Asia/Kolkata","daily_units":{"time":"iso8601"},
        "daily":{"time":["2026-10-18","2026-10-19","2026-10-20"],"temperature_2m_max":[26.1,28.4,28.2],
        "temperature_2m_min":[20.5,19.7,19.6],"precipitation_probability_max":[72,84,51],"weathercode":[80,95,51]}}"""

    @Test fun openMeteoPicksTomorrow() {
        val f = OpenMeteo.parse(openMeteo, LocalDate.of(2026, 10, 19))!!
        assertEquals(28.4, f.tMaxC, 1e-9)
        assertEquals(19.7, f.tMinC, 1e-9)
        assertEquals(84, f.precipProbPct)
        assertEquals(WeatherCodes.Kind.STORM, f.kind)
        assertEquals("thunderstorms", f.descriptionEn)
        assertEquals(DailyForecast.Source.LIVE, f.source)
        assertNull(OpenMeteo.parse(openMeteo, LocalDate.of(2026, 12, 1)))
        assertNull(OpenMeteo.parse("not json", LocalDate.of(2026, 10, 19)))
    }

    @Test fun scheduleAlwaysHasANextMeeting() {
        for (h in 0..23) for (m in listOf(0, 29, 45)) {
            val now = LocalTime.of(h, m)
            val ev = ScheduleGenerator.forDay(now)
            assertEquals(3, ev.size)
            if (now.isBefore(LocalTime.of(23, 30))) assertNotNull("no next meeting at $now", ScheduleGenerator.next(ev, now))
            assertTrue(ev.zipWithNext().all { (a, b) -> a.start.isBefore(b.start) })
        }
        val day = ScheduleGenerator.forDay(LocalTime.of(10, 0))
        assertEquals(LocalTime.of(11, 0), day[0].start)
        assertEquals("IISc Computational Intelligence Lab Review", day[1].title)
        val status = ScheduleGenerator.withStatus(day, LocalTime.of(12, 0)).map { it.second }
        assertEquals(listOf(EventStatus.DONE, EventStatus.NEXT, EventStatus.UPCOMING), status)
    }

    @Test fun parsesEnglishAndHinglishTimes() {
        assertEquals(17 to 0, LocalIntents.parseTime("Can you set a reminder for my meeting at 5 PM?", false))
        assertEquals(17 to 0, LocalIntents.parseTime("meeting at 5 p.m.", false))
        assertEquals(16 to 30, LocalIntents.parseTime("lab review at 4:30 pm", false))
        assertEquals(7 to 0, LocalIntents.parseTime("Mujhe kal subah 7 baje ka alarm set karna hai.", true))
        assertEquals(7 to 0, LocalIntents.parseTime("muji kalsuba, 7-7 baji ka alarm is 8 karna hai", true))
        assertEquals(19 to 0, LocalIntents.parseTime("shaam saat baje yaad dilana", false))
        assertEquals(17 to 0, LocalIntents.parseTime("set it for 17:00", false))
        assertNull(LocalIntents.parseTime("what is the weather", false))
    }

    @Test fun detectsTheFourScenarios() {
        assertEquals(setOf(AgentIntent.WEATHER), LocalIntents.detect("What is the weather forecast for Bangalore tomorrow?"))
        assertEquals(setOf(AgentIntent.REMINDER), LocalIntents.detect("Can you set a reminder for my meeting at 5 PM?"))
        assertEquals(setOf(AgentIntent.ALARM), LocalIntents.detect("Mujhe kal subah 7 baje ka alarm set karna hai."))
        assertEquals(setOf(AgentIntent.SCHEDULE), LocalIntents.detect("Aaj ka schedule kya hai, aur meri next meeting kab hai?"))
        assertTrue(LocalIntents.mentionsTomorrow("muji kalsuba 7 baje"))
    }

    @Test fun alarmDateRollsToTomorrow() {
        assertEquals(LocalDate.of(2026, 10, 19), nextOccurrence(at(15, 0), 7, 0, saidTomorrow = false))
        assertEquals(LocalDate.of(2026, 10, 19), nextOccurrence(at(5, 0), 7, 0, saidTomorrow = true))
        assertEquals(LocalDate.of(2026, 10, 18), nextOccurrence(at(15, 0), 17, 0, saidTomorrow = false))
    }

    private fun snapshot(now: ZonedDateTime) = ToolSnapshot(
        now, OpenMeteo.parse(openMeteo, now.toLocalDate().plusDays(1))!!,
        ScheduleGenerator.forDay(now.toLocalTime()), emptyList(), emptyList(),
    )

    @Test fun offlineResponderAnswersInTheUsersLanguage() {
        val s = snapshot(at(15, 42))
        val w = LocalResponder.respond("What is the weather forecast for Bangalore tomorrow?", "en", s)
        assertTrue(w.text, w.text.contains("28°C") && w.text.contains("thunderstorms"))
        val r = LocalResponder.respond("Can you set a reminder for my meeting at 5 PM?", "en", s)
        assertEquals(17, r.actions.single().hour)
        val a = LocalResponder.respond("Mujhe kal subah 7 baje ka alarm set karna hai.", "hi", s)
        assertEquals(ActionType.SET_ALARM, a.actions.single().type)
        assertTrue(a.text, a.text.startsWith("Theek hai") && a.text.contains("7:00 AM"))
        val sc = LocalResponder.respond("Aaj ka schedule kya hai, aur meri next meeting kab hai?", "hi", s)
        assertTrue(sc.text, sc.text.contains("4:30 PM") && sc.text.contains("IISc Computational Intelligence Lab Review"))
    }

    @Test fun contextBlockIsGrounded() {
        val ctx = ToolContext.render(snapshot(at(15, 42)))
        assertTrue(ctx, ctx.contains("Sunday, 18 Oct 2026, 3:42 PM IST"))
        assertTrue(ctx, ctx.contains("thunderstorms, 19.7–28.4°C, 84% chance of rain [Open-Meteo live]"))
        assertTrue(ctx, ctx.contains("4:30 PM IISc Computational Intelligence Lab Review (NEXT, in 48 min)"))
        assertTrue(ctx, ctx.contains("Active alarms: none"))
    }
}
