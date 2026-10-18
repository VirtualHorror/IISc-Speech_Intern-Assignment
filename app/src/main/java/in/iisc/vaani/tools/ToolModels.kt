package `in`.iisc.vaani.tools

import org.json.JSONObject
import java.time.LocalDate
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlin.math.roundToInt

val IST: ZoneId = ZoneId.of("Asia/Kolkata")

private val TIME_12H: DateTimeFormatter = DateTimeFormatter.ofPattern("h:mm a", Locale.US)

/** "7:00 AM" style clock string. */
fun clock12(hour: Int, minute: Int): String = LocalTime.of(hour, minute).format(TIME_12H)

// =============================================================================
//  Weather (Open-Meteo, Bengaluru 12.9716 N, 77.5946 E)
// =============================================================================

/** Tomorrow's Bengaluru forecast, cached in memory for 0 ms lookup during a voice turn. */
data class DailyForecast(
    val date: LocalDate,
    val tMaxC: Double,
    val tMinC: Double,
    val precipProbPct: Int,
    val weatherCode: Int,
    val source: Source,
    val fetchedAt: ZonedDateTime? = null,
) {
    enum class Source(val label: String) { LIVE("Open-Meteo live"), FALLBACK("offline estimate") }

    val descriptionEn: String get() = WeatherCodes.en(weatherCode)
    val descriptionHi: String get() = WeatherCodes.hi(weatherCode)
    val kind: WeatherCodes.Kind get() = WeatherCodes.kind(weatherCode)
    val maxRounded: Int get() = tMaxC.roundToInt()
    val minRounded: Int get() = tMinC.roundToInt()

    companion object {
        /** Realistic post-monsoon Bengaluru day, used until (or if) the live fetch succeeds. */
        fun bengaluruFallback(tomorrow: LocalDate) =
            DailyForecast(tomorrow, 27.5, 19.5, 55, 80, Source.FALLBACK)
    }
}

/** WMO weather interpretation codes (Open-Meteo `weathercode`). */
object WeatherCodes {
    enum class Kind { CLEAR, CLOUD, FOG, RAIN, STORM, SNOW }

    fun kind(code: Int): Kind = when (code) {
        0, 1 -> Kind.CLEAR
        2, 3 -> Kind.CLOUD
        45, 48 -> Kind.FOG
        in 51..67, in 80..82 -> Kind.RAIN
        in 71..77, 85, 86 -> Kind.SNOW
        in 95..99 -> Kind.STORM
        else -> Kind.CLOUD
    }

    fun en(code: Int): String = when (code) {
        0 -> "clear skies"; 1 -> "mainly clear skies"; 2 -> "partly cloudy skies"; 3 -> "overcast skies"
        45, 48 -> "fog"; 51, 53, 55 -> "drizzle"; 56, 57 -> "freezing drizzle"
        61 -> "light rain"; 63 -> "moderate rain"; 65 -> "heavy rain"; 66, 67 -> "freezing rain"
        71, 73, 75, 77 -> "snow"; 80 -> "light rain showers"; 81 -> "rain showers"; 82 -> "heavy rain showers"
        85, 86 -> "snow showers"; 95 -> "thunderstorms"; 96, 99 -> "thunderstorms with hail"
        else -> "mixed weather"
    }

    fun hi(code: Int): String = when (code) {
        0, 1 -> "saaf aasman"; 2 -> "halke badal"; 3 -> "ghane badal"; 45, 48 -> "kohra"
        in 51..57 -> "boonda-baandi"; 61 -> "halki baarish"; 63 -> "baarish"; 65 -> "tez baarish"
        66, 67 -> "thandi baarish"; 80, 81 -> "baarish ki bauchhaar"; 82 -> "tez bauchhaar"
        in 95..99 -> "garaj ke saath baarish"
        else -> "mila-jula mausam"
    }
}

object OpenMeteo {
    const val LAT = 12.9716
    const val LON = 77.5946
    const val URL = "https://api.open-meteo.com/v1/forecast?latitude=12.9716&longitude=77.5946" +
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode" +
        "&timezone=Asia%2FKolkata"

    /** Picks [day] from an Open-Meteo `daily` response (null if absent / malformed). */
    fun parse(json: String, day: LocalDate, fetchedAt: ZonedDateTime? = null): DailyForecast? = runCatching {
        val daily = JSONObject(json).getJSONObject("daily")
        val dates = daily.getJSONArray("time")
        val iso = day.toString()
        val idx = (0 until dates.length()).firstOrNull { dates.getString(it) == iso } ?: return null
        DailyForecast(
            date = day,
            tMaxC = daily.getJSONArray("temperature_2m_max").getDouble(idx),
            tMinC = daily.getJSONArray("temperature_2m_min").getDouble(idx),
            precipProbPct = daily.getJSONArray("precipitation_probability_max").optInt(idx, 0),
            weatherCode = daily.getJSONArray("weathercode").optInt(idx, 3),
            source = DailyForecast.Source.LIVE,
            fetchedAt = fetchedAt,
        )
    }.getOrNull()
}

// =============================================================================
//  Today's IISc Speech Lab schedule (anchored to "now" so a next meeting exists)
// =============================================================================

data class ScheduleEvent(val title: String, val start: LocalTime, val durationMin: Int) {
    val end: LocalTime get() = start.plusMinutes(durationMin.toLong())
    val startLabel: String get() = clock12(start.hour, start.minute)
}

enum class EventStatus { DONE, ONGOING, NEXT, UPCOMING }

object ScheduleGenerator {
    val TITLES = listOf(
        "INT8 Speech Quantization Sync",
        "IISc Computational Intelligence Lab Review",
        "Edge DSP Telemetry Benchmark",
    )
    private val DEFAULT_STARTS = listOf(LocalTime.of(11, 0), LocalTime.of(16, 30), LocalTime.of(18, 0))
    private val DURATIONS = listOf(30, 45, 30)

    /**
     * The canonical day (11:00 AM / 4:30 PM / 6:00 PM). Once all three are behind us, the
     * day is re-anchored so the last meeting is ~30 min ahead: there is always a clear
     * "next meeting" to answer with, whatever time the demo is recorded.
     */
    fun forDay(now: LocalTime): List<ScheduleEvent> {
        val starts = if (DEFAULT_STARTS.any { it.isAfter(now.plusMinutes(10)) }) DEFAULT_STARTS else {
            val minutesNow = now.hour * 60 + now.minute
            val next = minOf(23 * 60 + 45, ((minutesNow + 25 + 14) / 15) * 15)
            val nextT = if (next <= minutesNow) LocalTime.of(23, 59) else LocalTime.of(next / 60, next % 60)
            val second = maxOf(9 * 60, (nextT.hour * 60 + nextT.minute) - 90)
            val first = maxOf(8 * 60 + 30, second - 300)
            listOf(LocalTime.of(first / 60, first % 60), LocalTime.of(second / 60, second % 60), nextT)
        }
        return TITLES.indices.map { ScheduleEvent(TITLES[it], starts[it], DURATIONS[it]) }
    }

    fun withStatus(events: List<ScheduleEvent>, now: LocalTime): List<Pair<ScheduleEvent, EventStatus>> {
        var nextAssigned = false
        return events.map { e ->
            val st = when {
                !e.end.isAfter(now) -> EventStatus.DONE
                !e.start.isAfter(now) -> EventStatus.ONGOING
                !nextAssigned -> { nextAssigned = true; EventStatus.NEXT }
                else -> EventStatus.UPCOMING
            }
            e to st
        }
    }

    fun next(events: List<ScheduleEvent>, now: LocalTime): ScheduleEvent? = events.firstOrNull { it.start.isAfter(now) }
}

// =============================================================================
//  Alarms & reminders
// =============================================================================

data class AlarmItem(
    val id: Long, val hour: Int, val minute: Int, val label: String, val date: LocalDate,
    val handedToClock: Boolean = false,
) {
    val timeLabel: String get() = clock12(hour, minute)
    fun dayLabel(today: LocalDate): String = dayName(date, today)
}

data class ReminderItem(
    val id: Long, val hour: Int, val minute: Int, val label: String, val date: LocalDate,
    val addedToCalendar: Boolean = false,
) {
    val timeLabel: String get() = clock12(hour, minute)
    fun dayLabel(today: LocalDate): String = dayName(date, today)
}

fun dayName(date: LocalDate, today: LocalDate): String = when (date) {
    today -> "Today"
    today.plusDays(1) -> "Tomorrow"
    else -> date.dayOfWeek.name.lowercase().replaceFirstChar { it.uppercase() }
}

/** Date of the next occurrence of HH:MM (tomorrow if already past, or if the user said "tomorrow"). */
fun nextOccurrence(now: ZonedDateTime, hour: Int, minute: Int, saidTomorrow: Boolean): LocalDate {
    val today = now.toLocalDate()
    val atToday = now.withHour(hour).withMinute(minute).withSecond(0).withNano(0)
    return if (saidTomorrow || !atToday.isAfter(now)) today.plusDays(1) else today
}
