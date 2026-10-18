package `in`.iisc.vaani.tools

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.provider.AlarmClock
import android.provider.CalendarContract
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.time.ZonedDateTime
import java.util.concurrent.atomic.AtomicLong

/**
 * Grounded local tools for the four IISc evaluation scenarios.
 *
 *  1. Weather — tomorrow's Bengaluru forecast is fetched from Open-Meteo once at launch
 *     and cached in memory, so a weather question costs **0 ms of tool latency** during
 *     the voice turn; a realistic Bengaluru estimate stands in if the phone is offline.
 *  2. Reminders — `[ACTION:SET_REMINDER|17:00|Meeting]` becomes a structured reminder card
 *     with a one-tap Android Calendar insert intent.
 *  3. Alarms — `[ACTION:SET_ALARM|07:00|Morning Alarm]` becomes an alarm card that can be
 *     committed to the system Clock app via `AlarmClock.ACTION_SET_ALARM`.
 *  4. Schedule — a realistic IISc Speech Lab day, anchored to the current time so there is
 *     always a "next meeting" ([ScheduleGenerator]).
 *
 * [snapshot] is what the LLM prompt and the action cards are rendered from.
 */
class LocalToolManager(
    private val context: Context,
    private val http: OkHttpClient,
    private val clock: () -> ZonedDateTime = { ZonedDateTime.now(IST) },
) {
    private val ids = AtomicLong(1)
    private val _forecast = MutableStateFlow(DailyForecast.bengaluruFallback(clock().toLocalDate().plusDays(1)))
    private val _alarms = MutableStateFlow<List<AlarmItem>>(emptyList())
    private val _reminders = MutableStateFlow<List<ReminderItem>>(emptyList())

    val forecast: StateFlow<DailyForecast> = _forecast.asStateFlow()
    val alarms: StateFlow<List<AlarmItem>> = _alarms.asStateFlow()
    val reminders: StateFlow<List<ReminderItem>> = _reminders.asStateFlow()

    fun now(): ZonedDateTime = clock()

    fun schedule(): List<ScheduleEvent> = ScheduleGenerator.forDay(clock().toLocalTime())

    fun snapshot(): ToolSnapshot =
        ToolSnapshot(clock(), _forecast.value, schedule(), _alarms.value, _reminders.value)

    /** Fetches tomorrow's live forecast (call once at launch, off the main thread). */
    suspend fun refreshWeather(): DailyForecast = withContext(Dispatchers.IO) {
        val now = clock()
        val tomorrow = now.toLocalDate().plusDays(1)
        val live = runCatching {
            http.newCall(Request.Builder().url(OpenMeteo.URL).header("User-Agent", "VaaniEdge/2.0").build())
                .execute().use { resp ->
                    if (!resp.isSuccessful) null else OpenMeteo.parse(resp.body?.string() ?: "", tomorrow, now)
                }
        }.getOrNull()
        val f = live ?: DailyForecast.bengaluruFallback(tomorrow)
        _forecast.value = f
        f
    }

    /** Applies a tool action and returns the created item (alarm or reminder). */
    fun apply(action: ToolAction, transcript: String): Any {
        val now = clock()
        val date = nextOccurrence(now, action.hour, action.minute, LocalIntents.mentionsTomorrow(transcript))
        return when (action.type) {
            ActionType.SET_ALARM -> {
                val existing = _alarms.value.firstOrNull { it.hour == action.hour && it.minute == action.minute && it.date == date }
                existing ?: AlarmItem(ids.getAndIncrement(), action.hour, action.minute, action.label, date)
                    .also { a -> _alarms.update { it + a } }
            }
            ActionType.SET_REMINDER -> {
                val existing = _reminders.value.firstOrNull { it.hour == action.hour && it.minute == action.minute && it.date == date }
                existing ?: ReminderItem(ids.getAndIncrement(), action.hour, action.minute, action.label, date)
                    .also { r -> _reminders.update { it + r } }
            }
        }
    }

    /** One-tap hand-off to the system Clock app (AlarmClock.ACTION_SET_ALARM). */
    fun commitAlarmToClock(alarm: AlarmItem): Boolean {
        val intent = Intent(AlarmClock.ACTION_SET_ALARM)
            .putExtra(AlarmClock.EXTRA_HOUR, alarm.hour)
            .putExtra(AlarmClock.EXTRA_MINUTES, alarm.minute)
            .putExtra(AlarmClock.EXTRA_MESSAGE, alarm.label)
            .putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return launch(intent).also { ok ->
            if (ok) _alarms.update { list -> list.map { if (it.id == alarm.id) it.copy(handedToClock = true) else it } }
        }
    }

    /** One-tap hand-off to Android Calendar (insert event, 30 min, with the reminder title). */
    fun addReminderToCalendar(reminder: ReminderItem): Boolean {
        val start = reminder.date.atTime(reminder.hour, reminder.minute).atZone(IST).toInstant().toEpochMilli()
        val intent = Intent(Intent.ACTION_INSERT)
            .setData(CalendarContract.Events.CONTENT_URI)
            .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, start)
            .putExtra(CalendarContract.EXTRA_EVENT_END_TIME, start + 30 * 60_000L)
            .putExtra(CalendarContract.Events.TITLE, reminder.label)
            .putExtra(CalendarContract.Events.DESCRIPTION, "Created by the VaaniEdge voice agent")
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return launch(intent).also { ok ->
            if (ok) _reminders.update { list -> list.map { if (it.id == reminder.id) it.copy(addedToCalendar = true) else it } }
        }
    }

    private fun launch(intent: Intent): Boolean = try {
        context.startActivity(intent); true
    } catch (e: ActivityNotFoundException) {
        false
    } catch (e: SecurityException) {
        false
    }
}
