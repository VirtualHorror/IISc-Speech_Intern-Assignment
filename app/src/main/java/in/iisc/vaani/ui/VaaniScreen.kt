package `in`.iisc.vaani.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import `in`.iisc.vaani.agent.ActionCard
import `in`.iisc.vaani.agent.AgentState
import `in`.iisc.vaani.agent.AgentUiState
import `in`.iisc.vaani.agent.BENCHMARK_PROMPTS
import `in`.iisc.vaani.agent.BenchmarkPrompt
import `in`.iisc.vaani.agent.FeedItem
import `in`.iisc.vaani.agent.Telemetry
import `in`.iisc.vaani.agent.VoiceAgentViewModel
import `in`.iisc.vaani.tools.DailyForecast
import `in`.iisc.vaani.tools.EventStatus
import kotlinx.coroutines.flow.StateFlow
import java.time.LocalDate
import java.time.ZonedDateTime
import kotlin.math.ln

@Composable
fun VaaniScreen(vm: VoiceAgentViewModel) {
    val state by vm.ui.collectAsStateWithLifecycle()
    var showSettings by rememberSaveable { mutableStateOf(false) }

    Surface(color = Vaani.NightInk, modifier = Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing)) {
            TopBar(state, onLanguage = vm::cycleLanguageMode, onModel = { showSettings = true })
            TelemetryHud(state)
            BenchmarkStrip(state, onSpeak = vm::speakBenchmark, onLoop = vm::runLoopback)
            ConversationFeed(
                state = state,
                modifier = Modifier.weight(1f),
                onReplay = vm::replay,
                onCommitAlarm = vm::commitAlarm,
                onAddReminder = vm::addReminderToCalendar,
            )
            VoiceBar(state, vm.level, onMic = vm::onMicTap)
        }
    }
    if (showSettings) {
        SettingsDialog(
            state = state,
            keyStatus = vm.keyStatus(),
            onDismiss = { showSettings = false },
            onSave = { model, groq, gemini, mode, cloud, override ->
                vm.saveSettings(model, groq, gemini, mode, cloud, override)
                showSettings = false
            },
        )
    }
}

// =============================================================================
//  Top bar + status line
// =============================================================================

@Composable
private fun TopBar(state: AgentUiState, onLanguage: () -> Unit, onModel: () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(start = 16.dp, end = 12.dp, top = 10.dp, bottom = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                // One line on a 1080 px (~384 dp) phone: never wraps to "VaaniEd" / "ge".
                Text("VaaniEdge", style = MaterialTheme.typography.headlineSmall, color = Vaani.Paper,
                    maxLines = 1, softWrap = false)
                Text("IISc Speech Lab voice agent", style = MaterialTheme.typography.bodySmall, color = Vaani.Fog,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            Spacer(Modifier.width(4.dp))
            Pill(state.languageMode.label, Vaani.Saffron, onClick = onLanguage, description = "Language mode, tap to change")
            Spacer(Modifier.width(4.dp))
            Pill(
                if (state.cloudEnabled && state.cloudReady) state.model.label else "On-device replies",
                Vaani.Cloud, onClick = onModel, description = "LLM and API keys",
                trailing = { GlyphIcon(Glyph.SETTINGS, Vaani.Fog, 14.dp) },
                modifier = Modifier.widthIn(max = 132.dp),
            )
        }
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            val dot by animateColorAsState(
                when (state.state) {
                    AgentState.LOADING -> Vaani.Fog
                    AgentState.ERROR -> Vaani.Kumkum
                    AgentState.LISTENING -> Vaani.Saffron
                    AgentState.TRANSCRIBING, AgentState.STREAMING, AgentState.SPEAKING -> Vaani.Phosphor
                    AgentState.IDLE -> Vaani.Leaf
                }, label = "status",
            )
            Box(Modifier.size(8.dp).clip(CircleShape).background(dot))
            Spacer(Modifier.width(8.dp))
            Text(
                state.statusDetail,
                style = MaterialTheme.typography.labelMedium,
                color = if (state.state == AgentState.ERROR) Vaani.Kumkum else Vaani.Paper,
                maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
            if (state.runtimeSummary.isNotEmpty()) {
                Spacer(Modifier.width(8.dp))
                Text(state.runtimeSummary, style = MaterialTheme.typography.labelSmall, color = Vaani.Fog,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

@Composable
private fun Pill(
    text: String, accent: Color, onClick: () -> Unit, description: String,
    trailing: (@Composable () -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier
            .clip(RoundedCornerShape(50))
            .border(1.dp, accent.copy(alpha = 0.55f), RoundedCornerShape(50))
            .clickable(onClick = onClick)
            .semantics { contentDescription = description }
            .padding(horizontal = 8.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text, style = MaterialTheme.typography.labelMedium, color = accent, maxLines = 1,
            overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
        if (trailing != null) { Spacer(Modifier.width(4.dp)); trailing() }
    }
}

// =============================================================================
//  Research telemetry HUD
// =============================================================================

@Composable
private fun TelemetryHud(state: AgentUiState) {
    val t = state.telemetry
    Surface(
        color = Vaani.Slate,
        shape = MaterialTheme.shapes.medium,
        border = BorderStroke(1.dp, Vaani.Steel),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        Column(Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
            HudRow("STT (On-device)", t.sttMs?.let { "$it ms • RTF ${"%.2f".format(t.sttRtf ?: 0.0)}" }, Vaani.Saffron)
            HudRow("LLM TTFT", t.llmTtftMs?.let { "$it ms" + if (t.connectionReused == true) "  warm" else "" }, Vaani.Cloud)
            HudRow("TTS First-Audio", t.ttsFirstAudioMs?.let { "$it ms" }, Vaani.Phosphor)
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text("Total E2E Latency", style = MaterialTheme.typography.titleSmall, color = Vaani.Paper,
                    modifier = Modifier.weight(1f))
                Text(
                    t.e2eLatencyMs?.toString() ?: "—",
                    style = Figures.copy(fontSize = 34.sp, fontWeight = FontWeight.SemiBold, lineHeight = 36.sp),
                    color = Vaani.Paper,
                )
                Text(" ms", style = MaterialTheme.typography.titleSmall, color = Vaani.Fog, modifier = Modifier.padding(bottom = 4.dp))
            }
            Spacer(Modifier.height(6.dp))
            LatencyWaterfall(t)
            Spacer(Modifier.height(6.dp))
            Text(hudFootnote(state), style = MaterialTheme.typography.labelSmall, color = Vaani.Fog,
                maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

private fun hudFootnote(state: AgentUiState): String {
    val t = state.telemetry
    val parts = ArrayList<String>()
    t.lid?.let { parts.add("LID ${if (it == "hi") "HI" else "EN"} p(hi)=${"%.2f".format(t.pHindi ?: 0f)}") }
    t.tokensPerSec?.let { if (it > 0) parts.add("KV decode ${"%.0f".format(it)} tok/s") }
    t.encoderMs?.let { parts.add("encoder $it ms") }
    state.prewarm?.let { parts.add(it) }
    if (parts.isEmpty()) return "Speak or run a loopback prompt to measure a turn."
    return parts.joinToString("   ")
}

@Composable
private fun HudRow(label: String, value: String?, accent: Color) {
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(width = 3.dp, height = 14.dp).background(accent, RoundedCornerShape(2.dp)))
        Spacer(Modifier.width(8.dp))
        Text(label, style = MaterialTheme.typography.bodyMedium, color = Vaani.Fog, modifier = Modifier.weight(1f))
        Text(value ?: "—", style = Figures.copy(fontSize = 15.sp, fontWeight = FontWeight.Medium), color = Vaani.Paper)
    }
}

/** Where the end-to-end time went: on-device STT, cloud LLM, on-device TTS (proportional). */
@Composable
private fun LatencyWaterfall(t: Telemetry) {
    val stt = (t.sttMs ?: 0L).toFloat()
    val llm = (t.llmTtftMs ?: 0L).toFloat()
    val tts = (t.ttsFirstAudioMs ?: 0L).toFloat()
    val total = maxOf(stt + llm + tts, (t.e2eLatencyMs ?: 0L).toFloat(), 1f)
    val a by animateFloatAsState(stt / total, label = "stt")
    val b by animateFloatAsState(llm / total, label = "llm")
    val c by animateFloatAsState(tts / total, label = "tts")
    Canvas(Modifier.fillMaxWidth().height(8.dp).semantics { contentDescription = "Latency breakdown" }) {
        val w = size.width
        val h = size.height
        drawRoundRect(Vaani.Steel, size = Size(w, h), cornerRadius = CornerRadius(h / 2))
        var x = 0f
        for ((frac, color) in listOf(a to Vaani.Saffron, b to Vaani.Cloud, c to Vaani.Phosphor)) {
            val seg = frac * w
            if (seg > 1f) drawRoundRect(color, Offset(x, 0f), Size(seg - 2f, h), CornerRadius(h / 2))
            x += seg
        }
    }
}

// =============================================================================
//  One-tap IISc benchmark strip
// =============================================================================

@Composable
private fun BenchmarkStrip(state: AgentUiState, onSpeak: (Int) -> Unit, onLoop: (Int) -> Unit) {
    LazyRow(
        contentPadding = PaddingValues(horizontal = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
    ) {
        items(BENCHMARK_PROMPTS, key = { it.index }) { p ->
            PromptChip(
                prompt = p,
                highlighted = state.highlightedPrompt == p.index,
                busy = state.busyPrompt == p.index,
                enabled = state.runtimeReady,
                onSpeak = { onSpeak(p.index) },
                onLoop = { onLoop(p.index) },
            )
        }
    }
}

@Composable
private fun PromptChip(
    prompt: BenchmarkPrompt, highlighted: Boolean, busy: Boolean, enabled: Boolean,
    onSpeak: () -> Unit, onLoop: () -> Unit,
) {
    val border by animateColorAsState(if (highlighted) Vaani.Saffron else Vaani.Steel, label = "chip")
    Column(
        Modifier
            .width(232.dp)
            .clip(MaterialTheme.shapes.small)
            .background(if (highlighted) Vaani.SlateHigh else Vaani.Slate)
            .border(1.dp, border, MaterialTheme.shapes.small)
            .padding(start = 12.dp, end = 6.dp, top = 10.dp, bottom = 6.dp),
    ) {
        Text(
            prompt.text, style = MaterialTheme.typography.bodyMedium, color = Vaani.Paper,
            maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.height(40.dp),
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            LangBadge(prompt.lang)
            Spacer(Modifier.weight(1f))
            ChipAction(Glyph.MIC, "Speak", Vaani.Saffron, enabled, onSpeak)
            ChipAction(Glyph.LOOP, if (busy) "Running" else "Loopback", Vaani.Phosphor, enabled && !busy, onLoop)
        }
    }
}

@Composable
private fun ChipAction(glyph: Glyph, label: String, tint: Color, enabled: Boolean, onClick: () -> Unit) {
    Row(
        Modifier
            .clip(RoundedCornerShape(8.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .semantics { contentDescription = label }
            .padding(horizontal = 8.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        GlyphIcon(glyph, if (enabled) tint else Vaani.Steel, 16.dp)
        Spacer(Modifier.width(4.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = if (enabled) tint else Vaani.Steel)
    }
}

@Composable
private fun LangBadge(lang: String) {
    val hi = lang == "hi"
    Text(
        if (hi) "HI-EN" else "EN",
        style = MaterialTheme.typography.labelSmall,
        color = if (hi) Vaani.NightInk else Vaani.Fog,
        modifier = Modifier
            .clip(RoundedCornerShape(5.dp))
            .then(if (hi) Modifier.background(Vaani.Saffron) else Modifier.border(1.dp, Vaani.Steel, RoundedCornerShape(5.dp)))
            .padding(horizontal = 6.dp, vertical = 2.dp),
    )
}

// =============================================================================
//  Conversation + grounded action cards
// =============================================================================

@Composable
private fun ConversationFeed(
    state: AgentUiState, modifier: Modifier,
    onReplay: (Long) -> Unit, onCommitAlarm: (Long) -> Unit, onAddReminder: (Long) -> Unit,
) {
    val listState = rememberLazyListState()
    val last = state.feed.lastOrNull()
    val growth = (last as? FeedItem.Assistant)?.text?.length?.div(48) ?: 0
    LaunchedEffect(state.feed.size, growth) {
        if (state.feed.isNotEmpty()) listState.animateScrollToItem(state.feed.lastIndex)
    }
    if (state.feed.isEmpty()) {
        Box(modifier.fillMaxWidth().padding(24.dp), contentAlignment = Alignment.Center) {
            Text(
                if (state.runtimeReady) "Tap the mic and ask about tomorrow's weather, an alarm, a reminder or today's schedule. " +
                    "Loopback runs a prompt through the voice, recogniser and reply without you speaking."
                else "Preparing the on-device speech models. This takes a few seconds on first launch.",
                style = MaterialTheme.typography.bodyMedium, color = Vaani.Fog,
            )
        }
        return
    }
    LazyColumn(
        state = listState,
        modifier = modifier.fillMaxWidth(),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items(state.feed, key = { it.id }) { item ->
            when (item) {
                is FeedItem.User -> UserBubble(item)
                is FeedItem.Assistant -> AssistantBubble(item, onReplay)
                is FeedItem.Card -> CardView(item.card, state, onCommitAlarm, onAddReminder)
                is FeedItem.Notice -> Text(item.text, style = MaterialTheme.typography.bodySmall, color = Vaani.Fog,
                    modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp))
            }
        }
    }
}

@Composable
private fun UserBubble(item: FeedItem.User) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
        Column(
            Modifier
                .widthIn(max = 320.dp)
                .clip(RoundedCornerShape(16.dp, 16.dp, 4.dp, 16.dp))
                .background(Vaani.SlateHigh)
                .padding(horizontal = 12.dp, vertical = 9.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                LangBadge(item.lang)
                Spacer(Modifier.width(6.dp))
                Text(
                    buildString {
                        item.sttMs?.let { append("STT $it ms") }
                        item.sttModel?.takeIf { it.endsWith("-hi") }?.let { append(" · $it") }
                        if (item.source == "loopback") append("   loopback")
                        item.wer?.let { append("   WER ${"%.0f".format(it * 100)}%") }
                    },
                    style = Figures.copy(fontSize = 11.sp), color = Vaani.Fog,
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(item.text, style = MaterialTheme.typography.bodyLarge, color = Vaani.Paper)
            item.note?.let {
                Text(it, style = MaterialTheme.typography.bodySmall, color = Vaani.Saffron)
            }
            item.rawAsr?.let {
                Text("raw ASR · $it", style = MaterialTheme.typography.bodySmall.copy(fontFamily = PlexDevanagari), color = Vaani.Fog)
            }
        }
    }
}

@Composable
private fun AssistantBubble(item: FeedItem.Assistant, onReplay: (Long) -> Unit) {
    var phonetics by rememberSaveable(item.id) { mutableStateOf(false) }
    Column(
        Modifier
            .fillMaxWidth()
            .padding(end = 28.dp)
            .border(BorderStroke(1.dp, Vaani.Steel), RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
    ) {
        Text(
            item.text.ifEmpty { if (item.streaming) "…" else "" } + if (item.streaming && item.text.isNotEmpty()) " ▍" else "",
            style = MaterialTheme.typography.bodyLarge, color = Vaani.Paper,
        )
        item.note?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = Vaani.Kumkum) }
        Spacer(Modifier.height(6.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                item.engine + (item.telemetry?.let { t ->
                    listOfNotNull(
                        t.llmTtftMs?.let { "TTFT $it" }, t.ttsFirstAudioMs?.let { "audio $it" }, t.e2eLatencyMs?.let { "E2E $it ms" },
                    ).joinToString("  ", prefix = "   ")
                } ?: ""),
                style = Figures.copy(fontSize = 11.sp), color = Vaani.Fog, modifier = Modifier.weight(1f),
                maxLines = 2,
            )
            if (!item.streaming && item.acoustic.isNotEmpty()) {
                IconTextButton(if (phonetics) Glyph.CHEVRON_UP else Glyph.CHEVRON_DOWN, "Phonetics", Vaani.Fog) { phonetics = !phonetics }
                IconTextButton(Glyph.REPLAY, "Replay", Vaani.Phosphor) { onReplay(item.id) }
            }
        }
        if (phonetics) {
            Spacer(Modifier.height(6.dp))
            Text(item.acoustic, style = MaterialTheme.typography.bodyMedium.copy(fontFamily = PlexDevanagari), color = Vaani.Paper)
            Spacer(Modifier.height(4.dp))
            Text("/" + item.ipa + "/", style = MaterialTheme.typography.bodySmall, color = Vaani.Phosphor)
        }
    }
}

@Composable
private fun IconTextButton(glyph: Glyph, label: String, tint: Color, onClick: () -> Unit) {
    Row(
        Modifier.clip(RoundedCornerShape(8.dp)).clickable(onClick = onClick)
            .semantics { contentDescription = label }.padding(horizontal = 6.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        GlyphIcon(glyph, tint, 16.dp)
        Spacer(Modifier.width(3.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = tint)
    }
}

@Composable
private fun CardView(card: ActionCard, state: AgentUiState, onCommitAlarm: (Long) -> Unit, onAddReminder: (Long) -> Unit) {
    val today = LocalDate.now(`in`.iisc.vaani.tools.IST)
    when (card) {
        is ActionCard.Weather -> WeatherCard(state.forecast ?: card.forecast)
        is ActionCard.Alarm -> state.alarms.firstOrNull { it.id == card.alarmId }?.let { a ->
            ToolCard(Vaani.Saffron, Glyph.ALARM, "Alarm", a.timeLabel, "${a.dayLabel(today)}   ${a.label}",
                action = if (a.handedToClock) "Sent to Clock" else "Set in Clock",
                actionEnabled = !a.handedToClock) { onCommitAlarm(a.id) }
        }
        is ActionCard.Reminder -> state.reminders.firstOrNull { it.id == card.reminderId }?.let { r ->
            ToolCard(Vaani.Cloud, Glyph.BELL, "Reminder", r.timeLabel, "${r.dayLabel(today)}   ${r.label}",
                action = if (r.addedToCalendar) "Added to Calendar" else "Add to Calendar",
                actionEnabled = !r.addedToCalendar) { onAddReminder(r.id) }
        }
        is ActionCard.Schedule -> ScheduleCard(card)
    }
}

@Composable
private fun CardShell(accent: Color, content: @Composable () -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(IntrinsicSize.Min).clip(MaterialTheme.shapes.medium).background(Vaani.Slate)
            .border(1.dp, Vaani.Steel, MaterialTheme.shapes.medium),
    ) {
        Box(Modifier.width(4.dp).fillMaxHeight().background(accent))
        Box(Modifier.weight(1f).padding(12.dp)) { content() }
    }
}

@Composable
private fun ToolCard(
    accent: Color, glyph: Glyph, kind: String, big: String, detail: String,
    action: String, actionEnabled: Boolean, onAction: () -> Unit,
) {
    CardShell(accent) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            GlyphIcon(glyph, accent, 28.dp)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(kind, style = MaterialTheme.typography.labelMedium, color = Vaani.Fog)
                Text(big, style = Figures.copy(fontSize = 26.sp, fontWeight = FontWeight.SemiBold), color = Vaani.Paper)
                Text(detail, style = MaterialTheme.typography.bodySmall, color = Vaani.Fog)
            }
            Text(
                action, style = MaterialTheme.typography.labelLarge,
                color = if (actionEnabled) Vaani.NightInk else Vaani.Fog,
                modifier = Modifier
                    .clip(RoundedCornerShape(10.dp))
                    .background(if (actionEnabled) accent else Color.Transparent)
                    .clickable(enabled = actionEnabled, onClick = onAction)
                    .padding(horizontal = 12.dp, vertical = 8.dp),
            )
        }
    }
}

@Composable
private fun WeatherCard(f: DailyForecast) {
    CardShell(Vaani.Phosphor) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            WeatherGlyph(f.kind)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("Bengaluru tomorrow", style = MaterialTheme.typography.labelMedium, color = Vaani.Fog)
                Row(verticalAlignment = Alignment.Bottom) {
                    Text("${f.maxRounded}°", style = Figures.copy(fontSize = 28.sp, fontWeight = FontWeight.SemiBold), color = Vaani.Paper)
                    Text(" / ${f.minRounded}°C", style = Figures.copy(fontSize = 16.sp), color = Vaani.Fog,
                        modifier = Modifier.padding(bottom = 4.dp))
                }
                Text(f.descriptionEn.replaceFirstChar { it.uppercase() } + ", ${f.precipProbPct}% chance of rain",
                    style = MaterialTheme.typography.bodySmall, color = Vaani.Paper)
                Text(
                    if (f.source == DailyForecast.Source.LIVE) "Weather data by Open-Meteo.com" else "Offline estimate, live data unavailable",
                    style = MaterialTheme.typography.labelSmall, color = Vaani.Fog,
                )
            }
        }
    }
}

@Composable
private fun ScheduleCard(card: ActionCard.Schedule) {
    val now = ZonedDateTime.now(`in`.iisc.vaani.tools.IST).toLocalTime()
    CardShell(Vaani.Saffron) {
        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                GlyphIcon(Glyph.CALENDAR, Vaani.Saffron, 22.dp)
                Spacer(Modifier.width(8.dp))
                Text("Today's schedule", style = MaterialTheme.typography.titleSmall, color = Vaani.Paper)
            }
            Spacer(Modifier.height(6.dp))
            for ((e, st) in card.events) {
                Row(Modifier.padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(e.startLabel, style = Figures.copy(fontSize = 13.sp), color = if (st == EventStatus.NEXT) Vaani.Saffron else Vaani.Fog,
                        modifier = Modifier.width(76.dp))
                    Text(
                        e.title, style = MaterialTheme.typography.bodyMedium,
                        color = if (st == EventStatus.DONE) Vaani.Fog else Vaani.Paper,
                        textDecoration = if (st == EventStatus.DONE) TextDecoration.LineThrough else null,
                        modifier = Modifier.weight(1f),
                    )
                    when (st) {
                        EventStatus.NEXT -> Text("next, in ${(e.start.toSecondOfDay() - now.toSecondOfDay()).coerceAtLeast(0) / 60} min",
                            style = MaterialTheme.typography.labelSmall, color = Vaani.Saffron)
                        EventStatus.ONGOING -> Text("now", style = MaterialTheme.typography.labelSmall, color = Vaani.Phosphor)
                        else -> Unit
                    }
                }
            }
        }
    }
}

// =============================================================================
//  Bottom voice control bar
// =============================================================================

@Composable
private fun VoiceBar(state: AgentUiState, level: StateFlow<Float>, onMic: () -> Unit) {
    val speaking = state.state == AgentState.SPEAKING || state.state == AgentState.STREAMING
    val waveColor = if (speaking) Vaani.Phosphor else Vaani.Saffron
    Column(
        Modifier.fillMaxWidth().background(Vaani.Slate).padding(top = 10.dp, bottom = 14.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Waveform(level, waveColor, Modifier.fillMaxWidth().height(44.dp).padding(horizontal = 16.dp))
        Spacer(Modifier.height(8.dp))
        MicButton(state.state, enabled = state.runtimeReady, onClick = onMic)
        Spacer(Modifier.height(8.dp))
        Text(
            if (state.state == AgentState.LISTENING && state.highlightedPrompt != null)
                "Listening... say: " + BENCHMARK_PROMPTS[state.highlightedPrompt].text
            else state.state.label,
            style = MaterialTheme.typography.labelLarge,
            color = when (state.state) {
                AgentState.LISTENING -> Vaani.Saffron
                AgentState.SPEAKING, AgentState.STREAMING, AgentState.TRANSCRIBING -> Vaani.Phosphor
                AgentState.ERROR -> Vaani.Kumkum
                else -> Vaani.Paper
            },
            maxLines = 2, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(horizontal = 24.dp),
        )
    }
}

@Composable
private fun MicButton(state: AgentState, enabled: Boolean, onClick: () -> Unit) {
    val fill by animateColorAsState(
        when (state) {
            AgentState.LISTENING -> Vaani.Saffron
            AgentState.SPEAKING, AgentState.STREAMING -> Vaani.Phosphor
            else -> Vaani.SlateHigh
        }, label = "micFill",
    )
    val ring by animateColorAsState(
        when (state) {
            AgentState.IDLE -> Vaani.Saffron
            AgentState.TRANSCRIBING -> Vaani.Phosphor
            else -> Color.Transparent
        }, label = "micRing",
    )
    val glyph = if (state == AgentState.SPEAKING || state == AgentState.STREAMING || state == AgentState.LISTENING) Glyph.STOP else Glyph.MIC
    val glyphTint = if (fill == Vaani.SlateHigh) Vaani.Paper else Vaani.NightInk
    Box(
        Modifier
            .size(76.dp)
            .clip(CircleShape)
            .background(fill)
            .border(2.dp, ring, CircleShape)
            .clickable(enabled = enabled, onClick = onClick)
            .semantics { contentDescription = state.label },
        contentAlignment = Alignment.Center,
    ) {
        if (state == AgentState.TRANSCRIBING) Spinner(Vaani.Phosphor)
        GlyphIcon(glyph, if (enabled) glyphTint else Vaani.Steel, 30.dp)
    }
}

@Composable
private fun Spinner(color: Color) {
    var t by remember { mutableLongStateOf(0L) }
    LaunchedEffect(Unit) { while (true) withFrameNanos { t = it } }
    Canvas(Modifier.size(76.dp)) {
        val sweepStart = ((t / 4_000_000L) % 360).toFloat()
        drawArc(color, sweepStart, 90f, false, Offset(3f, 3f), Size(size.width - 6f, size.height - 6f),
            style = Stroke(width = 3f, cap = StrokeCap.Round))
    }
}

/**
 * 60 fps scrolling RMS waveform. Levels arrive at ~31 Hz (32 ms mic chunks / 46 ms TTS
 * chunks); each display frame eases toward the newest level and the history scrolls on a
 * fixed 30 ms clock, so motion stays smooth regardless of the audio callback cadence.
 * Drawing reads frame-time state in the draw phase only (no recomposition per frame).
 */
@Composable
private fun Waveform(level: StateFlow<Float>, color: Color, modifier: Modifier) {
    val bars = 56
    val history = remember { FloatArray(bars) }
    var head by remember { mutableLongStateOf(0L) }
    var frame by remember { mutableLongStateOf(0L) }
    var eased by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(Unit) {
        var lastPush = 0L
        while (true) {
            withFrameNanos { now ->
                val raw = level.value.coerceAtLeast(0f)
                val target = (ln(1f + 60f * raw) / ln(61f) * 1.35f).coerceIn(0f, 1f)
                eased += (target - eased) * 0.3f
                if (now - lastPush > 30_000_000L) {
                    history[(head % bars).toInt()] = eased
                    head++
                    lastPush = now
                }
                frame = now
            }
        }
    }
    Canvas(modifier.semantics { contentDescription = "Audio level" }) {
        frame.let { }                      // redraw every frame
        val w = size.width
        val h = size.height
        val gap = w / bars
        val barW = gap * 0.55f
        val mid = h / 2f
        for (i in 0 until bars) {
            val idx = ((head + i) % bars).toInt()
            val v = maxOf(history[idx], 0.04f)
            val age = i.toFloat() / bars
            val bh = v * h * 0.95f
            drawRoundRect(
                color.copy(alpha = 0.25f + 0.75f * age),
                Offset(i * gap + (gap - barW) / 2f, mid - bh / 2f),
                Size(barW, bh), CornerRadius(barW / 2f),
            )
        }
    }
}
