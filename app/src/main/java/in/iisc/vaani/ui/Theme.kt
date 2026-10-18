package `in`.iisc.vaani.ui

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import `in`.iisc.vaani.R

/**
 * VaaniEdge instrument palette. Two accents with a meaning, never decoration:
 * saffron = the human voice going in (mic, listening waveform, Hinglish badge),
 * phosphor = the machine coming out (telemetry figures, TTS waveform).
 */
object Vaani {
    val NightInk = Color(0xFF121722)
    val Slate = Color(0xFF1B2231)
    val SlateHigh = Color(0xFF232C3E)
    val Steel = Color(0xFF2C3547)
    val Paper = Color(0xFFE8ECF2)
    val Fog = Color(0xFF8E9AAE)
    val Saffron = Color(0xFFF4A340)
    val Phosphor = Color(0xFF5CD6C7)
    val Cloud = Color(0xFFA99BF0)      // cloud LLM time in the latency waterfall
    val Kumkum = Color(0xFFE5484D)
    val Leaf = Color(0xFF7BC67E)
}

val PlexSans = FontFamily(
    Font(R.font.plex_sans_regular, FontWeight.Normal),
    Font(R.font.plex_sans_medium, FontWeight.Medium),
    Font(R.font.plex_sans_semibold, FontWeight.SemiBold),
)

val PlexDevanagari = FontFamily(Font(R.font.plex_sans_devanagari_regular, FontWeight.Normal))

/** Tabular figures so live latency numbers never jitter as digits change. */
val Figures = TextStyle(fontFamily = PlexSans, fontFeatureSettings = "tnum")

private val VaaniTypography = Typography(
    headlineSmall = TextStyle(fontFamily = PlexSans, fontWeight = FontWeight.SemiBold, fontSize = 22.sp, lineHeight = 28.sp),
    titleMedium = TextStyle(fontFamily = PlexSans, fontWeight = FontWeight.SemiBold, fontSize = 16.sp, lineHeight = 22.sp),
    titleSmall = TextStyle(fontFamily = PlexSans, fontWeight = FontWeight.Medium, fontSize = 14.sp, lineHeight = 20.sp),
    bodyLarge = TextStyle(fontFamily = PlexSans, fontSize = 16.sp, lineHeight = 23.sp),
    bodyMedium = TextStyle(fontFamily = PlexSans, fontSize = 14.sp, lineHeight = 20.sp),
    bodySmall = TextStyle(fontFamily = PlexSans, fontSize = 12.sp, lineHeight = 17.sp),
    labelLarge = TextStyle(fontFamily = PlexSans, fontWeight = FontWeight.Medium, fontSize = 14.sp, lineHeight = 18.sp),
    labelMedium = TextStyle(fontFamily = PlexSans, fontWeight = FontWeight.Medium, fontSize = 12.sp, lineHeight = 16.sp),
    labelSmall = TextStyle(fontFamily = PlexSans, fontWeight = FontWeight.Medium, fontSize = 11.sp, lineHeight = 14.sp),
)

private val VaaniShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(14.dp),
    large = RoundedCornerShape(20.dp),
)

@Composable
fun VaaniTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Vaani.Saffron,
            onPrimary = Vaani.NightInk,
            secondary = Vaani.Phosphor,
            onSecondary = Vaani.NightInk,
            background = Vaani.NightInk,
            onBackground = Vaani.Paper,
            surface = Vaani.Slate,
            onSurface = Vaani.Paper,
            surfaceVariant = Vaani.SlateHigh,
            onSurfaceVariant = Vaani.Fog,
            outline = Vaani.Steel,
            error = Vaani.Kumkum,
        ),
        typography = VaaniTypography,
        shapes = VaaniShapes,
        content = content,
    )
}
