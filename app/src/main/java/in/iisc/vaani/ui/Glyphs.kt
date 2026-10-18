package `in`.iisc.vaani.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import `in`.iisc.vaani.tools.WeatherCodes

/** Small hand-drawn line glyphs (no icon-font dependency). */
enum class Glyph { MIC, STOP, LOOP, REPLAY, SETTINGS, ALARM, BELL, CALENDAR, CHEVRON_DOWN, CHEVRON_UP }

@Composable
fun GlyphIcon(glyph: Glyph, tint: Color, size: Dp = 20.dp, modifier: Modifier = Modifier) {
    Canvas(modifier.size(size)) {
        val w = this.size.width
        val s = w / 24f
        val stroke = Stroke(width = 1.9f * s, cap = StrokeCap.Round)
        fun p(x: Float, y: Float) = Offset(x * s, y * s)
        when (glyph) {
            Glyph.MIC -> {
                drawRoundRect(tint, p(8.5f, 2.5f), Size(7f * s, 12f * s), CornerRadius(3.5f * s), style = stroke)
                val arc = Path().apply {
                    moveTo(5f * s, 11f * s)
                    cubicTo(5f * s, 19.5f * s, 19f * s, 19.5f * s, 19f * s, 11f * s)
                }
                drawPath(arc, tint, style = stroke)
                drawLine(tint, p(12f, 18f), p(12f, 21.5f), stroke.width, StrokeCap.Round)
            }
            Glyph.STOP -> drawRoundRect(tint, p(6f, 6f), Size(12f * s, 12f * s), CornerRadius(2.5f * s))
            Glyph.LOOP -> {
                drawArc(tint, 200f, 250f, false, p(4f, 4f), Size(16f * s, 16f * s), style = stroke)
                val head = Path().apply { moveTo(17.5f * s, 2.5f * s); lineTo(20f * s, 7.5f * s); lineTo(14.6f * s, 7.8f * s) }
                drawPath(head, tint, style = stroke)
            }
            Glyph.REPLAY -> {
                drawArc(tint, 120f, 290f, false, p(4f, 4f), Size(16f * s, 16f * s), style = stroke)
                val tri = Path().apply { moveTo(10f * s, 9f * s); lineTo(15f * s, 12f * s); lineTo(10f * s, 15f * s); close() }
                drawPath(tri, tint)
            }
            Glyph.SETTINGS -> {
                drawCircle(tint, 3.2f * s, p(12f, 12f), style = stroke)
                for (k in 0 until 8) {
                    val a = Math.toRadians(k * 45.0)
                    val c = Offset(12f * s, 12f * s)
                    drawLine(
                        tint,
                        c + Offset((6.4f * s * Math.cos(a)).toFloat(), (6.4f * s * Math.sin(a)).toFloat()),
                        c + Offset((9f * s * Math.cos(a)).toFloat(), (9f * s * Math.sin(a)).toFloat()),
                        stroke.width, StrokeCap.Round,
                    )
                }
            }
            Glyph.ALARM -> {
                drawCircle(tint, 8f * s, p(12f, 13f), style = stroke)
                drawLine(tint, p(12f, 13f), p(12f, 9f), stroke.width, StrokeCap.Round)
                drawLine(tint, p(12f, 13f), p(14.8f, 14.8f), stroke.width, StrokeCap.Round)
                drawLine(tint, p(3.5f, 5.5f), p(6.5f, 3f), stroke.width, StrokeCap.Round)
                drawLine(tint, p(20.5f, 5.5f), p(17.5f, 3f), stroke.width, StrokeCap.Round)
            }
            Glyph.BELL -> {
                val bell = Path().apply {
                    moveTo(6f * s, 17f * s); lineTo(6f * s, 11f * s)
                    cubicTo(6f * s, 5f * s, 18f * s, 5f * s, 18f * s, 11f * s)
                    lineTo(18f * s, 17f * s); lineTo(4.5f * s, 17f * s); lineTo(19.5f * s, 17f * s)
                }
                drawPath(bell, tint, style = stroke)
                drawLine(tint, p(10.5f, 20f), p(13.5f, 20f), stroke.width, StrokeCap.Round)
            }
            Glyph.CALENDAR -> {
                drawRoundRect(tint, p(4f, 5.5f), Size(16f * s, 14.5f * s), CornerRadius(2.5f * s), style = stroke)
                drawLine(tint, p(4f, 10f), p(20f, 10f), stroke.width)
                drawLine(tint, p(8.5f, 3.5f), p(8.5f, 7f), stroke.width, StrokeCap.Round)
                drawLine(tint, p(15.5f, 3.5f), p(15.5f, 7f), stroke.width, StrokeCap.Round)
            }
            Glyph.CHEVRON_DOWN, Glyph.CHEVRON_UP -> {
                val up = glyph == Glyph.CHEVRON_UP
                val path = Path().apply {
                    moveTo(6f * s, (if (up) 15f else 9f) * s)
                    lineTo(12f * s, (if (up) 9f else 15f) * s)
                    lineTo(18f * s, (if (up) 15f else 9f) * s)
                }
                drawPath(path, tint, style = stroke)
            }
        }
    }
}

/** Weather pictogram drawn from the WMO code family. */
@Composable
fun WeatherGlyph(kind: WeatherCodes.Kind, size: Dp = 44.dp) {
    Canvas(Modifier.size(size)) {
        val s = this.size.width / 48f
        val stroke = Stroke(width = 2.2f * s, cap = StrokeCap.Round)
        fun cloud(color: Color, dy: Float = 0f) {
            drawCircle(color, 8f * s, Offset(18f * s, (24f + dy) * s))
            drawCircle(color, 10f * s, Offset(28f * s, (21f + dy) * s))
            drawRoundRect(color, Offset(10f * s, (24f + dy) * s), Size(28f * s, 9f * s), CornerRadius(4.5f * s))
        }
        when (kind) {
            WeatherCodes.Kind.CLEAR -> {
                drawCircle(Vaani.Saffron, 8f * s, Offset(24f * s, 24f * s))
                for (k in 0 until 8) {
                    val a = Math.toRadians(k * 45.0)
                    val c = Offset(24f * s, 24f * s)
                    drawLine(Vaani.Saffron,
                        c + Offset((12f * s * Math.cos(a)).toFloat(), (12f * s * Math.sin(a)).toFloat()),
                        c + Offset((16f * s * Math.cos(a)).toFloat(), (16f * s * Math.sin(a)).toFloat()),
                        stroke.width, StrokeCap.Round)
                }
            }
            WeatherCodes.Kind.CLOUD, WeatherCodes.Kind.FOG -> {
                drawCircle(Vaani.Saffron, 6f * s, Offset(32f * s, 14f * s))
                cloud(Vaani.Fog)
                if (kind == WeatherCodes.Kind.FOG) drawLine(Vaani.Fog, Offset(10f * s, 40f * s), Offset(38f * s, 40f * s), stroke.width, StrokeCap.Round)
            }
            WeatherCodes.Kind.RAIN, WeatherCodes.Kind.SNOW -> {
                cloud(Vaani.Fog, -4f)
                for (x in listOf(16f, 24f, 32f)) {
                    drawLine(Vaani.Phosphor, Offset(x * s, 34f * s), Offset((x - 2.5f) * s, 41f * s), stroke.width, StrokeCap.Round)
                }
            }
            WeatherCodes.Kind.STORM -> {
                cloud(Vaani.Fog, -4f)
                val bolt = Path().apply {
                    moveTo(25f * s, 31f * s); lineTo(19f * s, 39f * s); lineTo(24f * s, 39f * s); lineTo(21f * s, 46f * s)
                    lineTo(29f * s, 36.5f * s); lineTo(24f * s, 36.5f * s); close()
                }
                drawPath(bolt, Vaani.Saffron)
                drawLine(Vaani.Phosphor, Offset(14f * s, 34f * s), Offset(12f * s, 40f * s), stroke.width, StrokeCap.Round)
                drawLine(Vaani.Phosphor, Offset(34f * s, 34f * s), Offset(32f * s, 40f * s), stroke.width, StrokeCap.Round)
            }
        }
    }
}
