package `in`.iisc.vaani.app

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

/**
 * Minimal launcher surface for Session 1.
 *
 * Its only job this session is to give `assembleDebug` a launchable entry point that
 * links the custom native DSP library (`libvaani_dsp_neon.so`) and the on-device
 * neural engines. The real press-to-talk conversational UI — live transcript,
 * barge-in TTS playback, latency HUD — is a Session 2/3 deliverable.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = TextView(this).apply {
            text = buildString {
                append("VaaniEdge\n")
                append("On-Device Speech AI\n")
                append("IISc Computational Intelligence Lab")
            }
            textSize = 20f
            setPadding(48, 96, 48, 48)
        }
        setContentView(view)
    }
}
