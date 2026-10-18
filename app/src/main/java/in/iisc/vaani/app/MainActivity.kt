package `in`.iisc.vaani.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import `in`.iisc.vaani.agent.VoiceAgentViewModel
import `in`.iisc.vaani.ui.VaaniScreen
import `in`.iisc.vaani.ui.VaaniTheme

/**
 * VaaniEdge launcher: hosts the Jetpack Compose voice-agent UI, requests the microphone
 * permission, and keeps the screen awake while the research HUD is in use.
 */
class MainActivity : ComponentActivity() {

    private val vm: VoiceAgentViewModel by viewModels()

    private val micPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        vm.setMicPermission(granted)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        vm.setMicPermission(granted)
        if (!granted) micPermission.launch(Manifest.permission.RECORD_AUDIO)

        setContent {
            VaaniTheme { VaaniScreen(vm) }
        }
    }
}
