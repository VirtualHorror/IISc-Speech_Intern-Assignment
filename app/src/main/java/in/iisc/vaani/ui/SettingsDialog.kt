package `in`.iisc.vaani.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import `in`.iisc.vaani.agent.AgentUiState
import `in`.iisc.vaani.llm.LlmModel
import `in`.iisc.vaani.llm.LlmModels
import `in`.iisc.vaani.neural.LanguageMode

/** LLM provider/model, runtime API keys (override BuildConfig) and recognition language. */
@Composable
fun SettingsDialog(
    state: AgentUiState,
    keyStatus: Map<String, String>,
    onDismiss: () -> Unit,
    onSave: (LlmModel, String?, String?, LanguageMode, Boolean, String) -> Unit,
) {
    var model by remember { mutableStateOf(LlmModels.ALL.firstOrNull { it.provider == state.model.provider && (it.id == state.model.id || it.id == state.model.fallbackId) } ?: state.model) }
    var override by remember { mutableStateOf(state.modelIdOverride) }
    var groq by remember { mutableStateOf("") }
    var gemini by remember { mutableStateOf("") }
    var mode by remember { mutableStateOf(state.languageMode) }
    var cloud by remember { mutableStateOf(state.cloudEnabled) }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Vaani.Slate,
        title = { Text("Assistant settings", style = MaterialTheme.typography.titleMedium, color = Vaani.Paper) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Cloud replies", style = MaterialTheme.typography.titleSmall, color = Vaani.Paper)
                        Text("Off: answers come from the on-device fallback, no network.",
                            style = MaterialTheme.typography.bodySmall, color = Vaani.Fog)
                    }
                    Switch(
                        checked = cloud, onCheckedChange = { cloud = it },
                        colors = SwitchDefaults.colors(checkedTrackColor = Vaani.Cloud, checkedThumbColor = Vaani.NightInk),
                    )
                }
                Spacer(Modifier.height(12.dp))
                Text("Model", style = MaterialTheme.typography.titleSmall, color = Vaani.Paper)
                for (m in LlmModels.ALL) {
                    Row(
                        Modifier.fillMaxWidth().clickable { model = m }.padding(vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(selected = model == m, onClick = { model = m },
                            colors = RadioButtonDefaults.colors(selectedColor = Vaani.Cloud))
                        Column {
                            Text(m.label, style = MaterialTheme.typography.bodyMedium, color = Vaani.Paper)
                            Text(m.id, style = MaterialTheme.typography.labelSmall, color = Vaani.Fog)
                        }
                    }
                }
                OutlinedTextField(
                    value = override, onValueChange = { override = it },
                    label = { Text("Model id override (optional)") },
                    placeholder = { Text("e.g. a newer ${model.provider.label} model id") },
                    supportingText = { Text("Use if the provider retires ${model.id}. Blank uses the model above.") },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Vaani.Cloud, unfocusedBorderColor = Vaani.Steel,
                        focusedLabelColor = Vaani.Cloud, unfocusedLabelColor = Vaani.Fog,
                        focusedTextColor = Vaani.Paper, unfocusedTextColor = Vaani.Paper,
                        focusedSupportingTextColor = Vaani.Fog, unfocusedSupportingTextColor = Vaani.Fog,
                    ),
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(12.dp))
                KeyField("Groq API key", "gsk_…", keyStatus["groq"] ?: "", groq) { groq = it }
                Spacer(Modifier.height(8.dp))
                KeyField("Gemini API key", "AIza… or AQ.…", keyStatus["gemini"] ?: "", gemini) { gemini = it }
                Spacer(Modifier.height(12.dp))
                Text("Recognition language", style = MaterialTheme.typography.titleSmall, color = Vaani.Paper)
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    for (lm in LanguageMode.values()) {
                        FilterChip(
                            selected = mode == lm, onClick = { mode = lm },
                            label = { Text(lm.label) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = Vaani.Saffron, selectedLabelColor = Vaani.NightInk,
                                labelColor = Vaani.Fog,
                            ),
                        )
                    }
                }
                Text("Auto picks English or Hinglish from Whisper's first decoding step.",
                    style = MaterialTheme.typography.bodySmall, color = Vaani.Fog)
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onSave(model, groq.takeIf { it.isNotBlank() }, gemini.takeIf { it.isNotBlank() }, mode, cloud, override)
            }) { Text("Save", color = Vaani.Saffron) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel", color = Vaani.Fog) } },
    )
}

@Composable
private fun KeyField(label: String, hint: String, current: String, value: String, onChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onChange,
        label = { Text(label) },
        placeholder = { Text(hint) },
        supportingText = { Text("Current: $current. Leave blank to keep it.") },
        singleLine = true,
        visualTransformation = PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = Vaani.Cloud, unfocusedBorderColor = Vaani.Steel,
            focusedLabelColor = Vaani.Cloud, unfocusedLabelColor = Vaani.Fog,
            focusedTextColor = Vaani.Paper, unfocusedTextColor = Vaani.Paper,
            focusedSupportingTextColor = Vaani.Fog, unfocusedSupportingTextColor = Vaani.Fog,
        ),
        modifier = Modifier.fillMaxWidth(),
    )
}
