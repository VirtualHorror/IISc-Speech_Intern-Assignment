package `in`.iisc.vaani.llm

import android.content.Context
import android.content.SharedPreferences
import `in`.iisc.vaani.BuildConfig
import `in`.iisc.vaani.neural.LanguageMode

/**
 * Runtime LLM configuration. Keys default to the build-time `BuildConfig.DEFAULT_*_API_KEY`
 * (injected from the git-ignored local.properties) and can be overridden in the app's
 * Settings dialog; overrides live in app-private SharedPreferences (backups disabled).
 */
class LlmSettings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("vaani_llm", Context.MODE_PRIVATE)

    var groqKey: String
        get() = prefs.getString(KEY_GROQ, null)?.takeIf { it.isNotBlank() } ?: BuildConfig.DEFAULT_GROQ_API_KEY
        set(v) = prefs.edit().putString(KEY_GROQ, v.trim()).apply()

    var geminiKey: String
        get() = prefs.getString(KEY_GEMINI, null)?.takeIf { it.isNotBlank() } ?: BuildConfig.DEFAULT_GEMINI_API_KEY
        set(v) = prefs.edit().putString(KEY_GEMINI, v.trim()).apply()

    /**
     * The primary rung: Groq Qwen3.8-27B unless the user picked another model. A stored id that
     * is no longer offered (e.g. the retired Llama models) falls back to the default.
     */
    var model: LlmModel
        get() = LlmModels.byId(prefs.getString(KEY_MODEL, null)) ?: LlmModels.DEFAULT
        set(v) = prefs.edit().putString(KEY_MODEL, v.id).apply()

    /** Optional model-id override (same provider), e.g. when a provider retires a model. */
    var modelIdOverride: String
        get() = prefs.getString(KEY_MODEL_OVERRIDE, "") ?: ""
        set(v) = prefs.edit().putString(KEY_MODEL_OVERRIDE, v.trim()).apply()

    /** The primary actually requested: [model] with [modelIdOverride] applied. */
    fun effectiveModel(): LlmModel {
        val m = model
        val o = modelIdOverride
        return if (o.isBlank() || o == m.id) m else m.copy(id = o, label = "${m.provider.label} $o", note = "id override")
    }

    /**
     * The failover cascade for one turn: the primary, then [LlmModels.CASCADE] (which still
     * contains the un-overridden model). Rungs whose provider has no key are dropped.
     */
    fun cascade(): List<LlmTarget> = LlmModels.cascadeFrom(effectiveModel())
        .filter { hasKey(it.provider) }
        .map { LlmTarget(it, keyFor(it.provider)) }

    var languageMode: LanguageMode
        get() = runCatching { LanguageMode.valueOf(prefs.getString(KEY_LANG, "AUTO")!!) }.getOrDefault(LanguageMode.AUTO)
        set(v) = prefs.edit().putString(KEY_LANG, v.name).apply()

    /** When off (or no key), turns are answered by the on-device LocalResponder. */
    var cloudEnabled: Boolean
        get() = prefs.getBoolean(KEY_CLOUD, true)
        set(v) = prefs.edit().putBoolean(KEY_CLOUD, v).apply()

    fun keyFor(provider: LlmProvider): String = if (provider == LlmProvider.GROQ) groqKey else geminiKey

    fun hasKey(provider: LlmProvider): Boolean = keyFor(provider).isNotBlank()

    /** Masked key for display, e.g. `AQ.Ab8…IsW`. */
    fun maskedKey(provider: LlmProvider): String {
        val k = keyFor(provider)
        return when {
            k.isBlank() -> "not set"
            k.length <= 8 -> "••••"
            else -> k.take(5) + "…" + k.takeLast(3)
        }
    }

    fun keySource(provider: LlmProvider): String {
        val pref = prefs.getString(if (provider == LlmProvider.GROQ) KEY_GROQ else KEY_GEMINI, null)
        return when {
            !pref.isNullOrBlank() -> "Settings"
            keyFor(provider).isNotBlank() -> "BuildConfig"
            else -> "-"
        }
    }

    private companion object {
        const val KEY_GROQ = "groq_api_key"
        const val KEY_GEMINI = "gemini_api_key"
        const val KEY_MODEL = "model_id"
        const val KEY_MODEL_OVERRIDE = "model_id_override"
        const val KEY_LANG = "language_mode"
        const val KEY_CLOUD = "cloud_enabled"
    }
}
