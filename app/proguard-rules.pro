# VaaniEdge R8 rules. Not applied at the moment: release ships with isMinifyEnabled = false
# (the R8 build closed at launch on the OnePlus 11R). Kept for a future R8 attempt.
# Nothing in the app uses reflection-based serialisation (configs, vocab and SSE chunks are
# parsed by hand with org.json), so the keeps below are about native code and readable traces.

# ---- JNI: libvaani_dsp_neon.so binds Java_in_iisc_vaani_dsp_NeonAudioProcessor_* by name ----
-keep class in.iisc.vaani.dsp.** { *; }
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# ---- ONNX Runtime: its JNI layer looks up ai.onnxruntime classes, fields, constructors and
#      callbacks by name (OnnxTensor, OrtSession$Result, OrtException, ...) ----
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# ---- @Keep-annotated code ----
-keep @androidx.annotation.Keep class * { *; }
-keepclasseswithmembers class * {
    @androidx.annotation.Keep <methods>;
}
-keepclasseswithmembers class * {
    @androidx.annotation.Keep <fields>;
}

# ---- Model/config data + the LLM/SSE layer: kept by name so on-device crash traces, the
#      VaaniTelemetry JSON and the failover notes stay readable (tiny size cost) ----
-keep class in.iisc.vaani.neural.VaaniTtsConfig { *; }
-keep class in.iisc.vaani.neural.SttResult { *; }
-keep class in.iisc.vaani.llm.** { *; }

# ---- OkHttp / Okio ship consumer rules; the optional desktop TLS providers do not exist on Android ----
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**

# ---- Readable stack traces from the field ----
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
