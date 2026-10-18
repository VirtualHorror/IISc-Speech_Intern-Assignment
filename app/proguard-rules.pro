# Keep native-method holders so JNI binding to libvaani_dsp_neon.so survives R8.
-keepclasseswithmembernames class * {
    native <methods>;
}
-keep class in.iisc.vaani.dsp.NeonAudioProcessor { *; }

# ONNX Runtime ships its own native bindings; keep its API surface.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
