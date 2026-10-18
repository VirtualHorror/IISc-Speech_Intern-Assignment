import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Cloud-LLM keys come from the git-ignored local.properties (or the environment) and are
// compiled into BuildConfig so the debug APK works on the demo phone out of the box.
// They never enter version control; at runtime the Settings dialog can override them.
val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

fun secret(name: String): String {
    val raw = (localProps.getProperty(name) ?: System.getenv(name) ?: "").trim()
    return "\"" + raw.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
}

android {
    // Package root is `in.iisc.vaani` (in = India TLD, iisc = Indian Institute of Science).
    namespace = "in.iisc.vaani"
    compileSdk = 34
    ndkVersion = "26.1.10909125"

    defaultConfig {
        applicationId = "in.iisc.vaani"
        minSdk = 28
        targetSdk = 34
        versionCode = 2
        versionName = "2.0.0-session2"

        buildConfigField("String", "DEFAULT_GROQ_API_KEY", secret("GROQ_API_KEY"))
        buildConfigField("String", "DEFAULT_GEMINI_API_KEY", secret("GEMINI_API_KEY"))

        // On-device target: OnePlus 11R 5G (Snapdragon 8+ Gen 1). Ship arm64 only.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3", "-ffast-math", "-funroll-loops")
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"   // matches Kotlin 1.9.24
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // On-device ONNX graphs and the raw Mel-filterbank matrix must be memory-mappable,
    // so they are stored uncompressed inside the APK.
    androidResources {
        noCompress.add("onnx")
        noCompress.add("bin")
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // Official Microsoft ONNX Runtime for Android (MIT). Provides libonnxruntime.so
    // for arm64-v8a plus the ai.onnxruntime.* Java/Kotlin API. All inference engines
    // in in.iisc.vaani.neural.* are written against this API from first principles.
    // Every shipped graph is gated by speech_lab/check_ort_compat.py under 1.18.0.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // Streaming cloud LLM transport (HTTP/2 + TLS 1.3 connection pooling).
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Jetpack Compose (Material 3) UI.
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.2")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.2")

    testImplementation("junit:junit:4.13.2")
    // Real org.json on the host JVM (android.jar only ships stubs) so codec/G2P tests can
    // parse the shipped vaani_tts_config.json and export closed-loop fixtures.
    testImplementation("org.json:json:20240303")
    // Local SSE server for the streaming LLM client tests (no network, no keys).
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
}
