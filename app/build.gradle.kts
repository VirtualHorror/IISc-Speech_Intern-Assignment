plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
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
        versionCode = 1
        versionName = "1.0.0-session1"

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
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    testImplementation("junit:junit:4.13.2")
}
