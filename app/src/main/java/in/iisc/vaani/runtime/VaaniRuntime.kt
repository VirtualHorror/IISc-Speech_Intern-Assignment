package `in`.iisc.vaani.runtime

import android.content.Context
import `in`.iisc.vaani.BuildConfig
import `in`.iisc.vaani.codec.EnglishLexicon
import `in`.iisc.vaani.codec.HinglishCodec
import `in`.iisc.vaani.dsp.NeonAudioProcessor
import `in`.iisc.vaani.neural.LanguageMode
import `in`.iisc.vaani.neural.VaaniSttEngine
import `in`.iisc.vaani.neural.VaaniTtsConfig
import `in`.iisc.vaani.neural.VaaniTtsEngine
import `in`.iisc.vaani.neural.VaaniVadEngine
import `in`.iisc.vaani.neural.VaaniVocab
import `in`.iisc.vaani.neural.WhisperFiles
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Asset -> engine glue: owns the three on-device neural engines for the app's lifetime.
 *
 * `.onnx` graphs are stored uncompressed in the APK; each is extracted ONCE to
 * `filesDir/models/v<versionCode>/` and opened by path, so ONNX Runtime reads it natively
 * (no 30 MB Java-heap byte arrays). The Mel filterbank `.bin` is decoded as little-endian
 * float32 [80 x 201]; the vocab / TTS config / CMU lexicon are parsed from JSON / text.
 * A short warm-up pass runs every engine once so the first real turn is not cold.
 */
class VaaniRuntime private constructor(
    val vad: VaaniVadEngine,
    val stt: VaaniSttEngine,
    val tts: VaaniTtsEngine,
    val report: LoadReport,
) : AutoCloseable {

    data class LoadReport(
        val totalMs: Long,
        val extractMs: Long,
        val sttLoadMs: Long,
        val ttsLoadMs: Long,
        val warmupMs: Long,
        val kvCache: Boolean,
        val modelMb: Double,
        val lexiconWords: Int,
        val hindiModel: String? = null,
    )

    override fun close() {
        runCatching { vad.close() }
        runCatching { stt.close() }
        runCatching { tts.close() }
    }

    companion object {
        private const val DIR = "models"
        const val ENCODER = "vaani_stt_encoder.int8.onnx"
        const val DECODER = "vaani_stt_decoder.int8.onnx"
        const val DECODER_PAST = "vaani_stt_decoder_with_past.int8.onnx"
        // Hindi specialist (optional): a Hindi-trained Whisper-tiny for every Hindi turn.
        const val HI_ENCODER = "vaani_stt_hi_encoder.fp16.onnx"
        const val HI_DECODER = "vaani_stt_hi_decoder.int8.onnx"
        const val HI_DECODER_PAST = "vaani_stt_hi_decoder_with_past.int8.onnx"
        const val HI_LABEL = "whisper-tiny-hi"
        const val TTS = "vaani_tts_vits.fp16.onnx"
        const val VAD = "vaani_vad.onnx"
        private const val MEL = "vaani_mel_filters.bin"
        private const val VOCAB = "vaani_vocab.json"
        private const val TTS_CONFIG = "vaani_tts_config.json"
        private const val LEXICON = "lexicon/vaani_en_cmudict.txt"

        suspend fun load(context: Context, onProgress: (String) -> Unit = {}): VaaniRuntime =
            withContext(Dispatchers.IO) {
                val t0 = System.nanoTime()
                NeonAudioProcessor.ensureLoaded()

                onProgress("Extracting INT8 models…")
                val dir = File(context.filesDir, "$DIR/v${BuildConfig.VERSION_CODE}").apply { mkdirs() }
                File(context.filesDir, DIR).listFiles()?.forEach { if (it != dir) it.deleteRecursively() }
                val hindiNames = listOf(HI_ENCODER, HI_DECODER, HI_DECODER_PAST)
                val withHindi = hindiNames.all { assetExists(context, it) }
                val names = listOf(ENCODER, DECODER, DECODER_PAST, TTS, VAD) + if (withHindi) hindiNames else emptyList()
                val paths = names.associateWith { extract(context, dir, it) }
                val tExtract = System.nanoTime()

                onProgress("Loading lexicon + tokenizer…")
                val lexicon = context.assets.open(LEXICON).bufferedReader().use { EnglishLexicon.parse(it.readText()) }
                HinglishCodec.englishLexicon = lexicon
                val vocab = VaaniVocab.fromJson(readAsset(context, VOCAB))
                val ttsConfig = VaaniTtsConfig.fromJson(readAsset(context, TTS_CONFIG))
                val mel = loadMelFilters(context)

                onProgress("Loading Whisper INT8 (KV-cache) …")
                val vad = VaaniVadEngine(paths.getValue(VAD).path)
                val stt = VaaniSttEngine(
                    WhisperFiles(paths.getValue(ENCODER).path, paths.getValue(DECODER).path, paths[DECODER_PAST]?.path, "whisper-tiny"),
                    vocab, mel,
                    hindi = if (withHindi) WhisperFiles(paths.getValue(HI_ENCODER).path, paths.getValue(HI_DECODER).path,
                        paths.getValue(HI_DECODER_PAST).path, HI_LABEL) else null,
                )
                val tStt = System.nanoTime()

                onProgress("Loading Indic VITS voice…")
                val tts = VaaniTtsEngine(paths.getValue(TTS).path, ttsConfig)
                val tTts = System.nanoTime()

                onProgress("Warming up NEON + ONNX kernels…")
                runCatching {
                    vad.process(FloatArray(VaaniVadEngine.CHUNK_SAMPLES)); vad.reset()
                    stt.transcribe(FloatArray(NeonAudioProcessor.SAMPLE_RATE / 2), LanguageMode.ENGLISH)
                    if (withHindi) stt.transcribe(FloatArray(NeonAudioProcessor.SAMPLE_RATE / 2), LanguageMode.HINGLISH)
                    tts.synthesize(HinglishCodec.toAcousticDevanagariScript("Hello", "en"))
                }
                val tEnd = System.nanoTime()

                val mb = paths.values.sumOf { it.length() } / 1e6
                VaaniRuntime(
                    vad, stt, tts,
                    LoadReport(
                        totalMs = (tEnd - t0) / 1_000_000,
                        extractMs = (tExtract - t0) / 1_000_000,
                        sttLoadMs = (tStt - tExtract) / 1_000_000,
                        ttsLoadMs = (tTts - tStt) / 1_000_000,
                        warmupMs = (tEnd - tTts) / 1_000_000,
                        kvCache = stt.usesKvCache,
                        modelMb = mb,
                        lexiconWords = lexicon.size,
                        hindiModel = stt.hindiModelLabel,
                    ),
                )
            }

        private fun extract(context: Context, dir: File, name: String): File {
            val out = File(dir, name)
            val expected = runCatching { context.assets.openFd("$DIR/$name").use { it.length } }.getOrDefault(-1L)
            if (out.exists() && (expected < 0 || out.length() == expected)) return out
            val tmp = File(dir, "$name.part")
            context.assets.open("$DIR/$name").use { input ->
                tmp.outputStream().use { input.copyTo(it, 1 shl 20) }
            }
            if (!tmp.renameTo(out)) { tmp.copyTo(out, overwrite = true); tmp.delete() }
            return out
        }

        private fun assetExists(context: Context, name: String): Boolean =
            runCatching { context.assets.openFd("$DIR/$name").use { true } }.getOrDefault(false)

        private fun readAsset(context: Context, name: String): String =
            context.assets.open("$DIR/$name").bufferedReader(Charsets.UTF_8).use { it.readText() }

        /** `vaani_mel_filters.bin`: little-endian float32, row-major [80][201] = 16080 floats. */
        private fun loadMelFilters(context: Context): FloatArray {
            val bytes = context.assets.open("$DIR/$MEL").use { it.readBytes() }
            val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val out = FloatArray(fb.remaining()).also { fb.get(it) }
            require(out.size == NeonAudioProcessor.N_MELS * NeonAudioProcessor.N_FFT_BINS) {
                "mel filterbank has ${out.size} floats"
            }
            return out
        }
    }
}
