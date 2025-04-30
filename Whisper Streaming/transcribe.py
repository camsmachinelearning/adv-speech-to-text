from whisper_online import *
import sounddevice as sd


source_language = "en"
target_language = "en"

faster_whisper_asr = FasterWhisperASR(source_language, "large-v2")
faster_whisper_asr.use_vad()

online_processor = OnlineASRProcessor(faster_whisper_asr)

while True:
    try:
        pass
    except KeyboardInterrupt:
        o = online.finish()
        print(o)
        break