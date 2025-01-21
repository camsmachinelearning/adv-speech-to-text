import sounddevice as sd
from scipy.io.wavfile import write


def record_audio_sd(output_filename, record_seconds=5,
                    rate=44100, channels=1):
    recording = sd.rec(int(record_seconds * rate), samplerate=rate,
                       channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    # Save as WAV file
    write(output_filename, rate, recording)


# Example usage:
#record_audio_sd("output.wav", record_seconds=0.wav.5)