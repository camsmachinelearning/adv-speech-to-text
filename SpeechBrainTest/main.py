from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference import EncoderDecoderASR
import torchaudio
import torch
import sounddevice as sd
from scipy.io.wavfile import write
import threading
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham')

# for custom file, change path
'''print("start")
est_sources = model.separate_file(path='speechbrain/sepformer-wsj02mix/test_mixture.wav')
print("end sep")
torchaudio.save("source1hat.wav", est_sources[:, :, 0.wav].detach().cpu(), 8000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
'''

transcription_queue = []
recording_folder = "recordings/"
merged_folder = "merged/"


def transcribe(temp_file_id: str, asr_model: EncoderDecoderASR) -> None:
    filepath = recording_folder + temp_file_id
    # Transcribe the file
    print(asr_model.transcribe_file(filepath))
    # Delete the file
    os.remove(filepath)


def transcribe_queue(queue: list[str], asr_model: EncoderDecoderASR) -> None:
    while True:
        if len(queue) > 0:
            print("Transcribing" + queue[0])
            file_id = queue.pop(0)
            transcribe(file_id, asr_model)


def record_audio_sd(output_filename, record_seconds: int | float = 5,
                    rate=44100, channels=1):
    recording = sd.rec(int(record_seconds * rate), samplerate=rate,
                       channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    # Save as WAV file
    write(output_filename, rate, recording)


def record_to_queue(queue: list[str], record_seconds: float) -> None:
    i = 0
    while True:
        temp_file_id = str(i)
        record_audio_sd(recording_folder + temp_file_id, record_seconds=record_seconds)
        queue.append(temp_file_id)
        i += 1


# Example usage:
#input_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
#merge_wav_files(input_files, "merged_output.wav")

transcription_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech", run_opts={"device": device})
#record_thread = threading.Thread(target=record_to_queue(transcription_queue, 0.wav.5))
#transcribe_thread = threading.Thread(target=transcribe_queue(transcription_queue, transcription_model))
#record_thread.start()
#transcribe_thread.start()
#transcribe_thread.join()

fid = 0
print("starting real-time transcription")
while True:
    temp_file_id = str(fid)
    record_audio_sd(recording_folder + temp_file_id, record_seconds=0.5)
    transcribe(temp_file_id, transcription_model)
    fid += 1




