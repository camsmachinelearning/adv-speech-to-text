from pyannote.audio import Pipeline
import sounddevice as sd
from pyannote.audio import Model
m = Model

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="")

with sd.RawInputStream(samplerate=16e3, channels=1):
    

# run the pipeline on an audio file
diarization = pipeline("./audio.wav")


# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)