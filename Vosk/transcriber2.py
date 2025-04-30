import sys
import os
import queue
import sounddevice as sd

from vosk import Model, KaldiRecognizer, SpkModel

model = Model(model_path="./vosk-danzu")

SPK_MODEL_PATH = "model-spk"

if not os.path.exists(SPK_MODEL_PATH):
  print("SPK model not found")

  sys.exit(1)

samplerate = 16e3 # 16kHz

rec = KaldiRecognizer(model, samplerate)

q = queue.Queue()

def audio_callback(data_in, frames, time, status):
  if status:
    print(status, file=sys.stderr)
  q.put(bytes(data_in))

try:
  with sd.RawInputStream(samplerate=samplerate,
                         blocksize=8000,
                         dtype="int16",
                         channels=1,
                         callback=audio_callback):
    
    print("Listening... Press Ctrl+C to exit")

    while True:
      data = q.get()
      if rec.AcceptWaveform(data):
          print(rec.Result())
      else:
          print(rec.PartialResult())
except KeyboardInterrupt:
  print("\nDone Listening")
except Exception as e:
  print(f"Error: {e}")

