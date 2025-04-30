import sys
import os
import queue
import sounddevice as sd
import numpy as np
import json

from vosk import Model, KaldiRecognizer, SpkModel

print(sd.query_devices())

model = Model(model_path="./Gigaspeech")

SPK_MODEL_PATH = "model-spk"

if not os.path.exists(SPK_MODEL_PATH):
  print("SPK model not found")

  sys.exit(1)


samplerate = 16e3 # 16kHz

rec = KaldiRecognizer(model, samplerate)
rec.SetSpkModel(SpkModel(SPK_MODEL_PATH))

speaker_signatures = []

def cosine_dist(x, y):
  nx = np.array(x)
  ny = np.array(y)
  return np.dot(nx, ny) / (np.linalg.norm(nx) * np.linalg.norm(ny))

q = queue.Queue()

def audio_callback(data_in, frames, time, status):
  if status:
    print(status, file=sys.stderr)
  q.put(bytes(data_in))
  
def handle_speaker(speaker_signature):
  global speaker_signatures

  if len(speaker_signatures) == 0:
    speaker_signatures.append(speaker_signature)
    return 1
  else:
    for i, signature in enumerate(speaker_signatures):
      distance = cosine_dist(signature, speaker_signature)
      print(distance)
      if distance < 0.75:  # Threshold for similarity
        # Update signature with rolling average
        speaker_signatures[i] = np.mean([signature, speaker_signature], axis=0)
        return i + 1
      
    # If no match found, add new signature
    speaker_signatures.append(speaker_signature)
    return len(speaker_signatures)

try:
  with sd.RawInputStream(samplerate=samplerate,
                         blocksize=8000,
                         dtype="int16",
                         channels=1,
                         callback=audio_callback,
                         device=0):
    
    print("Listening... Press Ctrl+C to exit")

    while True:
      data = q.get()
      if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print("Text:", result["text"])
        if "spk" in result:
          print("Speaker:", handle_speaker(result["spk"]))
      else:
        partial = rec.PartialResult()
        print(partial)
except KeyboardInterrupt:
  print("\nDone Listening")
except Exception as e:
  print(f"Error: {e}")