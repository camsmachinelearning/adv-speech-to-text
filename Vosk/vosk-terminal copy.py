import os
import sys
import wave
import json
import queue
import numpy as np
import sounddevice as sd

from vosk import Model, KaldiRecognizer, SpkModel


SPK_MODEL_PATH = "model-spk"

if not os.path.exists(SPK_MODEL_PATH):
  print("SPK model not found")

  sys.exit(1)

model = Model(lang="en-us")
spk_model = SpkModel(SPK_MODEL_PATH)

samplerate = 16e3 # 16kHz

rec = KaldiRecognizer(model, samplerate)
rec.SetSpkModel(spk_model)

q = queue.Queue()

def audio_callback(data_in, frames, time, status):
  if status:
    print(status, file=sys.stderr)
  q.put(bytes(data_in))


# base speaker signature
base_speaker = [-1.110417,0.09703002,1.35658,0.7798632,-0.305457,-0.339204,0.6186931,
        -0.4521213,0.3982236,-0.004530723,0.7651616,0.6500852,-0.6664245,0.1361499,
        0.1358056,-0.2887807,-0.1280468,-0.8208137,-1.620276,-0.4628615,0.7870904,
        -0.105754,0.9739769,-0.3258137,-0.7322628,-0.6212429,-0.5531687,-0.7796484,
        0.7035915,1.056094,-0.4941756,-0.6521456,-0.2238328,-0.003737517,0.2165709,
        1.200186,-0.7737719,0.492015,1.16058,0.6135428,-0.7183084,0.3153541,0.3458071,
        -1.418189,-0.9624157,0.4168292,-1.627305,0.2742135,-0.6166027,0.1962581,
        -0.6406527,0.4372789,-0.4296024,0.4898657,-0.9531326,-0.2945702,0.7879696,
        -1.517101,-0.9344181,-0.5049928,-0.005040941,-0.4637912,0.8223695,-1.079849,
        0.8871287,-0.9732434,-0.5548235,1.879138,-1.452064,-0.1975368,1.55047,
        0.5941782,-0.52897,1.368219,0.6782904,1.202505,-0.9256122,-0.9718158,
        -0.9570228,-0.5563112,-1.19049,-1.167985,2.606804,-2.261825,0.01340385,
        0.2526799,-1.125458,-1.575991,-0.363153,0.3270262,1.485984,-1.769565,
        1.541829,0.7293826,0.1743717,-0.4759418,1.523451,-2.487134,-1.824067,
        -0.626367,0.7448186,-1.425648,0.3524166,-0.9903384,3.339342,0.4563958,
        -0.2876643,1.521635,0.9508078,-0.1398541,0.3867955,-0.7550205,0.6568405,
        0.09419366,-1.583935,1.306094,-0.3501927,0.1794427,-0.3768163,0.9683866,
        -0.2442541,-1.696921,-1.8056,-0.6803037,-1.842043,0.3069353,0.9070363,-0.486526]

speakers = []

speaker_counter = 1

def cosine_dist(x, y):
  nx = np.array(x)
  ny = np.array(y)
  return 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)

def difference(a, b):
  return abs(a-b)

def handle_speaker(spk_result):
  global speaker_counter
  speaker_cosine_distance = cosine_dist(base_speaker, spk_result)
  speaker_found = False
  this_speaker_id = 0
  for id, distance in speakers:
    if difference(speaker_cosine_distance, distance) < 0.15:
      speaker_found = True
      this_speaker_id = id
      break
  
  if not speaker_found:
    speakers.append([speaker_counter, speaker_cosine_distance])
    this_speaker_id = speaker_counter
    speaker_counter += 1

  return this_speaker_id


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

