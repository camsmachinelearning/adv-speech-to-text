import sounddevice as sd
import numpy as np

# Print available devices
print("\nAvailable devices:")
for i, device in enumerate(sd.query_devices()):
    print(f"{i}: {device['name']}")

# Find devices by name instead of hardcoded IDs
devices = sd.query_devices()
input_device = 1#next((i for i, d in enumerate(devices) if "BlackHole" in d["name"]), None)
output_device = 2#next((i for i, d in enumerate(devices) if "Multi-Output" in d["name"]), None)

# Set the default input and output devices
sd.default.device = input_device, output_device

SAMPLE_RATE = 16000
print("Recording audio...")
recording = sd.rec(int(10 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')

sd.wait()
print("Recording complete")

# Amplify the audio by multiplying (adjust the factor as needed)
# amplified_recording = (recording * 20.0).astype(np.int16)

sd.play(recording, samplerate=SAMPLE_RATE)
sd.wait()
print("Playback complete")