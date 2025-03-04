import threading
import time
import pyaudio
import wave
import os
import tempfile

# Imports for macOS on‑device speech recognition via PyObjC
import objc
from Foundation import NSRunLoop, NSDate
import AVFoundation
import Speech

from pyannote.audio import Pipeline

# --- Mac On‑Device Speech Recognition Setup ---

def recognition_handler(result, error):
    if result is not None:
        # Get the best transcription from the result
        transcription = result.bestTranscription().formattedString()
        print("[Mac Transcription]", transcription)

def start_mac_speech_recognition():
    # Check authorization status (in production, request authorization if not granted)
    auth_status = Speech.SFSpeechRecognizer.authorizationStatus()
    if auth_status != Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
        print("Speech recognition not authorized. Please enable it in System Settings.")
        return

    recognizer = Speech.SFSpeechRecognizer.alloc().init()
    if not recognizer:
         print("Speech recognizer unavailable.")
         return

    request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
    audioEngine = AVFoundation.AVAudioEngine.alloc().init()
    inputNode = audioEngine.inputNode()
    format = inputNode.outputFormatForBus_(0)

    # Define the tap block that feeds audio into the recognition request
    def tap_block(buffer, when):
        request.appendAudioPCMBuffer_(buffer)
    inputNode.installTapOnBus_bufferSize_format_block_(0, 1024, format, tap_block)

    audioEngine.prepare()
    if not audioEngine.startAndReturnError_(objc.nil):
         print("Audio engine failed to start.")
         return

    # Start the recognition task with the result handler
    recognizer.recognitionTaskWithRequest_resultHandler_(request, recognition_handler)
    # Run the run loop indefinitely to keep receiving callbacks
    NSRunLoop.currentRunLoop().run()

# --- Diarization Audio Capture Setup ---

def diarize_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SEGMENT_SECONDS = 30  # Accumulate 30-second segments for diarization

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    print("Starting diarization audio capture...")
    # Load the diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    
    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * SEGMENT_SECONDS)):
            frames.append(stream.read(CHUNK))
        segment = b"".join(frames)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wf = wave.open(tmp.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(segment)
            wf.close()
            tmp_filename = tmp.name
        diarization = pipeline(tmp_filename)
        print("[Diarization]")
        for seg, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Speaker {speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
        os.remove(tmp_filename)

# --- Main: Run both processes concurrently ---

if __name__ == "__main__":
    threading.Thread(target=start_mac_speech_recognition, daemon=True).start()
    threading.Thread(target=diarize_audio, daemon=True).start()
    print("Running macOS on‑device transcription and diarization concurrently...")
    while True:
        time.sleep(1)