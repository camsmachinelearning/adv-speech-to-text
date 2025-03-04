import objc
from Foundation import NSRunLoop
import AVFoundation
import Speech

last_transcription = ""

def recognition_handler(result, error):
    global last_transcription
    if result is not None:
        # Only print if the result is final or if it's a new update
        if result.isFinal() or result.bestTranscription().formattedString() != last_transcription:
            transcription = result.bestTranscription().formattedString()
            print("[Transcription]:", transcription)
            last_transcription = transcription

def start_mac_speech_recognition():
    # Check if speech recognition is authorized
    auth_status = Speech.SFSpeechRecognizer.authorizationStatus()

    recognizer = Speech.SFSpeechRecognizer.alloc().init()
    if recognizer is None:
        print("Speech recognizer unavailable.")
        return

    # Create a recognition request
    request = Speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
    # Set up audio engine for live audio capture
    audioEngine = AVFoundation.AVAudioEngine.alloc().init()
    inputNode = audioEngine.inputNode()
    format = inputNode.outputFormatForBus_(0)

    # Tap the audio buffer and feed it into the request
    def tap_block(buffer, when):
        request.appendAudioPCMBuffer_(buffer)
    inputNode.installTapOnBus_bufferSize_format_block_(0, 1024, format, tap_block)

    # Prepare and start the audio engine
    audioEngine.prepare()
    if not audioEngine.startAndReturnError_(objc.nil):
        print("Audio engine failed to start.")
        return

    # Start the speech recognition task
    recognizer.recognitionTaskWithRequest_resultHandler_(request, recognition_handler)

    # Keep the run loop running to receive callbacks
    NSRunLoop.currentRunLoop().run()

if __name__ == '__main__':
    start_mac_speech_recognition()