import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from collections import deque

class RealtimeWhisperTranscriber:
    def __init__(self, model_path, window_length_s=5, overlap_s=1, sample_rate=16000):
        """
        Real-time Whisper transcriber with sliding window.
        
        Args:
            model_path: Path to Whisper model
            window_length_s: Length of each transcription window
            overlap_s: Overlap between windows for context
            sample_rate: Audio sample rate
        """
        self.window_length_s = window_length_s
        self.overlap_s = overlap_s
        self.sample_rate = sample_rate
        self.window_samples = int(window_length_s * sample_rate)
        self.overlap_samples = int(overlap_s * sample_rate)
        
        # Audio buffer using deque for efficient sliding window
        self.audio_buffer = deque(maxlen=self.window_samples)
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Initialize Whisper
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        print(f"Device set to use {device}")
        
        if device == "mps":
            torch.mps.empty_cache()
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Store model and processor for direct use
        self.model = model
        self.processor = processor
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Threading control
        self.recording = False
        self.processing = False
        
    def audio_callback(self, indata, frames, time, status):
        """Sounddevice callback for incoming audio data."""
        if status:
            print(f"Audio callback status: {status}")
        audio_data = indata[:, 0]  # Take first channel
        self.audio_queue.put(audio_data.copy())
    
    def audio_processor(self):
        """Process incoming audio data in separate thread."""
        while self.recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                self.audio_buffer.extend(audio_chunk)
                
                if len(self.audio_buffer) >= self.window_samples:
                    audio_window = np.array(list(self.audio_buffer))
                    
                    if not self.transcription_queue.full():
                        self.transcription_queue.put(audio_window.copy())
                    
                    for _ in range(min(len(self.audio_buffer), 
                                     self.window_samples - self.overlap_samples)):
                        self.audio_buffer.popleft()
                        
            except queue.Empty:
                continue
    
    def transcription_processor(self, config):
        """Process transcription requests in separate thread."""
        while self.processing:
            try:
                audio_window = self.transcription_queue.get(timeout=0.1)
                
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Use model and processor directly
                inputs = self.processor(
                    audio_window,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
                
                # Generate WITH timestamps but WITHOUT no_speech_threshold
                generate_kwargs = {
                    "language": config.get("language", "english"),
                    "task": "transcribe",
                    "num_beams": config.get("num_beams", 1),
                    "condition_on_prev_tokens": config.get("condition_on_prev_tokens", False),
                    "max_new_tokens": 400,
                    "return_timestamps": True,  # This enables timestamps
                    # Removed no_speech_threshold to avoid the bug
                }
                
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        **generate_kwargs
                    )
                
                # Decode WITH timestamps
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True,
                    decode_with_timestamps=True  # This preserves timestamp tokens
                )[0]
                
                # Simple silence detection based on transcription content
                if transcription.strip() and len(transcription.strip()) > 3:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {transcription}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
    
    def start_realtime_transcription(self, config=None):
        """Start real-time transcription from microphone."""
        if config is None:
            config = {
                "language": "english",
                "condition_on_prev_tokens": False,
                "num_beams": 1,
                # Removed no_speech_threshold but kept timestamps
            }
        
        print(f"Starting real-time transcription...")
        print(f"Window: {self.window_length_s}s, Overlap: {self.overlap_s}s")
        print(f"Sample rate: {self.sample_rate}Hz")
        print("Speak into your microphone. Press Ctrl+C to stop.")
        
        self.recording = True
        self.processing = True
        
        audio_thread = threading.Thread(target=self.audio_processor, daemon=True)
        transcription_thread = threading.Thread(
            target=self.transcription_processor, 
            args=(config,),
            daemon=True
        )
        
        audio_thread.start()
        transcription_thread.start()
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=1024,
                callback=self.audio_callback,
                device=0
            ):
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        
        self.recording = False
        self.processing = False
        
        audio_thread.join()
        transcription_thread.join()
        
        print("Transcription stopped.")

# Usage
if __name__ == "__main__":
    transcriber = RealtimeWhisperTranscriber(
        model_path="./models/whisper-large-v3",
        window_length_s=3,
        overlap_s=0.5,
        sample_rate=16000
    )
    
    config = {
        "language": "english",
        "condition_on_prev_tokens": False,
        "num_beams": 1,
        # Timestamps are enabled in the generation process
    }
    
    transcriber.start_realtime_transcription(config)