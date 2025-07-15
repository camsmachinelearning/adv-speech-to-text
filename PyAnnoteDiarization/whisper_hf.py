import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class WhisperTranscriber:
    """
    Offline Whisper transcriber using Hugging Face Transformers with MPS support.
    Optimized for Apple Silicon devices with Metal Performance Shaders.
    """
    
    def __init__(self, model_path):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_id: Hugging Face model identifier
        """
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
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
        
        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=1,
        )
    
    def transcribe(self, audio_path, config):
        """
        Transcribe a local audio file.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional transcription parameters
            
        Returns:
            Transcription result
        """
        return self.pipe(str(audio_path), return_timestamps=True, generate_kwargs=config)
    

config = {
    "language": "english",
    "condition_on_prev_tokens": False,
    "num_beams": 1
}

transcriber = WhisperTranscriber(model_path="./models/whisper-large-v3")

transcription = transcriber.transcribe("sample.wav", config)

# print(transcription)
print(transcription['chunks'])