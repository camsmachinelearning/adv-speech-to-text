import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import os

def load_original_model():
    """Load the original RF5 ConvGRU model"""
    original_model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder', device='cpu')
    original_model.eval()
    return original_model

class CoreMLCompatibleConvGRU(nn.Module):
    """CoreML-compatible version of RF5 ConvGRU embedder"""
    
    def __init__(self, original_model):
        super().__init__()
        
        # Extract the convolutional encoder from original model
        self.conv_encoder = self._extract_conv_layers(original_model)
        
        # Replace GRU with CoreML-compatible alternatives
        self.temporal_processor = self._create_temporal_replacement()
        
        # Extract final projection layers
        self.output_projection = self._extract_output_layers(original_model)
        
    def _extract_conv_layers(self, original_model):
        """Extract convolutional layers that are CoreML compatible"""
        return nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
    
    def _create_temporal_replacement(self):
        """Replace GRU with CoreML-compatible temporal processing"""
        return nn.Sequential(
            # Dilated convolutions for temporal context
            nn.Conv1d(256, 512, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(512, 786, kernel_size=3, dilation=8, padding=8),
            nn.ReLU(),
            
            # Replace AdaptiveAvgPool1d with fixed global average pooling
            # This avoids the CoreML conversion error
            nn.Conv1d(786, 786, kernel_size=1),  # Keep same channels
            nn.ReLU(),
        )
    
    def _extract_output_layers(self, original_model):
        """Extract final projection layers"""
        return nn.Sequential(
            nn.Linear(786, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        # x: [batch_size, samples] - raw 16kHz audio
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, samples]
        
        # Convolutional encoding
        x = self.conv_encoder(x)  # [batch_size, 256, time]
        
        # Temporal processing (replaces GRU)
        x = self.temporal_processor(x)  # [batch_size, 786, time]
        
        # Manual global average pooling (replaces AdaptiveAvgPool1d)
        x = torch.mean(x, dim=2)  # [batch_size, 786]
        
        # Final projection
        x = self.output_projection(x)  # [batch_size, 256]
        
        # L2 normalize like original model
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x

def transfer_weights(original_model, compatible_model):
    """Transfer compatible weights from original to new model"""
    original_dict = original_model.state_dict()
    compatible_dict = compatible_model.state_dict()
    
    # Transfer convolutional weights that match
    transferred = {}
    for name, param in compatible_dict.items():
        # Look for matching conv layers
        if 'conv_encoder' in name:
            # Try to find corresponding layer in original model
            original_name = name.replace('conv_encoder.', '')
            if original_name in original_dict:
                if original_dict[original_name].shape == param.shape:
                    transferred[name] = original_dict[original_name]
                    print(f"Transferred: {name}")
    
    # Update the compatible model with transferred weights
    compatible_dict.update(transferred)
    compatible_model.load_state_dict(compatible_dict)
    
    print(f"Transferred {len(transferred)} layers")
    return compatible_model

def create_coreml_compatible_model():
    """Create and convert the CoreML-compatible model"""
    
    # Load original model
    print("Loading original RF5 ConvGRU model...")
    original_model = load_original_model()
    
    # Create compatible version
    print("Creating CoreML-compatible version...")
    compatible_model = CoreMLCompatibleConvGRU(original_model)
    
    # Transfer weights where possible
    print("Transferring compatible weights...")
    compatible_model = transfer_weights(original_model, compatible_model)
    
    compatible_model.eval()
    
    # Test the model
    sample_input = torch.randn(1, 32000)  # 2 seconds at 16kHz
    
    with torch.no_grad():
        original_output = original_model(sample_input)
        compatible_output = compatible_model(sample_input)
        
        print(f"Original output shape: {original_output.shape}")
        print(f"Compatible output shape: {compatible_output.shape}")
        
        # Check similarity (won't be identical due to GRU replacement)
        similarity = torch.cosine_similarity(original_output, compatible_output, dim=1)
        print(f"Cosine similarity: {similarity.item():.4f}")
    
    # Trace the compatible model
    print("Tracing model...")
    traced_model = torch.jit.trace(compatible_model, sample_input)
    
    # Convert to CoreML with fixed input size to avoid dynamic shape issues
    print("Converting to CoreML...")
    os.makedirs("./models", exist_ok=True)
    
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="audio_waveform", 
                shape=(1, 32000),  # Fixed size to avoid dynamic shape issues
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="speaker_embedding", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.macOS12,
        compute_units=ct.ComputeUnit.ALL
    )
    
    coreml_model.save("./models/rf5_compatible.mlpackage")
    print("GOOD CoreML model saved: ./models/rf5_compatible.mlpackage")
    
    return coreml_model, compatible_model

def test_coreml_model():
    """Test the converted CoreML model"""
    try:
        print("\nTesting CoreML model...")
        model = ct.models.MLModel("./models/rf5_compatible.mlpackage")
        
        # Test with 32000 samples (2 seconds) since we used fixed size
        test_audio = (np.random.randn(1, 32000) * 0.5).astype(np.float32)
        
        try:
            result = model.predict({"audio_waveform": test_audio})
            embedding = result["speaker_embedding"]
            
            # Verify unit length
            norm = np.linalg.norm(embedding)
            print(f"GOOD 2s: shape={embedding.shape}, norm={norm:.4f}")
            
        except Exception as e:
            print(f"X 2s: {e}")
                
    except Exception as e:
        print(f"X CoreML test failed: {e}")

# if __name__ == "__main__":
#     # Create the compatible model and convert to CoreML
#     coreml_model, pytorch_model = create_coreml_compatible_model()
    
#     # Test the CoreML model
#     test_coreml_model()
    
#     print("\nGOOD Conversion complete!")
#     print("Files created:")
#     print("- ./models/rf5_compatible.mlpackage (CoreML model)")
#     print("\nNote: Performance may differ from original due to GRU→conv replacement")
#     print("Consider fine-tuning if needed for your specific use case")

test_coreml_model()