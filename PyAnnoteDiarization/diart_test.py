# Copyright (c) 2025 Kavi Gupta. All rights reserved.
# This software is provided under the MIT License.

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import MicrophoneAudioSource, FileAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter
from diart.models import PyannoteLoader, SegmentationModel, EmbeddingModel
import torch

device = torch.device("cpu")

if torch.backends.mps.is_available():
  device = torch.device("mps")




segmentation = SegmentationModel(PyannoteLoader("./models/segmentation-3.0.bin")).to(device) # pyannote/segmentation-3.0
embedding = EmbeddingModel(PyannoteLoader("./models/embedding.bin")).to(device) # pyannote/wespeaker-voxceleb-resnet34-LM

diarization_config = SpeakerDiarizationConfig(
  segmentation,
  embedding,
  device=torch.device("mps")
)

pipeline = SpeakerDiarization(diarization_config)
file = FileAudioSource("./sample.wav", sample_rate=16e3)
mic = MicrophoneAudioSource(device=0)
source = mic
inference = StreamingInference(pipeline, source, do_plot=True)
inference.attach_observers(RTTMWriter(source.uri, "./output/sample.rttm"))
prediction = inference()
