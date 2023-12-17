import torch

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.utils import ModelOutput

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    head_attention: torch.FloatTensor = None


@dataclass
class SiameseNetworkOutput(ModelOutput):
    siamese_loss: Optional[torch.FloatTensor] = None
    classify_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    anchor_logits: torch.FloatTensor = None
    another_logits: torch.FloatTensor = None
    head_attention: Optional[torch.FloatTensor] = None

@dataclass
class HeadClassifierOutput(ModelOutput):
    logits: torch.FloatTensor = None
    attention: torch.FloatTensor = None