import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from model.modeling_output import (
    SpeechClassifierOutput,
    SiameseNetworkOutput,
    HeadClassifierOutput
)
from model.losses import ContrastiveLoss, NPairLoss

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec2 classification task."""

    def __init__(self, args):
        super().__init__()
        self.a_fc1 = nn.Linear(args.classifier.hidden_size, 1)
        self.a_fc2 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(args.classifier.final_dropout)
        self.hidden_proj = nn.Linear(args.classifier.hidden_size, args.classifier.proj_size)
        self.out_proj = nn.Linear(args.classifier.proj_size, args.classifier.num_labels)
        
        if args.model.freeze_encoder:
            self.a_fc1.requires_grad = False
            self.a_fc2.requires_grad = False

    def forward(self, features):
        # features -> (B, T, 768)
        v = self.sigmoid(self.a_fc1(features))
        alphas = self.softmax(self.a_fc2(v).squeeze())
        res_att = (alphas.unsqueeze(2) * features).sum(dim=1)

        x = res_att
        x = self.dropout(self.relu(self.hidden_proj(x)))
        x = self.out_proj(x)

        return HeadClassifierOutput(
            logits=x,
            attention=res_att,
        )


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config, args):
        super(Wav2Vec2ForSpeechClassification, self).__init__(config)
        self.num_labels = args.classifier.num_labels
        self.config = config
        self.args = args

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(args)

        if args.model.freeze_feature_encoder:
            self.wav2vec2.freeze_feature_encoder()

        self.n_pair_loss = NPairLoss()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.args.classifier.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state

        head_output = self.classifier(hidden_states)
        logits = head_output.logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            y_ground_truth = torch.argmax(labels, dim=1)
            n_pair_loss = self.n_pair_loss(head_output.logits, y_ground_truth)

            entropy_loss = loss_fct(logits.view(-1, self.num_labels), labels)
            loss = self.args.siamese.coff_siamese_loss * n_pair_loss \
                 + (1 - self.args.siamese.coff_siamese_loss) * entropy_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            encoder_attentions=outputs.attentions,
            head_attention=head_output.attention
        )


class SiameseNetworkForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config, args):
        super(SiameseNetworkForSpeechClassification, self).__init__(config)
        self.num_labels = args.classifier.num_labels
        self.config = config
        self.coff_siamese_loss = args.siamese.coff_siamese_loss

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(args)

        if args.model.freeze_feature_encoder:
            self.wav2vec2.freeze_feature_encoder()
        
        if args.model.freeze_encoder:
            for name, param in self.wav2vec2.named_parameters():
                param.requires_grad = False
        
        self.contrastive_loss = ContrastiveLoss(args.siamese.margin)


    def sub_forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs.last_hidden_state

        head_output = self.classifier(hidden_states)
        logits = head_output.logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            encoder_attentions=outputs.attentions,
            head_attention=head_output.attention,
        )

    def forward(
        self,
        anchor_input,
        another_input,
        anchor_labels,
        another_labels,
        siamese_labels,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.args.classifier.use_return_dict

        anchor_output = self.sub_forward(anchor_input, labels=anchor_labels, return_dict=return_dict)
        anchor_attention = anchor_output.head_attention

        another_output = self.sub_forward(another_input, labels=another_labels, return_dict=return_dict)
        another_attention = another_output.head_attention

        classify_loss = anchor_output.loss + another_output.loss
        
        siamese_loss = self.contrastive_loss(anchor_attention, another_attention, siamese_labels)

        total_loss = 1.0e-3 * self.coff_siamese_loss * siamese_loss + (1 - self.coff_siamese_loss) * classify_loss
        
        return SiameseNetworkOutput(
            siamese_loss=siamese_loss,
            classify_loss=classify_loss,
            loss=total_loss,
            anchor_logits=anchor_output.logits,
            another_logits=another_output.logits,
            head_attention=anchor_attention
        )