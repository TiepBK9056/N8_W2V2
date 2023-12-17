import torch
import numpy as np

from statistics import mean
from typing import Any, Dict, Union
from torch import nn
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor
)
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.models import (
    Wav2Vec2ForSpeechClassification,
    SiameseNetworkForSpeechClassification
)
from data_utils import DataCollator
from utils import plot


class Trainer(object):
    def __init__(self, args, train_dataset=None, valid_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        if not args.train.use_gpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = Wav2Vec2Config.from_pretrained(args.model.model_name_or_path)

        if args.model.use_siamese:
            self.model = SiameseNetworkForSpeechClassification.from_pretrained(args.model.model_name_or_path, self.args)
        else:
            self.model = Wav2Vec2ForSpeechClassification.from_pretrained(args.model.model_name_or_path, self.args)
        self.model = self.model.to(device=self.device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model.model_name_or_path)


    def train(self):
        global_steps = 0
        global_acc = 0
        optimizer = AdamW(self.model.parameters(), lr=self.args.optimizer.learning_rate, eps=self.args.optimizer.adam_epsilon)
        
        datacollator = DataCollator(self.feature_extractor,
                                    self.args.classifier.num_labels,
                                    siamese_network=self.args.model.use_siamese,
                                )
        train_dataloader = DataLoader(self.train_dataset, 
                                    collate_fn=datacollator,
                                    batch_size=self.args.train.batch_size,
                                    num_workers=8, 
                                    shuffle=True, 
                                    pin_memory=True,
                                )
        eval_dataloader = DataLoader(self.valid_dataset, 
                                    collate_fn=datacollator,
                                    batch_size=self.args.train.batch_size,
                                    num_workers=8, 
                                    shuffle=True, 
                                    pin_memory=True,
                                )

        self.model.train()

        train_iterator = trange(int(self.args.train.num_train_epochs), desc="Epoch")
        eval_loss = 0
        eval_acc = 0
        for it in train_iterator:
            losses = []
            siamese_losses = []
            classify_loss = []
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):
                optimizer.zero_grad()
                
                if not self.args.model.use_siamese: 
                    input_values = batch[0]['input_values'].to(device=self.device)
                    labels = batch[1].to(device=self.device, dtype=torch.float32)

                    
                    outputs = self.model(input_values=input_values, labels=labels, return_dict=True)
                else:
                    anchor_input = batch[0].to(device=self.device)
                    another_input = batch[1].to(device=self.device)
                    targets = batch[2].to(device=self.device)
                    anchor_labels = batch[3].to(device=self.device, dtype=torch.float32)
                    another_labels = batch[4].to(device=self.device, dtype=torch.float32)
                    outputs = self.model(
                        anchor_input=anchor_input,
                        another_input=another_input,
                        anchor_labels=anchor_labels,
                        another_labels=another_labels,
                        siamese_labels=targets,
                        return_dict=True,
                    )
                    siamese_losses.append(outputs.siamese_loss.item())
                    classify_loss.append(outputs.classify_loss.item())

                loss = outputs.loss
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if global_steps % self.args.train.eval_step == 0:
                    eval_loss, eval_acc = self.evaluation(self.model, eval_dataloader, self.device)
                    if eval_acc >= global_acc:
                        global_acc = eval_acc
                        self.config.save_pretrained(self.args.model.checkpoint_folder)
                        self.model.save_pretrained(self.args.model.checkpoint_folder)
                        self.feature_extractor.save_pretrained(self.args.model.checkpoint_folder)
                            
                    self.model.train()
                
                global_steps += 1
                if not self.args.model.use_siamese: 
                    epoch_iterator.set_postfix({'train_loss': round(mean(losses), 3), 'valid_loss': round(eval_loss, 3), 'valid_acc': round(eval_acc, 3)})
                else:
                    epoch_iterator.set_postfix({
                        'train_loss': round(mean(losses), 3),
                        'siamese_loss': round(mean(siamese_losses), 3),
                        'classify_loss': round(mean(classify_loss), 3),
                        'valid_loss': round(eval_loss, 3),
                        'valid_acc': round(eval_acc, 3)
                    })


    def evaluation(self, model, eval_loader, device='cpu'):
        model.eval()
        eval_losses = []
        y_ground_truth = []
        y_predict = []
        # head_attentions = []
        with torch.no_grad():
            for step, batch in enumerate(eval_loader):
                if not self.args.model.use_siamese: 
                    input_values = batch[0]['input_values'].to(device=device)
                    labels = batch[1].to(device=device, dtype=torch.float32)

                    outputs = self.model(input_values=input_values, labels=labels, return_dict=True)
                else:
                    anchor_input = batch[0].to(device=self.device)
                    another_input = batch[1].to(device=self.device)
                    targets = batch[2].to(device=self.device)
                    labels = batch[3].to(device=self.device, dtype=torch.float32)
                    another_labels = batch[4].to(device=self.device, dtype=torch.float32)
                    outputs = self.model(
                        anchor_input=anchor_input,
                        another_input=another_input,
                        anchor_labels=labels,
                        another_labels=another_labels,
                        siamese_labels=targets,
                        return_dict=True,
                    )
                # head_attentions.append(outputs.head_attention)
                loss = outputs.loss
                eval_losses.append(loss.item())
                
                if self.args.model.use_siamese: 
                    idxs_pred = torch.argmax(outputs.anchor_logits, dim=1)
                    y_predict = y_predict + list(idxs_pred.cpu().detach().numpy())
                else:
                    idxs_pred = torch.argmax(outputs.logits, dim=1)
                    y_predict = y_predict + list(idxs_pred.cpu().detach().numpy())
                idxs_gt = torch.argmax(labels, dim=1)
                y_ground_truth = y_ground_truth + list(idxs_gt.cpu().detach().numpy())

        # head_attentions = torch.cat(head_attentions, dim=0).cpu().detach().numpy()
        # plot(head_attentions, y_ground_truth)
        correct = (np.array(y_ground_truth) == np.array(y_predict))
        accuracy = correct.sum() / correct.size
        return mean(eval_losses), accuracy
        
