import os
import torch
import random

from datasets import Dataset, Audio
from transformers import Wav2Vec2FeatureExtractor
from torch import nn

from utils import load_filepaths


class AudioLabelLoader():
    def __init__(self, args, set_name):
        self.args = args
        self.set_name = set_name
    
    def get_data(self):
        metadata = load_filepaths(
            os.path.join(self.args.common.meta_file_folder, "{}.csv".format(self.set_name))
        )
        data = Dataset.from_dict(metadata).cast_column('audio', Audio())
        data = data.class_encode_column('emotion')
        return data



class DataCollator():
    def __init__(self, feature_extractor, num_labels, siamese_network=False):
        self.feature_extractor = feature_extractor
        self.num_labels = num_labels
        self.siamese_network = siamese_network
    
    def __call__(self, data):
        audio_arrays = []
        ground_truth = []
        emotion_dict = {}
        for idx, sample in enumerate(data):
            audio_arrays.append(sample['audio']['array'])
            ground_truth.append(sample['emotion'])
            current_emotion = str(sample['emotion'])
            if current_emotion not in emotion_dict:
                emotion_dict[current_emotion] = [idx]
            else:
                emotion_dict[current_emotion].append(idx)
        
        if not self.siamese_network:
            inputs = self.feature_extractor(
                audio_arrays, sampling_rate=self.feature_extractor.sampling_rate, padding='longest', return_tensors='pt'
            )

            return inputs, nn.functional.one_hot(torch.tensor(ground_truth), self.num_labels)
        else:
            inputs = self.feature_extractor(
                audio_arrays, sampling_rate=self.feature_extractor.sampling_rate, padding='longest', return_tensors='pt'
            )['input_values']
            
            anchor_audios = []
            negative_audios = []
            targets = []
            negative_emo = []
            for idx, target_emotion in enumerate(ground_truth):
                target = random.randint(0, 1)
                siamese_index = idx
                negative_sample_emo = target_emotion
                if target == 1:
                    siamese_index = random.choice(emotion_dict[str(target_emotion)])
                else:
                    # Get all type of emotion existed
                    exist_emo = list(emotion_dict.keys())
                    # Remove target emotion if it existed in list
                    if str(target_emotion) in exist_emo:
                        exist_emo.remove(str(target_emotion)) # List of negative emotion
                    
                    # Case: Only exist target emotion
                    if len(exist_emo) == 0:
                        target = 1
                        siamese_index = random.choice(emotion_dict[str(target_emotion)])
                    else:
                        negative_sample_emo = random.choice(exist_emo)
                        siamese_index = random.choice(emotion_dict[negative_sample_emo])
                anchor_audios.append(inputs[idx])
                negative_audios.append(inputs[siamese_index])
                targets.append(target)
                negative_emo.append(int(negative_sample_emo))

            return (torch.stack(anchor_audios),
                    torch.stack(negative_audios),
                    torch.tensor(targets),
                    nn.functional.one_hot(torch.tensor(ground_truth), self.num_labels),
                    nn.functional.one_hot(torch.tensor(negative_emo), self.num_labels)
                ) 