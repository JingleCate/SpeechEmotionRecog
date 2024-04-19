# 🚀 Single sentence emotion recognition(SSR) model

import os
import sys
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_PATH = r"C:/Users/21552/Desktop/Main/Projects/SpeechMotionRecog"
sys.path.append(PROJECT_PATH)

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Union

LABELS = [
     "neutral",
     "calm",  
     "happy", 
     "sad", 
     "angry",  
     "fearful",  
     "disgust", 
     "surprised"
]

class SSRNetwork(nn.Module):
    def __init__(self, 
            is_print = False,
            in_channels: int=768,
            hidden_layer: list=[128, 8],
            padding: str = "same",
            maxpool_config: dict = None,
            classes: int=8,
            device: str = "cpu",
        ):
        """Classificator of a single audio.
        
        Parameters
        ----------
        `is_print`: bool, optional
            Whether to print out the net info, by default False
        `in_channels`: int, optional
            Input channels, by default 39
        `hidden_layer`: list, optional
            Hidden layers channel list, by default [26, 13, 5, 1]
        `padding`: str, optional
            Conv1d padding mode, by default "same"
        `maxpool_config`: dict, optional
            Maxpool1d config, by default None
        `classes`: int, optional
            Classes of audio, by default 8
        """


        super(SSRNetwork, self).__init__()
        self.is_print = is_print
        self.in_channels = in_channels
        self.hidden_layer = hidden_layer
        self.padding = padding
        self.classes = classes

        self.sr = 16000

        # maxpool1d out length
        self.out_len1, self.out_len2 = self.compute_output_len(maxpool_config)
        self.processor, self.postprocessor = self.get_wav2vec2_exractor()
        # freeze the pretrained parameters
        # for param in self.processor.parameters():
        #     param.requires_grad = False
        # freeze the pretrained parameters.
        for param in self.postprocessor.parameters():
            param.requires_grad_(False)
        self.postprocessor.freeze_feature_encoder()

        self.device = device

        # Sliding window select the [batch_size, win, 768], step size is 149, win is 149(just in 3s)
        # 🎯 Althongh you can specify the step size there(50, 100, 150), however,
        # 🎯 according to my experiment, self.step = self.win has the best effect.
        # 🎯 if you need checkpoints of other steps ,you can download from 
        # 🎯 BaiduNetDisk https://pan.baidu.com/s/1GqLkkeJ-nS2RpxnairJP5A?pwd=xuwe
        self.step = 149             
        self.win = 149                      # just to 3s
        self.sliding_win = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_channels * self.win, self.hidden_layer[0]),   # (batch_size, win * in_chennels) -> (batch_size, hidden_layer[0])
            nn.PReLU(),
            nn.Linear(self.hidden_layer[0], self.hidden_layer[1]),          # (batch_size, hidden_layer[0]) -> (batch_size, self.hidden_layer[1])
            nn.PReLU()
        )   # -> (batches, hidden_layer[0])

        self.forward_layer = nn.Sequential(
            nn.Linear(self.hidden_layer[1], self.classes),
        )  # -> (batches, classes)
        # self.init_weight()

    def init_weight(self):
        for m in self.sliding_win:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.forward_layer:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, paths: Union[list, str]) -> torch.Tensor:
        # x <- (batch, frames, 768)
        paths = [paths] if not isinstance(paths, list) else paths
        x = self.extractor(paths)

        # sliding windows, (batches, win, 768)
        start = 0
        sx = 0
        logits = 0
        while True:
            if (start + self.win) <= x.shape[1]:
                seg = x[:, start:(start + self.win):, :]
                sx = self.sliding_win(seg)
            else:
                # last time, if not over win/2 , compute.
                if (start + self.win - x.shape[1]) < self.win / 2:
                    seg = x[:, (x.shape[1] - self.win)::, :]
                    sx = self.sliding_win(seg)
                break
            start += self.step
            logits += sx
        
        logits = self.forward_layer(logits)

        return logits
        
    
    def compute_output_len(self, maxpool_config):
        # there compute the maxpool layer output length.
        if maxpool_config is not None:
            self.in_len = maxpool_config["in_len"]
            self.pool_padding_size = maxpool_config["pool_padding_size"]
            self.dilation = maxpool_config["dilation"]
            self.kernal_size = maxpool_config["kernal_size"]
            self.stride = maxpool_config["stride"]
        else:
            self.in_len = [300, np.nan]
            self.pool_padding_size = [0, 0]
            self.dilation = [1, 1]
            self.kernal_size = [6, 6]
            self.stride = [1, 1]
        
        out_len1 = int(np.floor((self.in_len[0] + 2*self.pool_padding_size[0] 
                            - self.dilation[0] * (self.kernal_size[0] - 1) - 1) / self.stride[0] + 1))
        out_len2 = int(np.floor((out_len1 + 2*self.pool_padding_size[1] 
                            - self.dilation[1] * (self.kernal_size[1] - 1) - 1) / self.stride[1] + 1))
        
        return (out_len1, out_len2)
    
    # @staticmethod
    def get_wav2vec2_exractor(self):
        model_name = "facebook/wav2vec2-base-960h"
        saved_path = "checkpoints/init"
        
        
        if os.path.exists(saved_path + '/' + 'preprocessor_config.json'):
            processor = Wav2Vec2Processor.from_pretrained(saved_path)
            model = Wav2Vec2Model.from_pretrained(saved_path)
        else:
            model = Wav2Vec2Model.from_pretrained(model_name)
            model.save_pretrained(saved_path)
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            processor.save_pretrained(saved_path)
        return (processor, model)
    
    def extractor(self, paths):
        feats = self._get_feats_by_processor(paths)

        # torch.Size([1, 249, 768]) 249 represent time domain, 768 is general feature(is solidable)
        # 3 dim transpose [batch_size, 249, 768]
        ret = self.postprocessor(feats).last_hidden_state.to(self.device)   

        if self.is_print:
            print(feats.shape)
            print(ret.shape)
        return ret
    
    def _get_feats_by_processor(self, paths):
        max_length = 48000      # 3s
        temp_feats = []   # a list temply storing the feat(not padding)
        feats = torch.Tensor([]).to(self.device)    # padded feats
        for path in paths:
            # start from 0.0s and select all the audio
            X, sample_rate = librosa.load(path, sr=self.sr, offset=0)
            # torch.Size([1, 48000]),  if segment is 3s, sampling_rate is 16000, so get 48000 samples by precessor.
            feat = self.processor(X, return_tensors="pt", sampling_rate=self.sr).input_values.to(self.device)
            temp_feats.append(feat)
            max_length = feat.shape[1] if (feat.shape[1] > max_length) else max_length

        # Collate there by padding 0
        for feat in temp_feats:
            if feat.shape[1] < max_length:
                feat = F.pad(feat, (0, max_length - feat.shape[1], 0, 0), 'constant', value=0)
            # tranfer to feats
            if len(feats) == 0 :
                feats = feat
            else:
                feats = torch.cat((feats, feat), dim=0)
        # each data in batch have the same length(max_length)
        return feats


