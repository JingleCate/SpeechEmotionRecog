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
            in_channels: int=39,
            hidden_layer: list=[26, 13, 5, 1],
            padding: str = "same",
            maxpool_config: dict = None,
            classes: int=8
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


        super().__init__()
        self.is_print = is_print
        self.in_channels = in_channels
        self.hidden_layer = hidden_layer
        self.padding = padding
        self.classes = classes

        # maxpool1d out length
        self.out_len1, self.out_len2 = self.compute_output_len(maxpool_config)
        self.processor, self.postprocessor = self.get_wav2vec2_exractor()
        # freeze the pretrained parameters
        # for param in self.processor.parameters():
        #     param.requires_grad = False
        # for param in self.postprocessor.parameters():
        #     param.requires_grad_(False)
        self.postprocessor.freeze_feature_encoder()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            # 卷积层 + Relu激活层
            # nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=3, padding="same"),
            # nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_layer[0], kernel_size=5, padding=self.padding),
            nn.PReLU(),
            nn.Conv1d(in_channels=hidden_layer[0], out_channels=hidden_layer[1], kernel_size=5, padding=self.padding),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=maxpool_config["kernal_size"][0], stride=maxpool_config["stride"][0]),

            # Dropout 防止过拟合
            nn.Dropout(p=0.1),

            # there should be (batch_size, hidden_layer[1], 198)
            nn.Conv1d(in_channels=hidden_layer[1], out_channels=hidden_layer[2], kernel_size=5, padding=self.padding),
            nn.PReLU(),
            nn.Conv1d(in_channels=hidden_layer[2], out_channels=hidden_layer[3], kernel_size=5, padding=self.padding),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=maxpool_config["kernal_size"][1], stride=maxpool_config["stride"][1]),

            # 展平 + 全连接
            nn.Flatten(),
            nn.Linear(self.out_len2, self.classes),
            nn.PReLU()
            # nn.Softmax(dim=1) #  nn.Softmax is in CrossEntropyLoss, dont use CELoss, this will cause train error
        )
        self.init_weight()
        self.beta_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768*249, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.classes),
            nn.Softmax(dim=1)
        )

    def init_weight(self):
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, paths):
        x = self.extractor(paths)
        # x = self.net(x) 这样写无法查看和调试中间参数形状
        if self.is_print:
            print(self.out_len1, self.out_len2)
        for i in range(len(self.net)):
            x = self.net[i](x)
            if self.is_print:
                print(self.net[i], x.shape)
                print("-----------------------------------------------------------------------------------------------------")
        return x
        # if self.is_print:
        #     print(self.out_len1, self.out_len2)
        # for i in range(len(self.beta_net)):
        #     x = self.beta_net[i](x)
        #     if self.is_print:
        #         print(self.beta_net[i], x.shape)
        #         print("-----------------------------------------------------------------------------------------------------")
        # return x
        
    
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
        saved_path = "checkpoints/pretrained"
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
        feats = torch.Tensor([]).to(self.device)
        for path in paths:
            X, sample_rate = librosa.load(path, sr=16000, offset=0, duration=5.0)
            feat = self.processor(X, return_tensors="pt", sampling_rate=sample_rate).input_values.to(self.device)
    
            # torch.Size([1, 80000]),  Segment is 5s, sampling_rate is 16000, so get 80000 samples by precessor.
            if feat.shape[1] != 80000:
                feat = F.pad(feat, (0, 80000 - feat.shape[1], 0, 0), 'constant', value=0)
            if len(feats) == 0 :
                feats = feat
            else:
                feats = torch.cat((feats, feat), dim=0)
        # print(feats.shape)

        feats = torch.Tensor(feats).to(self.device)
        # torch.Size([1, 249, 768]) 249 represent time domain, 768 is general feature(is solidable)
        # 3 dim transpose [batch_size, 249, 768] -> [batch_size, 768, 249]
        ret = self.postprocessor(feats).last_hidden_state.permute(0, 2, 1).to(self.device)     
        if self.is_print:
            print(feats.shape)
            print(ret.shape)
        return ret

