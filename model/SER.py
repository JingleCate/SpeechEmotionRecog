import sys
import torch
import librosa

import torch.nn as nn

sys.path.append("C:/Users/21552/Desktop/Main/Projects/SpeechMotionRecog")
from model.single_sentence_recog import LABELS, SSRNetwork

class SER(nn.Module):
    def __init__(self, 
                 snet_config: dict,
                 infer: bool=False,
                 checkpoint_path: str = "checkpoints/SSR_checkpoint.pt",
                 device: str="cpu"):
        super().__init__()
        self.sample_rate = 16000

        # single sentence recognition, cut into 3s.
        self.backbone = SSRNetwork(is_print=False,
                                   in_channels=snet_config["in_channels"],
                                   hidden_layer=snet_config["hidden_layer"],
                                   padding=snet_config["padding"],
                                   maxpool_config=snet_config["maxpool"],
                                   classes=snet_config["classes"],
                                   infer=infer,
                                   device=device
                                 )
        
        self.output_layer = nn.Softmax(dim=1)
        if infer:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.output_layer.parameters():
                param.requires_grad = False


        chpt = torch.load(checkpoint_path)
        self.backbone.load_state_dict(chpt["model_state_dict"])
        

    def forward(self, path: str)->torch.Tensor:
        is_seg, segments, weight = self.context_split(path)
        # return a output layer
        if is_seg:
            x = self.backbone.forward(segments, is_seg)
        else:
            x = self.backbone.forward(list([path]), is_seg)
        x = self.output_layer(x)        # cross softmax layer
        return (x, weight)

    def context_split(self, path):
        x, sr = librosa.load(path, sr=self.sample_rate)
        duration = librosa.get_duration(y=x, sr=sr)

        # segment the audio wave
        start, step = 0, 3
        segments = []
        is_segment = False
        lens = []
        # 大于3.5s就切片, 小于3.5s忽略最后0.x秒
        if duration > 3.5:
            is_segment = True
            for i in range(int(duration // step) + 1):
                if i < (duration // step):
                    segments.append(x[(start * sr):(start + step) * sr])
                    lens.append(step)
                else:
                    segments.append(x[(start * sr):])
                    lens.append(duration - start)
                start += step
        else:
            lens.append(duration)
        return (is_segment,  segments, lens)
        








        