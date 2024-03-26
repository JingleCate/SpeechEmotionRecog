import torch
import torch.nn as nn
import torch.nn.functional as feats
import numpy as np

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

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            # 卷积层 + Relu激活层
            # nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=3, padding="same"),
            # nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_layer[0], kernel_size=5, padding=self.padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_layer[0], out_channels=hidden_layer[1], kernel_size=5, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_config["kernal_size"][0], stride=maxpool_config["stride"][0]),

            # Dropout 防止过拟合
            nn.Dropout(p=0.1),

            # there should be (batch_size, hidden_layer[1], 198)
            nn.Conv1d(in_channels=hidden_layer[1], out_channels=hidden_layer[2], kernel_size=5, padding=self.padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_layer[2], out_channels=hidden_layer[3], kernel_size=5, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxpool_config["kernal_size"][1], stride=maxpool_config["stride"][1]),

            # 展平 + 全连接
            nn.Flatten(),
            nn.Linear(self.out_len2, self.classes),
            nn.Softmax(dim=1)
        )
        self.init_weight()


    def init_weight(self):
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.net(x) 这样写无法查看和调试中间参数形状
        if self.is_print:
            print(self.out_len1, self.out_len2)
        for i in range(len(self.net)):
            x = self.net[i](x)
            if self.is_print:
                print(self.net[i], x.shape)
                print("-----------------------------------------------------------------------------------------------------")
        return x
    
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
# if __name__ == "__main__":
#     # model = SSRNetwork().to('cpu')
#     model = SSRNetwork()
#     print(model)