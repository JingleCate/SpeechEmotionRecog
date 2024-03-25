import torch
import torch.nn as nn
import torch.nn.functional as feats

LABElS = [
    "Null"
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
            in_channels: int=216
        ):
        super().__init__()
        self.is_print = is_print
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            # 卷积层 + Relu激活层
            nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            # Dropout 防止过拟合
            nn.Dropout(p=0.1),
            # # 池化降维
            # nn.MaxPool1d(2),

            # # 卷积层 + Relu激活层
            # nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding="same"),
            # nn.ReLU(),
            # nn.BatchNorm1d(64),
            # nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding="same"),
            # nn.ReLU(),

            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, padding="same"),
            nn.ReLU(),
            # 展平 + 全连接
            nn.Flatten(),
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
        for i in range(len(self.net)):
            x = self.net[i](x)
            if self.is_print:
                print(self.net[i], x.shape, x)
                print("----------------------------------------------------------------------------------")
        return x
    
# if __name__ == "__main__":
#     # model = SSRNetwork().to('cpu')
#     model = SSRNetwork()
#     print(model)