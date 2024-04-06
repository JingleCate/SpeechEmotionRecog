import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np

from utils.logtool import log
from model.SER import SER
from model.single_sentence_recog import LABELS

def infer(file: str="datasets/archive/Actor_01/03-01-03-01-01-02-01.wav",
          is_raw: bool=True,
          is_set: bool=False, 
          set_path: str="datasets/test.csv",
          is_weighted: bool=False):
    # 载入配置
    config = None
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    net_config = config["single_speech_recog_net"]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    ser = SER(net_config, infer=True, checkpoint_path="checkpoints/SSR_checkpoint.pt", device=device)
    # ser.eval()

    label = "non"
    path = file
    if is_set:
        test_dataset = pd.read_csv(set_path, sep=',', usecols= ["path", "emotion"])
        idx = 0
        # loc 区域选取，idx 为行， "path"为列
        path = test_dataset.loc[idx, "path"]
        # 原始标签为"01", "02" --> 1, 2
        label = test_dataset.loc[idx, "emotion"]
    if is_raw:          # raw test dataset audio
        path = file
        label = os.path.basename(path).split('.')[0].split('-')[2]
    
    ret, weight = ser.forward(path)
    ret, weight = ret.numpy(), np.array(weight)
    weight = weight / np.sum(weight)
    print("Raw output probability: ", ret, f"\n{len(weight)} segments weigths: ", weight)

    # weighted probability
    prob = 0
    if weight.shape[0] > 1:
        for i in range(len(weight)):
            prob += weight[i] * ret[i]
    else:
        # just one segment
        prob = ret[0]
    print("Weighted probability: ", prob)

    pred = np.argmax(prob)
    print("ground truth: ", LABELS[int(label) - 1], "    prediction: ", LABELS[pred])


if __name__ == "__main__":
    infer()
    # TODO 测试集分类准确率雷达图
    # TODO 只载入一个模型文件，即transformer的Wav2Vec
