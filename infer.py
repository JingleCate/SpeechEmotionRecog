import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import yaml

from model.SER import SER
from model.SSR import LABELS, SSRNetwork
from utils.logtool import log


def infer(file: str="datasets/archive/Actor_01/03-01-03-01-01-02-01.wav",
          is_raw: bool=True,
          output_onnx: bool=False):
    """Infer the specified file emotion labels from the raw test dataset or others.

    Parameters
    ----------
    `file`: str, optional
        Audio file path, recommended size is `2.5s-3.5s`, by default "datasets/archive/Actor_01/03-01-03-01-01-02-01.wav"
    `is_raw`: bool, optional
        Is raw test dataset file or not, by default True
    """
    # 载入配置
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    net_config = config["single_speech_recog_net"]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = SSRNetwork(is_print=False, in_channels=net_config["in_channels"], 
                     hidden_layer=net_config["hidden_layer"], padding=net_config["padding"],
                     maxpool_config=net_config["maxpool"], classes=net_config["classes"],
                     device=device).to(device)
    
    checkpoint = torch.load(net_config["checkpoint_path"])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    label = os.path.basename(file).split('.')[0].split('-')[2] if is_raw else "No label"
    prob = torch.nn.functional.softmax(net.forward(file), dim=1).detach().numpy()
    
    print("Prob: ", prob)
    pred = np.argmax(prob)
    print("ground truth: ", LABELS[int(label) - 1], "    prediction: ", LABELS[pred])
    
    # onnx ouput, warning: if you select this option, you need `pip install oxxn`
    if output_onnx:
        input_x = ["./datasets/archive/audio_speech_actors_01-24/Actor_24/03-01-08-01-02-01-24.wav",
                "./datasets/archive/audio_speech_actors_01-24/Actor_23/03-01-02-02-02-01-23.wav",
                "./datasets/archive/audio_speech_actors_01-24/Actor_22/03-01-06-01-02-01-22.wav",
                "./datasets/archive/audio_speech_actors_01-24/Actor_20/03-01-06-01-02-02-20.wav"
                ]

        torch.onnx.export(net,
                        input_x,
                        "./checkpoints/SSR_checkpoint.onnx",
                        input_names=["audio paths"],
                        output_names=["Label logits(no softmax)"],
                        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

if __name__ == "__main__":
    infer(output_onnx=False)
