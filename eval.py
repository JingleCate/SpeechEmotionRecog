import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechDataset
from model.SER import SER
from model.SSR import LABELS, SSRNetwork
from utils.logtool import log
from utils.plot import plot_coffusion_matrix, plot_radar, plot_bar


def get_eval_scores(labels, preds):
    report_dict = classification_report(labels, preds, labels=range(len(LABELS)), 
                                        output_dict=True, zero_division=np.nan)
    # Just select the macro method for dataset is well-distribute.
    correct_rate = report_dict["accuracy"]
    macro_prec = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    return (correct_rate, macro_prec, macro_recall, macro_f1)

def eval_model(batch_size: int = 4,
               shuffle: bool = True):
    # Dataset
    test_dataset = SpeechDataset(r"datasets/test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # Network
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    net_config = config["single_speech_recog_net"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SSRNetwork(is_print=False, in_channels=net_config["in_channels"], 
                     hidden_layer=net_config["hidden_layer"], padding=net_config["padding"],
                     maxpool_config=net_config["maxpool"], classes=net_config["classes"],
                     device=device).to(device)
    
    checkpoint = torch.load(net_config["checkpoint_path"])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # test dataset inference
    total_labels, total_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for idx, sample_batch in enumerate(tqdm(test_dataloader, desc="Processing bar: ")):
            # path, label is a batch list.
            # labels: tensor([2, 4, 6, 2]), values: 0~7 mapping for 8 locations
            paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)
            outputs = net(paths).to(device)
            values, predict = torch.max(outputs.data, dim=1)

            total_labels = torch.hstack((total_labels, labels.data))
            total_pred = torch.hstack((total_pred, predict.data))
    
    # detach and eval
    total_labels = total_labels.cpu().numpy().astype(int)
    total_pred = total_pred.cpu().numpy().astype(int)
    correct_rate, macro_prec, macro_recall, macro_f1 = get_eval_scores(total_labels, total_pred)

    print("\n-------------------------------------------------------------------------------")
    print("Acuracy: %.3f | macro_precision: %.3f | macro_recall: %.3f | macro_f1: %.3f " %
          (correct_rate, macro_prec, macro_recall, macro_f1))
    print("-------------------------------------------------------------------------------\n")
    
    plot_coffusion_matrix(total_labels, total_pred)
    plot_bar(total_labels, total_pred)
    plot_radar(total_labels, total_pred)
    

if __name__ == "__main__":
     eval_model(batch_size=4, shuffle=True)
    





    