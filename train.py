import os
import torch
import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from dataset import SpeechDataset
from dataproc import get_wav2vec2_exractor, extractor
from utils.logtool import log, myLogger
from model.single_sentence_recog import SSRNetwork, LABELS
from utils.plot import plot_coffusion_matrix

# Get a logger by filename.
logger = myLogger(filename=__file__.split('.')[0] + '.log').get_logger

@log("info", "Train the model.")
def train(
        epochs: int = 1000,
        device: str = "cpu",
        batch_size: int=16,
        shuffle: bool=True,
        num_workers: int=0,
        resume: bool=False,
        checkpoint_path: str = "checkpoint/checkpoint.pth",
        learning_rate: float = 1e-3,
        patience: int = 10,
        factor: float=0.95,
        net_config: dict = None,
        **kwargs
):  
    """Train the model.

    Parameters
    ----------
    `epochs`: int, optional
        Training epochs, by default 1000
    `device`: str, optional
        The device to be used for training model, by default "cpu"
    `batch_size`: int, optional
        Mini batch size, by default 16
    `shuffle`: bool, optional
        Whether to load dataset by shuffling, by default True
    `num_workers`: int, optional
        Whether to use subprocess when calling dataloader, by default 0
    `resume`: bool, optional
        Whether to resume training, by default False
    `checkpoint_path`: str, optional
        The model checkpoint path, by default "checkpoint/checkpoint.pth"
    `learning_rate`: float, optional
        Learning rate, by default 1e-3
    `patience`: int, optional
        The max amount of epoch about tolerating the average loss not descending, default is 100, by default 10
    `factor`: float, optional
        Learning rate decay ratio, by default 0.95
    `net_config`: dict, optional
        Net config dict, by default None
    
    Examples
    --------
    >>> train()
    """
    print("🔢 " + f"Using {device}.")
    # load train data
    train_dataset = SpeechDataset(r"datasets/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # load validation data
    val_dataset = SpeechDataset(r"datasets/val.csv")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # load test data
    test_dataset = SpeechDataset(r"datasets/test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 定义模型
    net = SSRNetwork(is_print=False,
                     in_channels=net_config["in_channels"],
                     hidden_layer=net_config["hidden_layer"],
                     padding=net_config["padding"],
                     maxpool_config=net_config["maxpool"],
                     classes=net_config["classes"],
                     )
    net.to(device)
    # 定义损失策略和优化器
    criterion = nn.CrossEntropyLoss()
    # 优化optimizer
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=1e-10, amsgrad=True)
    # lr scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-10)
    
    # load checkpoint and why?
    ep_temp = 0
    if resume:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        # net.eval()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ep_temp = checkpoint['epoch'] - 1
        loaded_loss = checkpoint['average_loss']
        logger.critical(f">>>>> Loaded model checkpoint from {checkpoint_path} at epoch {ep_temp + 1}.")
        logger.critical(f">>>>> Resume training from epoch { ep_temp + 1 }.")
    # print(learning_rate)
    logger.critical(f">>>>> Optimizer Curreent learning rate: { optimizer.state_dict()['param_groups'][0]['lr'] }")
    # logger.critical(f">>>>> Current learning rate: { scheduler.get_last_lr() }.")

    # 训练
    rec_epoch, losses_, acc_, prec_, recall_, f1_, lr_= [], [], [], [], [], [], []
    max_to_save, last_to_save = {}, {}        # save the max checkpoint until now.
    max_acc_ep = 0
    try:
        net.train()
        for epoch in range(epochs - ep_temp):
            running_loss = 0 # average loss
            counter = 0 # counter
            
            # each epoch training
            for idx, sample_batch in enumerate(tqdm(train_dataloader, desc="iters in epoch {}".format(epoch + ep_temp + 1))):
                # path, label is a batch list.
                # labels: tensor([2, 4, 6, 2]), values: 0~7 mapping for 8 locations
                paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)

                # feats = extractor(processor, postprocessor, paths, is_print=False).to(device) # to same device
                # print(feats, feats.shape) torch.Size([batch_size, 768, 249]) # 5s esgment.
                # return

                # Optimizing
                optimizer.zero_grad()
                # Forward propagation
                outputs = net(paths).to(device)
                # print(outputs, labels)
                # compute CRLoss
                loss = criterion(outputs, labels)
                # return

                # compute gradient, backpropagation
                loss.backward()
                # parameters update
                optimizer.step()

                running_loss += loss.item()    # add each iteration loss
                # print(loss.item())
                counter += 1
            # if idx % 150 == 0 and idx != 0:
            # Compute average loss each epoch
                
            # print('\n📸 [Epoch]: %d  🕐 [Iteration]: %5d  📉 [Average loss(each epoch)]: %.3f' % (epoch + 1, idx + 1, running_loss/counter))
            # logger.info('📸 [Epoch]: %d \t📉 [Average loss]: %.3f' % (epoch + 1 + ep_temp, running_loss/counter))
            # print('📸 [Epoch]: %d   📉 [Average loss(each epoch)]: %.3f' % (epoch + 1 + ep_temp, running_loss/counter), end='\n')

            losses_.append(running_loss/counter)
            rec_epoch.append(epoch + 1 + ep_temp) 
            scheduler.step(running_loss/counter)               

            # Validation
            correct_rate = 0
            if True:
                val_total_labels, val_total_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)
                net.eval()      # shut down the network batchnorm layer and dropout layer.
                with torch.no_grad():
                    for idx, sample_batch in enumerate(val_dataloader):
                        # path, label is a batch list.
                        # labels: tensor([2, 4, 6, 2])
                        paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)
                        # extract features, Simplified feature size:  (bacth_size, 216) 
                        # feats = extractor(processor, postprocessor, paths, is_print=False).to(device)

                        # (N,C,L), where N is the batch size, C is the number of features or channels, and L is the sequence length
                        outputs = net(paths)
                        # print(outputs.data, labels.data)
                        values, predict = torch.max(outputs.data, dim=1)
                        # print(values.data, predict.data)
                        # return

                        val_total_labels = torch.hstack((val_total_labels, labels.data))
                        val_total_pred = torch.hstack((val_total_pred, predict.data))

                        # total += labels.size(0)
                        # right += (predict == labels).sum().item()
                        # print(total, right)

                #               precision    recall  f1-score   support

                #            0       0.24      0.62      0.35        13
                #            1       0.24      0.25      0.24        16
                #            2       0.13      0.29      0.18        17
                #            3       0.50      0.05      0.09        21
                #            4       0.33      0.06      0.11        16
                #            5       0.27      0.14      0.18        22
                #            6       0.28      0.41      0.33        17
                #            7       0.36      0.23      0.28        22

                #     accuracy                           0.24       144
                #    macro avg       0.29      0.26      0.22       144
                # weighted avg       0.30      0.24      0.21       144
                # plot_coffusion_matrix(val_total_labels.cpu().numpy().astype(int),
                #                       val_total_pred.cpu().numpy().astype(int))
                # return
                report_dict = classification_report(val_total_labels.cpu().numpy().astype(int), 
                                                    val_total_pred.cpu().numpy().astype(int), 
                                                    labels=range(len(LABELS)), 
                                                    output_dict=True,
                                                    zero_division=np.nan)
                # Just select the macro method for dataset is well-distribute.
                correct_rate, macro_prec, macro_recall, macro_f1 = (report_dict["accuracy"],
                                                                    report_dict["macro avg"]["precision"],
                                                                    report_dict["macro avg"]["recall"], 
                                                                    report_dict["macro avg"]["f1-score"])

                logger.info("Epoch: %d\tls: %.3f | lr: %.3e | acc: %.3f | prec: %.3f | rc: %.3f | f1: %.3f\n" % 
                            (rec_epoch[-1], losses_[-1], scheduler.get_last_lr()[0], correct_rate, macro_prec, macro_recall, macro_f1))
                # logger.critical(f">>>>> Current learning rate(by scheduler): { scheduler.get_last_lr() }.")
                # Adam optimizer lr is changing while trainin(by gradient or gradient^2, but lr value is not changed), so such operations are not needed. 
                # logger.critical(f">>>>> Curreent learning rate: { optimizer.state_dict()['param_groups'][0]['lr'] }")
                # print("Accuracy of SSRNetwork on the validation set: %.3f %%" % (100 * correct_rate))
                
                acc_.append(correct_rate)
                prec_.append(macro_prec)
                recall_.append(macro_recall)
                f1_.append(macro_f1)
                lr_.append(scheduler.get_last_lr()[0])
                net.train()         # enable the network batchnorm layer and dropout layer.
            
            # beginning of training loop or accuracy increasing.
            if len(acc_) == 1:
                max_to_save = {
                    "epoch": epoch + 1 + ep_temp,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "average_loss": running_loss/counter,
                    "acc": correct_rate
                }
            if acc_[-1] > acc_[max_acc_ep]:
                max_acc_ep = len(acc_) - 1
                max_to_save = {
                    "epoch": epoch + 1 + ep_temp,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "average_loss": running_loss/counter,
                    "acc": correct_rate
                }
            last_to_save = {
                "epoch": epoch + 1 + ep_temp,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "average_loss": running_loss/counter,
                "acc": correct_rate
            }
            counter = 0
            running_loss = 0
    finally:
        df1 = pd.DataFrame({"epoch": rec_epoch,
                            "lr": lr_,
                            "losses": losses_,
                            "acc":acc_,
                            "prec":prec_,
                            "recall":recall_,
                            "f1":f1_})
        if not os.path.exists("./records"):
            os.makedirs("./records")
        # whether to insert columns headers or not.
        header = False if os.path.exists("./records/train_eval.csv") else True
        df1.to_csv("./records/train_eval.csv", mode='a', encoding="utf-8", header=header, index=False)
        
        # save checkpoint
        torch.save(max_to_save, r"checkpoints/SSR_epoch_%d_max_acc_%.3f.pth" % (max_to_save["epoch"], max_to_save["acc"]))
        torch.save(last_to_save, r"checkpoints/SSR_epoch_%d_last_acc_%.3f.pth" % (last_to_save["epoch"], last_to_save["acc"]))
        

if __name__ == "__main__":
    # 载入配置
    config = None
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train_config = config["train"]
    net_config = config["single_speech_recog_net"]

    parser = argparse.ArgumentParser(prog="train",
                                     description=">>> Train the model.",
                                     epilog=">>> For more information, please refer to the README.md file. The deault config file is config.yaml in directory of the project.")
    # parser.add_argument("plt", type=int, default=0, help="Test the positional parameters, default is 0.")
    # parser.add_argument("clt", type=int, default=0, help="Test the positional parameters, default is 0.")
    # All parameters are optional, if we need positional parameters, use "parser.add_argument('filename') "
    parser.add_argument("-e", "--epochs", type=int, default=train_config["epochs"], help="number of epochs, default is 100.")
    parser.add_argument("-b", "--batch_size", type=int, default=train_config["batch_size"], help="batch size, default is 16.")
    parser.add_argument("-n", "--num_workers", type=int, default=train_config["num_workers"], help="number of workers for data loading, default is 0.")
    parser.add_argument("-r", "--resume", type=bool, default=train_config["resume"], help="whether to resume training and use checkpoint, default is False.") 
    parser.add_argument("-chp","--checkpoint_path", type=str, default=train_config["checkpoint_path"], help="checkpoint path.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=train_config["learning_rate"], help="learning rate, default is 0.001.")
    parser.add_argument("-p", "--patience", type=int, default=train_config["patience"], help="The max amount of epoch about tolerating the average loss not descending, default is 100.")
    parser.add_argument("-f", "--factor", type=float, default=train_config["factor"], help="Learning rate decay ratio, default is 0.95.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train(device=device, 
          epochs=args.epochs, 
          batch_size=args.batch_size,
          num_workers=args.num_workers,
          resume=args.resume,
          checkpoint_path=args.checkpoint_path,
          learning_rate=args.learning_rate,
          patience=args.patience,
          factor=args.factor,
          net_config=net_config)
