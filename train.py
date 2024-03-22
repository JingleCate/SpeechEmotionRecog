import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd


from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechDataset
from dataproc import extract_features_of_batch
from logtool.logtool import log
from model.single_sentence_recog import SSRNetwork, LABElS

@log("info", "Train the model.")
def train(
        epochs: int = 100,
        device: str = "cpu",
        batch_size=1,
        shuffle=True,
        num_workers=0,
        use_checkpoint: bool=False,
        checkpoint_path: str = "checkpoint/checkpoint.pth",
        learning_rate: float = 0.001,
        **kwargs
):  
    print("🔢 " + f"Using {device}.")
    # load train data
    train_dataset = SpeechDataset(r"datasets/train/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # load validation data
    val_dataset = SpeechDataset(r"datasets/val/val.csv")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # load test data
    test_dataset = SpeechDataset(r"datasets/test/test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # 定义模型
    net = SSRNetwork(is_print=False)
    net.to(device)
    # 定义损失策略和优化器
    criterion = nn.CrossEntropyLoss()
    # TODO 优化optimizer
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=1e-8)
    # TODO load checkpoint and why?
    if use_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epoch'] - 1
        loaded_loss = checkpoint['average_loss']

    # 训练10个epoch
    # TODO 验证集和测试集处理
    losses_ = []
    acc_ = []
    for epoch in range(epochs):
        running_loss = 0 # average loss
        counter = 0 # counter
        for idx, sample_batch in enumerate(train_dataloader):
            # path, label is a batch list.
            # labels: tensor([2, 4, 6, 2]), values: 0~7 mapping for 8 locations
            paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)
            # extract features, Simplified feature size:  (bacth_size, 216) 
            feats = extract_features_of_batch(paths, is_print=False).to(device) # to same device
            # print(feats, feats.shape)
            # return

            # Optimizing
            optimizer.zero_grad()
            # Forward propagation
            outputs = net(feats).to(device)
            # print(outputs, labels)
            # compute CRLoss
            loss = criterion(outputs, labels)

            # compute gradient, backpropagation
            loss.backward()
            # parameters update
            optimizer.step()

            running_loss += loss.item()    # add each iteration loss
            counter += 1
        # if idx % 150 == 0 and idx != 0:
        # Compute average loss each epoch
        # print('\n📸 [Epoch]: %d  🕐 [Iteration]: %5d  📉 [Average loss(each epoch)]: %.3f' % (epoch + 1, idx + 1, running_loss/counter))
        print('📸 [Epoch]: %d   📉 [Average loss(each epoch)]: %.3f' % (epoch + 1, running_loss/counter), end='\n')
        losses_.append(running_loss/counter)                
        # Validation
        correct_rate = 0
        if epoch % 10 == 9:
            total, right = 0, 0
            with torch.no_grad():
                for idx, sample_batch in enumerate(val_dataloader):
                    # path, label is a batch list.
                    # labels: tensor([2, 4, 6, 2])
                    paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)
                    # extract features, Simplified feature size:  (bacth_size, 216) 
                    feats = extract_features_of_batch(paths, is_print=False).to(device)

                    # (N,C,L), where N is the batch size, C is the number of features or channels, and L is the sequence length
                    outputs = net(feats)
                    values, predict = torch.max(outputs.data, dim=1)
                    total += labels.size(0)
                    right += (predict == labels).sum().item()
            correct_rate = right / total
            print("Accuracy of SSRNetwork on the validation set: %.3f %%" % (100 * correct_rate))
            acc_.append(correct_rate)
        if epoch % 100 == 99:
            # Save checkpoint each 100 epochs.
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "average_loss": running_loss/counter
            }, r"checkpoints/SSR_epoch_%d_acc_%.3f.pth" % (epoch + 1, correct_rate))

        counter = 0
        running_loss = 0
    df1 = pd.DataFrame({"losses": losses_})
    df2 = pd.DataFrame({"acc": acc_})
    df1.to_csv("./records/losses.csv", mode='a')
    df2.to_csv("./records/acc.csv", mode='a')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training model...")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, default is 100.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size, default is 1.")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for data loading, default is 0.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="whether to use checkpoint, default is False.") 
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint/checkpoint.pth", help="checkpoint path.")
    # learning rate
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate, default is 0.001.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.system("python dataproc.py")         # execute this program to get the dataset paths.
    train(device=device, 
          epochs=args.epochs, 
          batch_size=args.batch_size,
          num_workers=args.num_workers,
          use_checkpoint=args.use_checkpoint,
          checkpoint_path=args.checkpoint_path,
          learning_rate=args.learning_rate)
