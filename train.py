import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechDataset
from dataproc import extract_features_of_batch
from logtool.logtool import log
from model.single_sentence_recog import SSRNetwork, LABElS

@log("info", "Train the model.")
def train(
        device: str = "cpu",
        batch_size=1,
        shuffle=True,
        num_workers=0,
        epochs=100,
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
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练10个epoch
    for epoch in range(epochs):
        running_loss = 0 # average loss
        counter = 0 # counter
        for idx, sample_batch in enumerate(train_dataloader):
            # path, label is a batch list.
            # labels: tensor([2, 4, 6, 2]), values: 0~7 mapping for 8 locations
            paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)
            # extract features, Simplified feature size:  (bacth_size, 216) 
            feats = extract_features_of_batch(paths, is_print=False).to(device) # to same device
            # print(feats.shape)
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
        print('\n📸 [Epoch]: %d   📉 [Average loss(each epoch)]: %.3f' % (epoch + 1, running_loss/counter))
        running_loss = 0
        
        
        # Validation
        if epoch % 5 == 0:
            total, right = 0, 0
            with torch.no_grad():
                for idx, sample_batch in enumerate(val_dataloader):
                    # path, label is a batch list.
                    # labels: tensor([2, 4, 6, 2])
                    paths, labels = sample_batch["path"], (sample_batch["label"] - 1).to(device)
                    # extract features, Simplified feature size:  (bacth_size, 216) 
                    feats = extract_features_of_batch(paths, is_print=False).to(device)

                    outputs = net(feats)
                    values, predict = torch.max(outputs.data, dim=1)
                    total += labels.size(0)
                    right += (predict == labels).sum().item()
            print("Accuracy of SSRNetwork on the validation set: %.3f %%" % (100 * right / total))
        if epoch % 50 == 49:
            torch.save(net.state_dict(), r"model/SSR_epoch_%d.pth" % (epoch + 1))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device=device, batch_size=8, epochs=500)
