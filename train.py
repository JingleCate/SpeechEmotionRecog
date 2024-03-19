import tqdm

from dataset import SpeechDataset
from torch.utils.data import DataLoader

from dataproc import extract_features_of_batch
from logtool.logtool import log

@log("info", "Train the model.")
def train(
        batch_size=1,
        shuffle=True,
        num_workers=0,
        **kwargs
):  
    # load train data
    train_dataset = SpeechDataset(r"datasets\train\train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # load validation data
    val_dataset = SpeechDataset(r"datasets\val\val.csv")
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # load test data
    test_dataset = SpeechDataset(r"datasets\test\test.csv")
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    for idx, sample_batch in enumerate(train_dataloader):
        # path, label is a batch list.
        paths, label = sample_batch["path"], sample_batch["label"]
        feats = extract_features_of_batch(paths, is_print=True)
        print(feats)
        return

if __name__ == "__main__":
    train(batch_size=4)
    pass
