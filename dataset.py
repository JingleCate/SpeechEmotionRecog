import pandas as pd

from torch.utils.data import Dataset, DataLoader


class SpeechDataset(Dataset):
    def __init__(self, path: str):
        """Speech dataset.

        Args:
            path (str): path to the speech dataset csv file.
        """
        self.dataset = pd.read_csv(path, sep=',', usecols= ["path", "emotion"])
        # print(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # loc 区域选取，idx 为行， "path"为列
        path = self.dataset.loc[idx, "path"]
        # 原始标签为"01", "02" --> 1, 2
        label = self.dataset.loc[idx, "emotion"]

        sample = {
            "path": path,
            "label": label
        }
        return sample

    def test(self, idx):
        pass

# if __name__ == "__main__":
#     d = SpeechDataset(r"datasets\test\test.csv")