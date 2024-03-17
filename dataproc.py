import os
import librosa
import pandas as pd
import csv
import numpy as np


def split_dataset(ratio: float, path: str, output_path: str):
    """Split the dataset into train and test sets. The output will be saved in .csv format, 
    which includes the path, type(train, test, validation), labels, etc.
    Args:
        ratio (float): training ratio. valid and test include the remaining files(average).
        path (str): dataset path
        output_path (str): csv output path
    """
    file_path, channel, emotion, eintensity, statement, actor = [], [], [], [], [], []
    train_ratio, valid_ratio, test_ratio = ratio, (1 - ratio)/2, (1 - ratio)/2
    # used for recording train(01), valid(02) and test(03)
    t_v_t = []

    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            # print(root, dirs, files, end="\n")
            train_size = int(len(files) * train_ratio)
            valid_size = test_size =  int(len(files) - train_size)//2
            count = 0
            for file in files:
                count += 1
                if count <= train_size:
                    t_v_t.append('01')
                elif count <= train_size + valid_size:
                    t_v_t.append('02')
                else:
                    t_v_t.append('03')
                fpath = os.path.join(root, file)
                file_path.append(fpath)
                fname, ext = os.path.splitext(file)
                split_fname = fname.split('-')

                '''file name is like 03-01-01-01-01-01-01.wav
                Filename identifiers
                    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
                    Vocal channel (01 = speech, 02 = song).
                    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
                    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
                    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
                    Repetition (01 = 1st repetition, 02 = 2nd repetition).
                    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
                '''
                assert len(split_fname) == 7
                channel.append(split_fname[1])
                emotion.append(split_fname[2])
                eintensity.append(split_fname[3])
                statement.append(split_fname[4])
                actor.append(split_fname[6])


    df = pd.DataFrame({
        'path': file_path,
        'channel': channel,
        'emotion': emotion,
        'e-intensity': eintensity,
        'statement': statement,
        'actor': actor,
        'split': t_v_t
    })
    df.to_csv(output_path)

def output_each_set(loaded_path: str, output_path: str):
    header = ['path', 'channel', 'emotion', 'e-intensity', 'statement', 'actor', 'split']
    train_path = os.path.join(output_path, 'train/train.csv')
    val_path = os.path.join(output_path, 'val/val.csv')
    test_path = os.path.join(output_path, 'test/test.csv')
    train_csv, val_csv, test_csv = [], [], []
    train_csv.append(header) 
    val_csv.append(header)
    test_csv.append(header)

    with open(loaded_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # one row is like ['', 'path', 'channel', 'emotion', 'e-intensity', 'statement', 'actor', 'split']
            if row[-1] == 'split':
                continue
            row.pop(0)  # remove firt line number
            if row[-1] == '01':
                train_csv.append(row)
            elif row[-1] == '02':
                val_csv.append(row)
            elif row[-1] == '03':
                test_csv.append(row)
    
    # print(val_csv)
    with open(train_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in train_csv:
            writer.writerow(row)
    with open(val_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in val_csv:
            writer.writerow(row)
    with open(test_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in test_csv:
            writer.writerow(row)


def extract_feature(set_path: str, output_path: str, catagory: str) :
    """Extract features from audio files from a set of train set or validation set or test set.

    Args:
        set_path (str): path to the audio files to extract features from
        output_path (str): path to .csv file for extracted features 
        catagory (str, optional): Which set. Defaults to "Train set" | "Test set" | "Valid set".
    """
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    features = []   # list of features
    with open(set_path, mode="r") as f:
        reader = csv.reader(f, delimiter=',')
        is_print = True
        for row in reader:
            if row[0] == "path":
                continue
            audio_path = row[0]
            X, sample_rate = librosa.load(audio_path, sr=44100, offset=0.5, duration=2.5)  # sr = None means take original sample  rate
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)    # this audio's mfcc, 2-dims
            feat = np.mean(mfccs, axis=0)   # simplify to 1-dim
            if is_print:
                print("One audio file mfcc size: ", mfccs.shape)  # mfccs.shape is like (13, 216)
                print("Simplified feature size: ", feat.shape)   # feat.shape is like (216, )
                is_print = False
            # padding 0 if not 216
            if len(feat)!= 216:
                feat = np.pad(feat, (0, 216 - len(feat)), 'constant', constant_values=0)
                # print(feat, feat.shape)
            features.append(feat)
    
    print(catagory + " features shape: ", np.array(features).shape) # features 
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for feat in features:
            writer.writerow(feat)
    print("----------------- "+catagory+" ---------------------")

if __name__ == '__main__':
    # split_dataset(ratio=0.8, path="./datasets/archive", output_path="./dataproc.csv")
    # output_each_set(loaded_path="./dataproc.csv", output_path="./datasets")
    extract_feature(set_path="./datasets/train/train.csv", output_path="./datasets/train/feats.csv", catagory="Train set")
    extract_feature(set_path="./datasets/val/val.csv", output_path="./datasets/val/feats.csv", catagory="Valid set")
    extract_feature(set_path="./datasets/test/test.csv", output_path="./datasets/test/feats.csv", catagory="Test set")
