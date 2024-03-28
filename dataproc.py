import os
import librosa
import pandas as pd
import csv
import numpy as np
import random
import torch
import warnings

from tqdm import tqdm
from utils.deprecated import deprecated
from transformers import Wav2Vec2Processor, Wav2Vec2Model

@deprecated
def extract_single_feature(path: str, is_print: bool = False) -> torch.Tensor:
    """extract features from a single audio file.

    Parameters
    ----------
    `path` : str
        Audio file path.
    `is_print` : bool, optional
        whether to print the shape of the extracted features.

    Returns
    -------
    `feat` : torch.Tensor
       Extracted features.
    """
    # Just select 3s.
    X, sample_rate = librosa.load(path, sr=44100, offset=0, duration=3.0)
    # this audio's mfcc, 2-dims
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=39)
    feat = mfccs
    # print(feat.shape)
    # feat = np.mean(mfccs, axis=0)   # simplify to 1-dim
    
    # padding 0 if not 216 because duration 2.5s is just converting to 216 frames.
    if feat.shape[1] != 300:
        # (39, ) -> (39, 300)
        feat = np.pad(feat, ((0, 0), (0, 300 - feat.shape[1])), 'constant', constant_values=0)
    # print(feat.shape)
    if is_print:
            # mfccs.shape is like (39, 300)
            print("One audio file mfcc size: ", mfccs.shape)
            print("One audio file feat size: ", feat.shape)
            is_print = False
    return feat

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
            valid_size = test_size = int(len(files) - train_size)//2
            count = 0
            # shuffle the files
            random.shuffle(files)

            for file in files:
                count += 1
                if count <= train_size:
                    t_v_t.append('01')
                elif count <= train_size + valid_size:
                    t_v_t.append('02')
                else:
                    t_v_t.append('03')
                fpath = os.path.join(root, file)
                fpath = fpath.replace("\\", "/")
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
    header = ['path', 'channel', 'emotion',
              'e-intensity', 'statement', 'actor', 'split']
    train_path = os.path.join(output_path, 'train.csv')
    val_path = os.path.join(output_path, 'val.csv')
    test_path = os.path.join(output_path, 'test.csv')
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
            row[0].replace("\\", "/")
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

# TODO modify the duration and offset.
@deprecated
def extract_feature(set_path: str, output_path: str, catagory: str):
    """Extract features from audio files from a set of train set or validation set or test set.

    Args:
        set_path (str): path to the audio files to extract features from
        output_path (str): path to .csv file for extracted features 
        catagory (str, optional): Which set. Defaults to "Train set" | "Test set" | "Valid set".
    """
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    warnings.warn("This function has been deprecated, please not use it.", DeprecationWarning)

    features = []   # list of features
    with open(set_path, mode="r") as f:
        reader = csv.reader(f, delimiter=',')
        is_print = True
        for row in tqdm(reader):
            if row[0] == "path":
                continue
            audio_path = row[0]
            # sr = None means take original sample  rate
            X, sample_rate = librosa.load(
                audio_path, sr=44100, offset=0.5, duration=2.5)
            # this audio's mfcc, 2-dims
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
            feat = np.mean(mfccs, axis=0)   # simplify to 1-dim
            if is_print:
                # mfccs.shape is like (13, 216)
                print("One audio file mfcc size: ", mfccs.shape)
                # feat.shape is like (216, )
                print("Simplified feature size: ", feat.shape)
                is_print = False
            # padding 0 if not 216
            if len(feat) != 216:
                feat = np.pad(feat, (0, 216 - len(feat)),
                              'constant', constant_values=0)
                # print(feat, feat.shape)
            feat = feat.tolist()    # convert to list
            feat.append(row[2])     # Add lable
            features.append(feat)

    print(catagory + " features shape(label added): ",
          np.array(features).shape)  # features
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for feat in features:
            writer.writerow(feat)
    print("----------------- "+catagory+" ---------------------")

@deprecated
def extract_features_of_batch(paths: list, is_print: bool = False) -> torch.Tensor:
    """Extract features of a path list from a batch.

    Args:
        paths (list): Paths list.
        is_print (bool, optional): Whether to print the feature shape.. Defaults to False.

    Returns:
        FloatTensor: Extracted features.
    """
    features = []
    for path in paths:
        # return (39, 300)
        feat = extract_single_feature(path, is_print=is_print)
        features.append(feat)
        
    
    # 输出列向量
    ret =  torch.Tensor(np.array(features))
    # print(ret)
    return ret
        
def get_wav2vec2_exractor(model_name: str ="facebook/wav2vec2-base-960h",
                          saved_path: str="./checkpoints/pretrained"):
    if os.path.exists(saved_path + '/' + 'preprocessor_config.json'):
        processor = Wav2Vec2Processor.from_pretrained(saved_path)
        model = Wav2Vec2Model.from_pretrained(saved_path)
    else:
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.save_pretrained(saved_path)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        processor.save_pretrained(saved_path)
    return (processor, model)

def extractor(processor, model, paths: list, is_print: bool = False) -> torch.Tensor:
    feats = np.array([])
    for path in paths:
        X, sample_rate = librosa.load(path, sr=16000, offset=0, duration=3.0)
        feat = processor(X, return_tensors="np", sampling_rate=sample_rate, do_normalize=True).input_values
 
        # torch.Size([1, 48000]),  Segment is 5s, sampling_rate is 16000, so get 48000 samples by precessor.
        if feat.shape[1] != 48000:
            feat = np.pad(feat, ((0, 0), (0, 48000 - feat.shape[1])), 'constant', constant_values=0)
        if len(feats) == 0 :
            feats = feat
        else:
            feats = np.concatenate((feats, feat), axis=0)

    feats = torch.Tensor(np.array(feats))
    # torch.Size([1, 149, 768]) 249 represent time domain, 768 is general feature(is solidable)
    # 3 dim transpose [batch_size, 249, 768] -> [batch_size, 768, 149]
    ret = model(feats).last_hidden_state.permute(0, 2, 1)     
    if is_print:
        print(feats.shape)
        print(ret.shape)
    return ret


# if __name__ == '__main__':
#     split_dataset(ratio=0.8, path="./datasets/archive",
#                   output_path="./dataproc.csv")
#     output_each_set(loaded_path="./dataproc.csv", output_path="./datasets")


    # extract_single_feature("./datasets/archive/Actor_01/03-01-01-01-01-01-01.wav")
    # paths = ["./datasets/archive/Actor_01/03-01-01-01-01-01-01.wav",
    #          "./datasets/archive/Actor_01/03-01-01-01-01-02-01.wav"]
    # p1, p2 = get_wav2vec2_exractor()
    # extractor(p1, p2, paths, True)



# ------------------- Codes below are deprecated, please not use it. ------------------- 
    # extract_feature(set_path="./datasets/train.csv",
    #                 output_path="./datasets/train/feats.csv", catagory="Train set")
    # extract_feature(set_path="./datasets/val.csv",
    #                 output_path="./datasets/val/feats.csv", catagory="Valid set")
    # extract_feature(set_path="./datasets/test.csv",
    #                 output_path="./datasets/test/feats.csv", catagory="Test set")
