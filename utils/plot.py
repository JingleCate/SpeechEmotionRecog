import sys
import librosa
import librosa.display
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import glob
import os, sys
import matplotlib.pyplot as plt
import scipy.io.wavfile as swf

sys.path.append(r"C:\Users\21552\Desktop\Main\Projects\SpeechMotionRecog")
from model.single_sentence_recog import LABELS

def draw_waveform(path: str, savepth: str,  start: int, end: int):
    """draw a waveform of the given path

    Args:
        path (str): audio path
        savepth (str): waveform save path
        start (int): where to start the waveform, min 0 and max 100
        end (int): where to stop the waveform, min 0 and max 100
    """
    # 读取音频数据
    data, sampling_rate = librosa.load(path)
    # fragment cutting

    data = data[(start*data.size//100):(end*data.size//100)]
  # 绘制音频图像
    fig = plt.figure(figsize=(15, 5))
    librosa.display.waveshow(data, sr=sampling_rate)

    # 设置画布标题
    plt.title('sound waveform')

    plt.savefig(savepth)
    # 显示画布
    plt.show()


def draw_spectrogram(audio_path, save_path):
    """draw speech spectrum

    Args:
        audio_path (str): audio path
        save_path (str): image save path
    """
    sr, data = swf.read(audio_path)

    # parameters: step size = 10, windows size = 30
    # sr = 48kHz, just in second
    nstep = int(sr * 0.01)
    nwin = int(sr * 0.03)
    nfft = nwin

    # hamming window
    win = np.hamming(nwin)
    # right window side list
    nn = range(nwin, len(data), nstep)
    x = np.zeros((len(nn), nfft//2))

    for i,n in enumerate(nn):
        data_seg = data[(n - nwin) : n]         # one segment, size = window size
        z = np.fft.fft(win * data_seg)
        x[i, :] = np.log(np.abs(z[:(nfft//2)]))

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(x.T, interpolation='nearest', origin="lower", aspect="auto")

    plt.title('Spectrogram')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.savefig(save_path)
    plt.show()


def plot_coffusion_matrix(groundtruth: list, pred: list):
    # 使用sklearn工具中confusion_matrix方法计算混淆矩阵
    confusion_mat = mt.confusion_matrix(groundtruth, pred)
    print("confusion_mat.shape : {}".format(confusion_mat.shape))
    print("confusion_mat : {}".format(confusion_mat))
    
    # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
    fig, ax = plt.subplots(figsize=(6, 6.1))
    ax.set_title("Confusion matrix")
    disp = mt.ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=LABELS)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 使用的sklearn中的默认值
        ax=ax,                        # 同上
        xticks_rotation="vertical",   
        values_format="d"               # 显示的数值格式
    )
    plt.savefig("imgs/confusion_matrix.png", dpi=600)
    plt.show()


def plot_loss():
    df = pd.read_csv("checkpoints/3rd/train_eval.csv")
    # print(df.head(5))

    # df.plot()
    # prepare original data
    epoch = df["epoch"].tolist()
    lr = df["lr"].tolist()
    loss = df["losses"].tolist()
    acc = df["acc"].tolist()
    prec = df["prec"].tolist()
    recall = df["recall"].tolist()
    f1 = df["f1"].tolist()

    fig1, ax = plt.subplots()
    ax.plot(epoch, loss, label ="average loss")
    ax.set_yscale("log")
    ax.set_title("Average loss in training period(per epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avarage loss")
    ax.legend()

    plt.savefig("imgs/loss.png", dpi=600)


def plot_eval_coeffs():
    df = pd.read_csv("checkpoints/3rd/train_eval.csv")
    # prepare data
    epoch = df["epoch"].tolist()
    lr = df["lr"].tolist()
    loss = df["losses"].tolist()
    acc = df["acc"].tolist()
    prec = df["prec"].tolist()
    recall = df["recall"].tolist()
    f1 = df["f1"].tolist()


    fig2, axs = plt.subplots()
    # weighted average (per 20)
    # acc = [( acc[i] * 0.01 + acc[i - 1] * 0.99 ) if i >= 50 else acc[i] for i in range(len(acc)) ]


    # resample per 20 points
    epoch = [epoch[i] for i in range(len(epoch)) if (i % 20 == 0) ]
    acc = [acc[i] for i in range(len(acc)) if i % 20 == 0 ]
    prec = [prec[i] for i in range(len(prec)) if i % 20 == 0 ]
    recall = [recall[i] for i in range(len(recall)) if i % 20 == 0 ]
    f1 = [f1[i] for i in range(len(f1)) if i % 20 == 0 ]


    axs.plot(epoch, acc, "C0", label="accuracy")
    axs.plot(epoch, recall, "C1", label="recall")
    axs.plot(epoch, prec, "C2", label="precison")
    axs.plot(epoch, f1, "C4", label="f1 score")
    axs.set_title("Evaluation coefficients in validation set(per epoch)")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Value")

    axs.legend()
    plt.savefig("imgs/eval_coeffs.png", dpi=600)

# if __name__ == '__main__':
#     # draw_waveform(path="datasets/archive/Actor_01/03-01-01-01-01-01-01.wav", savepth="imgs/wvfm2.png", start=49, end=50)
#     draw_spectrogram(r"datasets\archive\Actor_01\03-01-01-01-01-01-01.wav", "imgs/speech_spectrum")
    # plot_loss()
    # plot_eval_coeffs()