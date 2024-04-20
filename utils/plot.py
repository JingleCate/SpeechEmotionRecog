import glob
import os
import sys
from collections import Counter

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io.wavfile as swf
import sklearn.metrics as mt
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle


PROJECT_PATH = r"C:/Users/21552/Desktop/Main/Projects/SpeechMotionRecog"
sys.path.append(PROJECT_PATH)

from model.SSR import LABELS


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
    ax.set_title("Confusion matrix", fontsize=15)
    disp = mt.ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=LABELS)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 使用的sklearn中的默认值
        ax=ax,                        # 同上
        xticks_rotation="vertical",   
        values_format="d"               # 显示的数值格式
    )
    plt.savefig("assets/confusion_matrix.png", dpi=600)
    plt.show()


def plot_loss(path: str="../records/train_eval.csv"):
    df = pd.read_csv(path)
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

    plt.savefig("../assets/train_loss.png", dpi=600)


def plot_eval_coeffs(path: str="../records/train_eval.csv"):
    df = pd.read_csv(path)
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


    # resample per 20 points(downsample)
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
    plt.savefig("../assets/eval_coeffs.png", dpi=600)

def plot_radar(labels, preds):
    labels_count = Counter(labels)
    preds_count = Counter(preds)

    catagories = len(LABELS)
    labels = [ labels_count[i]  for i in range(catagories)]
    labels = np.concatenate((labels, [labels[0]]))      # 闭合
    preds = [ preds_count[i]  for i in range(catagories)]
    preds = np.concatenate((preds, [preds[0]]))         # 闭合
    angles = [ (i / catagories) * 2 * np.pi for i in range(catagories)]
    _angles = angles
    angles = np.concatenate((angles, [angles[0]]))      # 闭合
    names = LABELS
    names.append(LABELS[0])

    ax = plt.subplot(111, projection = 'polar')
    # plt.ylim(0, np.amax(labels))

    # ax.set_thetagrids(angles * 180 / np.pi, names)

    ax.plot(angles, labels, "b-", linewidth=1, label="Ground truth")
    ax.fill(angles, labels, 'b', alpha=0.2)         # fill the aere
    ax.plot(angles, preds, "r-", linewidth=1, label="Prediction")
    ax.fill(angles, preds, 'r', alpha=0.1)

    # ax.set_ylim(0, max(labels))
    ax.legend(bbox_to_anchor=(0.1, 0.1))
    plt.xticks(angles, names)
    ax.set_title("Radar map", fontsize=15)
    plt.savefig("assets/eval_radar.png", dpi=600)
    plt.show()

def plot_contrast():
    df = []
    df.append(pd.read_csv("../checkpoints/step-50/train_eval.csv"))
    df.append(pd.read_csv("../checkpoints/step-100/train_eval.csv"))
    df.append(pd.read_csv("../checkpoints/step-149/train_eval.csv"))

    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    
    #设置主标题
    fig.suptitle('Contrast Group', fontsize=20)
    
    step = 20
    titles = ["Accuracy", "Recall", "Precision", "f1-score"]
    locs = ["acc", "prec", "recall", "f1"]
    for i in range(2):
        axs[0, i].set_title(titles[i], fontsize=15)
        axs[0, i].plot(df[0]["epoch"][:500:step], df[0][locs[i]][:500:step], "C0", label="step 50")
        axs[0, i].plot(df[1]["epoch"][:500:step], df[1][locs[i]][:500:step], "C1", label="step 100")
        axs[0, i].plot(df[2]["epoch"][:500:step], df[2][locs[i]][:500:step], "C2", label="step 149")
        axs[0, i].legend()

    for i in range(2):
        axs[1, i].set_title(titles[i + 2], fontsize=15)
        axs[1, i].plot(df[0]["epoch"][:500:step], df[0][locs[i + 2]][:500:step], "C0", label="step 50")
        axs[1, i].plot(df[1]["epoch"][:500:step], df[1][locs[i + 2]][:500:step], "C1", label="step 100")
        axs[1, i].plot(df[2]["epoch"][:500:step], df[2][locs[i + 2]][:500:step], "C2", label="step 149")
        axs[1, i].legend()
    
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[:, 2:])
    ax.set_title("Training loss", fontsize=15)
    ax.plot(df[0]["epoch"][:500:step], df[0]["losses"][:500:step], "C0", label="step 50")
    ax.plot(df[1]["epoch"][:500:step], df[1]["losses"][:500:step], "C1", label="step 100")
    ax.plot(df[2]["epoch"][:500:step], df[2]["losses"][:500:step], "C2", label="step 149")
    ax.legend()

    axs[0, 2].xaxis.set_visible(False)
    axs[0, 3].xaxis.set_visible(False)
    axs[1, 2].xaxis.set_visible(False)
    axs[1, 3].xaxis.set_visible(False)
    axs[0, 2].yaxis.set_visible(False)
    axs[0, 3].yaxis.set_visible(False)
    axs[1, 2].yaxis.set_visible(False)
    axs[1, 3].yaxis.set_visible(False)
    
    plt.subplots_adjust(wspace = 0.24, hspace = 0.37)   #调整子图间距
    plt.savefig("../assets/contrast.png")
    plt.show()

def plot_bar(labels, preds):
    labels_count = Counter(labels)
    preds_count = Counter(preds)

    catagories = len(LABELS)
    labels = [ labels_count[i]  for i in range(catagories)]
    preds = [ preds_count[i]  for i in range(catagories)]
    bar_width = 0.35

    plt.bar(np.arange(len(LABELS)), labels, label = 'Groundtruth', color = 'C0', alpha = 0.7, width = bar_width)
    plt.bar(np.arange(len(LABELS)) + bar_width, preds, label= 'Prediction', color = 'C1', alpha =  0.7, width = bar_width)

    plt.xticks(np.arange(len(LABELS)) + bar_width / 2, LABELS)
    plt.xlabel('Classes')
    plt.ylabel('Numbers')
    plt.title('Bar chart of evaluation', fontsize=15)

    for x, y in enumerate(labels):
        plt.text(x, y+100, '%s' %y, ha = 'center')
    
    for x, y in enumerate(preds):
        plt.text(x + bar_width, y + 100, '%s' %y, ha = 'center')
    
    plt.legend()
    plt.savefig("assets/eval_bar.png", dpi=600)
    plt.show()


def plot_macro_roc(labels, probs: np.ndarray):
    """Plot roc curve function

    Parameters
    ----------
    `labels`: ArrayLike
        Sample labels
    `probs`: ArrayLike(n_samples, n_classes)
        Probabilities, 2-dim.
    """
    lb = LabelBinarizer().fit(labels)
    one_hot_labels = lb.transform(labels)

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, thres, roc_auc = dict(), dict(), dict(), dict()
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], thres[i] = mt.roc_curve(one_hot_labels[:, i], probs[:, i])
        roc_auc[i] = mt.auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0, 1, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(len(lb.classes_)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    
    # Average it and compute Macro AUC
    mean_tpr /= len(LABELS)
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = mt.auc(fpr["macro"], tpr["macro"])
    print("\n-------------------------------------------------------------------------------")
    print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")
    print("-------------------------------------------------------------------------------\n")

    # Micro
    fpr["micro"], tpr["micro"], _ = mt.roc_curve(one_hot_labels.ravel(), probs.ravel())
    roc_auc["micro"] = mt.auc(fpr["micro"], tpr["micro"])

    print("\n-------------------------------------------------------------------------------")
    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")
    print("-------------------------------------------------------------------------------\n")
    
    fig, ax = plt.subplots(figsize=(8, 7))

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="b",
        linewidth=2,
        alpha=1
    )
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="r",
        linewidth=2,
        alpha=1
    )
    
    colors = cycle(["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    for class_id, color in zip(range(len(LABELS)), colors):
        mt.RocCurveDisplay.from_predictions(
            one_hot_labels[:, class_id],
            probs[:, class_id],
            name=f"{LABELS[class_id]}(AUC = {roc_auc[class_id]:.2f})",
            lw=1.5,
            linestyle=":",
            alpha=0.8,
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 7),
        )
    ax.set(
        xlabel="FPS(False Positive Rate)",
        ylabel="TPS(True Positive Rate)",
    )
    ax.set_title("ROC on One-vs-Rest multiclass", fontsize=15)
    plt.savefig("assets/eval_roc.png", dpi=600)
    plt.show()

    

if __name__ == '__main__':
    # draw_waveform(path="datasets/archive/Actor_01/03-01-01-01-01-01-01.wav", savepth="assets/wvfm2.png", start=49, end=50)
    # draw_spectrogram(r"datasets\archive\Actor_01\03-01-01-01-01-01-01.wav", "assets/speech_spectrum")
    plot_loss()
    plot_eval_coeffs()
    plot_contrast()