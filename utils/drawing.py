import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob
import os, sys
import matplotlib.pyplot as plt
import scipy.io.wavfile as swf

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


if __name__ == '__main__':
    # draw_waveform(path="datasets/archive/Actor_01/03-01-01-01-01-01-01.wav", savepth="imgs/wvfm2.png", start=49, end=50)
    draw_spectrogram(r"datasets\archive\Actor_01\03-01-01-01-01-01-01.wav", "imgs/speech_spectrum")