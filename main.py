import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import math
import statistics as stat
from statsmodels.graphics.tsaplots import plot_acf

# TODO: RECORD MASKOFF TONE AGAIN


# 3: Extracts one second from maskoff_tone
def get_one_sec_maskoff():  #maskoff_tone.wav
    m_off_tone_audio = AudioSegment.from_file("../audio/maskoff_tone.wav", "wav")
    m_off_chunks = make_chunks(m_off_tone_audio, 1000)  # Make chunks of one sec

    for i, chunk in enumerate(m_off_chunks):  # Manually selected chunk of 1 sec
        if i == 2:
            chunk_name = "maskoff_tone_extract.wav".format(i)
            chunk.export(chunk_name, format="wav")
            chunk.export('./extract/' + chunk_name, format="wav")


# 3: Extracts one second from maskon_tone
def get_one_sec_maskon():
    m_on_tone_audio = AudioSegment.from_file("../audio/maskon_tone.wav", "wav")
    m_on_chunks = make_chunks(m_on_tone_audio, 1000)  # Make chunks of one sec
    # chooses most similar one by cross-correlation
    # Todo: cross-correlation np.correlate, np.corrcoef
    for i, chunk in enumerate(m_on_chunks):  # Manually selected chunk of 1 sec
        if i == 3: #3
            chunk_name = "maskon_tone_extract.wav".format(i)
            chunk.export('./extract/' + chunk_name, format="wav")


# 3a, 3b: normalizes and centers extracted second
def normalize_center(sound):
    sound = sound - np.mean(sound)
    sound = sound / np.abs(sound).max()

    return sound


# 3: divides extracted second into frames
def get_frames(tone, samplerate):
    frames = []

    # unit: ms
    # 16000 frames / sec -> 16000 frames / 1000 ms -> 16 frames / ms
    frames_per_s = samplerate / 1000
    frames_per_s = int(frames_per_s)

    frame_length = 20 * frames_per_s # 20ms * 16 frames (16frames / ms)
    frame_offset = 10 * frames_per_s # prekryv
    frame_pos = 0

    while (frame_pos + frame_length) <= len(tone):
        frames.append(tone[frame_pos:frame_pos+frame_length])
        frame_pos += frame_offset

    return frames


def center_clipping(frame):
    # compute max of absolute value
    maximum = max(frame.min(), frame.max(), key=abs)
    maximum = abs(maximum)

    thresh = maximum * 0.7 # threshold = 70% of maximum

    for i in range(0, len(frame)):
        # samples above 70% of max set to 1
        if frame[i] > thresh:
            frame[i] = 1
            print('teraz')
            print(i)
        # samples above 70% of -max set to -1
        elif frame[i] < (thresh * -1):
            frame[i] = -1
        # other samples are set to 0
        else:
            frame[i] = 0

    return frame

    # ------- PLOTS CLIPPED FRAME
    # plt.plot(frame)
    # plt.xlabel('clipped frame')
    # plt.show()


def autocorrelation(tone):
    threshold = 10
    frame = np.asarray(tone)

    def r(h):
        lag = ((frame[:(len(frame) - h)]) * (frame[h:])).sum()
        return round(lag, 3)

    x = np.arange(len(frame))
    autocorrelated = list(map(r, x))

    # ---- PLOTS AUTOCORRELATED FRAME
    # plt.axvline(x=10, color='black', label="Threshold")
    # # threshold at 10th sample
    # x = []
    # y = []
    # x.append((np.argmax(autocorrelated[10:]) + 10))
    # y.append(int(np.max(autocorrelated[10:])))
    # print(x, y)
    # plt.stem(x, y, linefmt='red', label="Lag")
    # plt.show()

    lag_index = (np.argmax(autocorrelated[threshold:]) + threshold)
    return autocorrelated, lag_index


def lag_convert(lag):
    f_s = 16000
    converted = f_s / lag
    return converted


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # TASK 3
    # ---------------------------------------------------------------------------

    # gets chunk of one sec from tone
    get_one_sec_maskon()
    get_one_sec_maskoff()

    # loads chunk of one sec to code
    samplerate_maskoff_tone, maskoff_tone = wavfile.read('./extract/maskoff_tone_extract.wav')
    samplerate_maskon_tone, maskon_tone = wavfile.read('./extract/maskon_tone_extract.wav')


    # 3a, 3b - centers and normalizes to range [-1;1]
    maskoff_tone = normalize_center(maskoff_tone)
    maskon_tone = normalize_center(maskon_tone)

    # 3: splits the tone into 20ms fragments
    maskoff_frames = get_frames(maskoff_tone, samplerate_maskoff_tone)
    maskon_frames = get_frames(maskon_tone, samplerate_maskon_tone)





    # --- PLOTS
    # fig, axs = plt.subplots(2)
    # axs[0].plot(maskoff_frames[2])
    # axs[0].plot(maskoff_frames[2], color = 'red')
    # axs[0].set_title('maskoff frame []')
    # #
    # axs[1].plot(maskon_frames[2])
    # axs[1].plot(maskon_frames[2], color = 'blue')
    # axs[1].set_title('maskon frame []')
    # #
    # # axs[2].plot(f0_maskoff)
    # # axs[2].plot(f0_maskon, color = 'red')
    # plt.show()
    # plt.savefig("random.png")


    # ---------------------------------------------------------------------------
    # TASK 4 - validity check
    # ---------------------------------------------------------------------------

    frame_maskoff_clipped = []
    frame_maskon_clipped = []

    frame_maskoff_autocorr = []
    frame_maskon_autocorr = []

    maskoff_autocorrelated = []
    maskon_autocorrelated = []

    f0_maskon = []
    f0_maskoff = []

    #f0 = fs/index lagu

    for i in range(0, len(maskoff_frames)):
        # 4a: frame after center clipping
        frame_maskoff_clipped.append(center_clipping(maskoff_frames[i]))

        # 4b: frame auto-correlation
        maskoff_autocorrelated, lag = autocorrelation(frame_maskoff_clipped[i])
        frame_maskoff_autocorr.append(maskoff_autocorrelated)

        # 4c: conversion of lag to frequency == np.argmax / fs
        # creation of f_0 for mask-off tone
        f0_maskoff.append(lag_convert(lag))
    # plt.plot(frame_maskoff_clipped[2])
    # plt.xlabel('CLIP')
    # plt.show()

    for i in range(0, len(maskon_frames)):
        # 4a: frame after center clipping
        frame_maskon_clipped.append(center_clipping(maskon_frames[i]))

        # 4b: frame auto-correlation
        maskon_autocorrelated, lag = autocorrelation(frame_maskon_clipped[i])
        frame_maskon_autocorr.append(maskon_autocorrelated)

        # 4c: conversion of lag to frequency == np.argmax / fs
        # creation of f_0 for mask-on tone
        f0_maskon.append(lag_convert(lag))

    #
    # temp_fr = maskoff_frames[4]
    # temp_fr = center_clipping(temp_fr)
    # plt.plot(temp_fr)
    # plt.xlabel('CLIP')
    # plt.show()

    # fig, axs = plt.subplots(2)
    # axs[0].plot(frame_maskoff_autocorr[2])
    # axs[0].plot(frame_maskoff_autocorr[2], color = 'red')
    # axs[0].set_title('autocorrelated maskoff frame')
    # #
    # axs[1].plot(frame_maskon_autocorr[2])
    # axs[1].plot(frame_maskon_autocorr[2], color = 'blue')
    # axs[1].set_title('autocorrelated maskon frame')
    #
    # axs[2].plot(f0_maskoff)
    # axs[2].plot(f0_maskon, color = 'red')
    # plt.show()

    # 4b - mean, variance
    maskoff_mean = np.mean(f0_maskoff)
    maskoff_var = np.var(f0_maskoff)

    maskon_mean = np.mean(f0_maskon)
    maskon_var = np.var(f0_maskon)

    print(maskoff_mean, maskoff_var)
    print(maskon_mean, maskon_var)

    # --- PLOTS
    # fig, axs = plt.subplots(3)
    # axs[0].plot(f0_maskoff)
    # axs[0].plot(f0_maskon, color = 'red')
    # plt.xlabel('f0')
    #
    # axs[1].plot(f0_maskoff)
    # axs[1].plot(f0_maskon, color = 'red')
    #
    # axs[2].plot(f0_maskoff)
    # axs[2].plot(f0_maskon, color = 'red')
    # plt.show()
    # plt.savefig("random.png")

    # ---------------------------------------------------------------------------
    # TASK 5 - discrete fourier transform
    # ---------------------------------------------------------------------------

    # 5a calculate DFT spectrum from each frame with N = 1024
    dft_maskoff = []
    dft_maskon = []

    # calculates dft for each frame
    #plt.imshow(np.array([np.fft.fft(x, n=1024) for x in maskoff_frames_1]))

    #--------- THIS WORKED BUT I HAVE CHANGED MY MIND
    # for i in range(0, len(maskoff_frames)):
    #     dft_maskoff.append(np.fft.fft(maskoff_frames[i], 1024))
    #     dft_maskon.append(np.fft.fft(maskon_frames[i], 1024))
    #
    # temp_dft = np.array(dft_maskoff)
    # temp_dft_on = np.array(dft_maskon)
    #
    # rows, cols = (len(maskoff_frames), 512)
    # log_dft_maskoff = [[0.0] * cols] * rows
    # log_dft_maskon = [[0.0] * cols] * rows
    #
    # temp_log = np.array(log_dft_maskoff)
    # temp_log_on = np.array(log_dft_maskon)


    # for i in range(0, len(maskoff_frames)):
    #     for j in range(0, 512):
    #         temp = abs(temp_dft[i][j])
    #         temp = temp ** 2
    #         temp = math.log((temp + 1e-20), 10)
    #         temp = temp * 10
    #         # if temp < -100:
    #         #     temp = 20
    #         temp_log[i][j] = temp
    #
    # for i in range(0, len(maskon_frames)):
    #     for j in range(0, 512):
    #         temp = abs(temp_dft_on[i][j])
    #         temp = temp ** 2
    #         temp = math.log((temp + 1e-20), 10)
    #         temp = temp * 10
    #         # if temp < -100:
    #         #     temp = 20
    #         temp_log_on[i][j] = temp
    # --------- END OF THIS WORKED BUT I HAVE CHANGED MY MIND

            # log_dft_maskoff[i][j] = 10.0 * math.log(((abs(dft_maskoff[i][j]) ** 2) +1e-20), 10)

    # --- PLOTS SPECTOGRAM

    #to_plot = np.array(log_dft_maskoff)
    # plt.imshow(to_plot.T, extent=[0, 1, 0, 8000], aspect='auto', origin='lower')


    # plt.figure(figsize=(9, 3))
    # plt.imshow(temp_log.T, extent=[0, 1, 0, 8000], aspect='auto', origin='lower')
    # plt.gca().set_xlabel('time')
    # plt.gca().set_ylabel('frequency')
    # plt.gca().set_title('mask-off spectogram')
    # plt.colorbar()
    # # cbar = plt.colorbar()
    # plt.show()

    # data z maskon_frames prezeniem dft, aplikujem vzorec s logaritmom a nasledne vykreslim spektogram
    # dft sa robi nad ustrednenym a normalizovanym signalom
    # ramce prevediem na spektogram pomocou imshow
    # dft robim nad maskoff_frames a maskon_frames

    # 5b implement own function for DFT and compare it with FFT

    # ---------------------------------------------------------------------------
    # TASK 6 -
    # ---------------------------------------------------------------------------
    # frekvencnu charakteristiku ziskame pre kazdy ramec
    # podiel vystupu pre nahravku s ruskou a bez maskon / maskoff
    # f_char = []
    # for i in range(0, len(dft_maskon)):
    #     f_char.append(temp_dft_on[i] / temp_dft[i])
    #
    # freq_char = np.array(f_char)
    # # plt.plot(freq_char)
    # # plt.xlabel('freq chars')
    # # plt.show()
    #
    # absolute_f_ch = abs(freq_char)
    # # spriemerujeme ju cez kazdy ramec, aby sme ziskali jednu -> priemerujeme len absolutne hodnoty
    # avg_freq_char = []
    #
    # for i in range(0, len(freq_char)):
    #     avg_freq_char.append(np.average(absolute_f_ch[i]))
    # freq_char = np.array(avg_freq_char)




    # plt.plot(freq_char)
    # plt.xlabel('average freq char')
    # plt.show()
    # vykreslit frekvencnu charakteristiku ako vykonove spektrum

    # ---------------------------------------------------------------------------
    # TASK 7 - filtration
    # ---------------------------------------------------------------------------
    # frekvencnu charakteristiku prevedieme na impulznu odozvu pomocou idft

    # ---------------------------------------------------------------------------
    # TASK 8 - mask simulation
    # ---------------------------------------------------------------------------

