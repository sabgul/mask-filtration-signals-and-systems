import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write
import scipy
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import math
import statistics as stat
from statsmodels.graphics.tsaplots import plot_acf


# plt.plot(temp_fr)
# plt.xlabel('CLIP')
# plt.show()

# 3: Extracts one second from maskoff_tone
def get_one_sec_maskoff():  #maskoff_tone.wav
    m_off_tone_audio = AudioSegment.from_file("../audio/maskoff_tone.wav", "wav")
    m_off_chunks = make_chunks(m_off_tone_audio, 1000)  # Make chunks of one sec

    for i, chunk in enumerate(m_off_chunks):  # Manually selected chunk of 1 sec
        if i == 2:
            chunk_name = "maskoff_tone_extract.wav".format(i)
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


# 4a: center clipping of individual frame
def center_clipping(frame):
    # compute max of absolute value
    maximum = max(frame.min(), frame.max(), key=abs)
    maximum = abs(maximum)

    thresh = maximum * 0.7 # threshold = 70% of maximum

    for i in range(0, len(frame)):
        # samples above 70% of max set to 1
        if frame[i] > thresh:
            frame[i] = 1
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


# 4b: auto-correlation
def autocorrelation(tone):
    threshold = 10
    frame = np.asarray(tone)

    def r(h):
        lag = ((frame[:(len(frame) - h)]) * (frame[h:])).sum()
        return round(lag, 3)

    x = np.arange(len(frame))
    autocorrelated = list(map(r, x))

    lag_index = (np.argmax(autocorrelated[threshold:]) + threshold)
    return autocorrelated, lag_index


# 4c: conversion of 'lag' to frequency
def lag_convert(lag):
    f_s = 16000
    converted = f_s / lag
    return converted


# 5: calculates dft and transforms it into form necessary for plotting of spectogram
def dft_log_transformation(frames):
    dft_frames = np.array(np.fft.fft(frames, 1024))
    log_frames = np.array(10*np.log10(np.square(np.abs([np.fft.fft(x, n=1024) for x in frames]))))

    to_plot = log_frames[0:99, 0:512]

    return dft_frames, to_plot


# 5: own implementation of dft
def my_dft(x_n, zero_pad):
    #zero padding
    curr_len = len(x_n)
    to_pad = zero_pad - curr_len
    np.pad(x_n, [(0, 0), (0, to_pad)], mode='constant')

    N = len(x_n)
    x_k = []
    const = 0
    for k in range(N):
        for n in range(N):
            exponent = -2j*np.pi*k*n*(1/N)
            const += x_n[n] * np.exp(exponent)
        x_k.append(const)
        const = 0

    return x_k


# 5: plots spectogram for mask-on / mask-off
def plot_spectogram(to_plot, label):
    plt.figure(figsize=(9, 5))
    plt.imshow(to_plot.T, extent=[0, 1, 0, 8000], aspect='auto', origin='lower')
    plt.gca().set_xlabel('time')
    plt.gca().set_ylabel('frequency')
    plt.gca().set_title(label)
    plt.colorbar()
    plt.show()


# 7: own implementation of idft
def my_idft(x_k, zero_pad):
    # zero padding
    curr_len = len(x_k)
    to_pad = zero_pad - curr_len
    np.pad(x_k, [(0, 0), (0, to_pad)], mode='constant')

    N = len(x_k)
    x_n = []
    const = 0
    for n in range(N):
        for k in range(N):
            exponent = 2j*np.pi*k*n*(1/N)
            const += x_k[k]*np.exp(exponent)
        const /= N
        x_n.append(const)
        const = 0

    return x_n


# 15: divides extracted second into frames
def get_frames_15(tone, samplerate):
    frames = []

    # unit: ms
    # 16000 frames / sec -> 16000 frames / 1000 ms -> 16 frames / ms
    frames_per_s = samplerate / 1000
    frames_per_s = int(frames_per_s)

    frame_length = 25 * frames_per_s # 25ms * 16 frames (16frames / ms)
    frame_offset = 10 * frames_per_s # prekryv
    frame_pos = 0

    while (frame_pos + frame_length) <= len(tone):
        frames.append(tone[frame_pos:frame_pos+frame_length])
        frame_pos += frame_offset

    return frames


def calculate_shift(moff_15_clipp, mon_15_clipp):
    # correlates maskoff x maskon
    tmp_off_on_corr = []
    for i in range(0, length):
        tmp_off_on_corr.append(scipy.signal.correlate(moff_15_clipp[i], mon_15_clipp[i]))

    # inverse correlation of maskon x maskoff
    tmp_on_off_corr = []
    for i in range(0, length):
        tmp_on_off_corr.append(scipy.signal.correlate(mon_15_clipp[i], moff_15_clipp[i]))

    off_on_corr = np.array(tmp_off_on_corr)
    shift_off_on = np.argmax(off_on_corr)
    on_off_corr = np.array(tmp_on_off_corr)
    shift_on_off = np.argmax(on_off_corr)

    lag_shift = int(np.max(off_on_corr) / 2)

    if shift_on_off < shift_off_on:
        shift = shift_on_off
    else:
        shift = shift_off_on

    return shift, lag_shift

if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # TASK 3 - preparation of data
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

    # Plots extracted frames
    # plt.plot(maskoff_frames[4], label = 'mask-off')
    # plt.plot(maskon_frames[4], label = 'mask-on')
    # plt.xlabel('samples')
    # plt.ylabel('y')
    # plt.legend(bbox_to_anchor=(1, 1),
    #            bbox_transform=plt.gcf().transFigure)
    # plt.savefig('./plot/extracted_frames.pdf')
    # plt.show()

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

    #f0 = fs/lag index

    moff_frames_np = np.array(maskoff_frames)
    for i in range(0, len(maskoff_frames)):
        # 4a: frame after center clipping
        frame_maskoff_clipped.append(center_clipping(moff_frames_np[i]))

        # 4b: frame auto-correlation
        maskoff_autocorrelated, lag = autocorrelation(frame_maskoff_clipped[i])
        frame_maskoff_autocorr.append(maskoff_autocorrelated)

        # 4c: conversion of lag to frequency == np.argmax / fs
        # creation of f_0 for mask-off tone
        f0_maskoff.append(lag_convert(lag))

    mon_frames_np = np.array(maskon_frames)
    for i in range(0, len(maskon_frames)):
        # 4a: frame after center clipping
        frame_maskon_clipped.append(center_clipping(mon_frames_np[i]))

        # 4b: frame auto-correlation
        maskon_autocorrelated, lag = autocorrelation(frame_maskon_clipped[i])
        frame_maskon_autocorr.append(maskon_autocorrelated)

        # 4c: conversion of lag to frequency == np.argmax / fs
        # creation of f_0 for mask-on tone
        f0_maskon.append(lag_convert(lag))

    # 4b data for protocol - mean, variance
    maskoff_mean = np.mean(f0_maskoff)
    maskoff_var = np.var(f0_maskoff)

    maskon_mean = np.mean(f0_maskon)
    maskon_var = np.var(f0_maskon)

    # ---------------------------------------------------------------------------
    # TASK 4 - output - In order to display data, uncomment whatever part necessary
    # ---------------------------------------------------------------------------

    # ----- PLOTS GRAPHS
    # # this value can be arbitrarily changed
    # plotted_frame = 2
    # fig, axs = plt.subplots(4, 1, figsize=(30, 20))
    # axs[0].plot(maskon_frames[plotted_frame])
    # axs[0].set_xlabel('time')
    # axs[0].set_ylabel('y')
    # axs[0].set_title('Frame (mask on)')
    #
    # axs[1].plot(frame_maskon_clipped[plotted_frame])
    # axs[1].set_xlabel('samples')
    # axs[1].set_ylabel('y')
    # axs[1].set_title('Center clipping 70% (mask on)')
    #
    # axs[2].plot(frame_maskon_autocorr[plotted_frame])
    # axs[2].set_xlabel('samples')
    # axs[2].set_ylabel('y')
    # axs[2].axvline(x=10, color='black', label="Threshold")
    # x = []
    # y = []
    # x.append((np.argmax(frame_maskon_autocorr[plotted_frame][10:]) + 10))
    # y.append(int(np.max(frame_maskon_autocorr[plotted_frame][10:])))
    # axs[2].stem(x, y, linefmt='red', label="Lag")
    # axs[2].legend(loc="upper right")
    # axs[2].set_title('Autocorrelation (mask on)')
    #
    # axs[3].plot(f0_maskoff, label='mask-off')
    # axs[3].set_xlabel('frames')
    # axs[3].set_ylabel('f0')
    # axs[3].plot(f0_maskon, label='mask-on')
    # axs[3].set_title('Base frequencies')
    # axs[3].legend(loc="upper right")
    # plt.savefig('./plot/similarity_check.pdf')
    # plt.show()
    # ----- END OF PLOT

    # print('Maskoff mean, maskoff variance:')
    # print(maskoff_mean, maskoff_var)
    #
    # print('Maskon mean, maskon variance:')
    # print(maskon_mean, maskon_var)

    # ---------------------------------------------------------------------------
    # TASK 5 - discrete fourier transform
    # ---------------------------------------------------------------------------
    # 5a calculate DFT spectrum from each frame with N = 1024
    np_maskon_frames = np.array(maskon_frames)
    dft_maskon, maskon_spec_plot = dft_log_transformation(np_maskon_frames)

    np_maskoff_frames = np.array(maskoff_frames)
    dft_maskoff, maskoff_spec_plot = dft_log_transformation(np_maskoff_frames)

    # 5b implement own function for DFT and compare it with FFT
    # implemented in function my_own_dft(signal, padding)

    # ---------------------------------------------------------------------------
    # TASK 5 - output - spectogram
    # ---------------------------------------------------------------------------

    # plot_spectogram(maskon_spec_plot, 'mask-on spectogram')
    # plot_spectogram(maskoff_spec_plot, 'mask-off spectogram')

    # ---------------------------------------------------------------------------
    # TASK 6 - frequency characteristics
    # ---------------------------------------------------------------------------

    #TODO: OPACNE DELENIE

    moff_frames = np.array(maskoff_frames)
    mon_frames = np.array(maskon_frames)

    dividend = np.array(dft_maskoff / dft_maskon)
    dividend = np.abs(dividend)

    dividend = dividend.T

    means = []
    for i in range(0, 512):
        means.append(np.array(np.average(dividend[i])))

    fchar = np.array(means)

    # logarithmic form is used only for plotting
    plot_frequency_char = np.array(10*np.log10(np.square(np.abs(fchar))))

    # plt.plot(plot_frequency_char)
    # plt.xlabel('Frequency characteristics')
    # plt.savefig('./plot/frequency_char.pdf', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()

    # ---------------------------------------------------------------------------
    # TASK 7 - filtration
    # ---------------------------------------------------------------------------

    fchar_half = fchar[:512]
    impulse_response = np.array(np.fft.ifft(fchar_half, 1024))
    impulse_response_half = impulse_response[:512]

    # 7b implement own function for IDFT
    # implemented in function my_own_idft(signal, padding)

    # plt.plot(impulse_response[:512])
    # plt.xlabel('Impulse response')
    # plt.savefig('./plot/impulse_response.pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()

    # ---------------------------------------------------------------------------
    # TASK 8 - mask simulation
    # ---------------------------------------------------------------------------

    samplerate_maskoff_sentence, maskoff_sentence = wavfile.read('../audio/maskoff_sentence.wav')
    samplerate_maskon_sentence, maskon_sentence = wavfile.read('../audio/maskon_sentence.wav')

    sim_maskon_sent = scipy.signal.lfilter(b=impulse_response_half.real, a=1, x=maskoff_sentence)
    sim_maskon_tone = scipy.signal.lfilter(b=impulse_response_half.real, a=1, x=maskoff_tone)

    # TODO normalizovat
    write("../audio/sim_maskon_sentence.wav", samplerate_maskon_tone, sim_maskon_sent)
    write("../audio/sim_maskon_tone.wav", samplerate_maskon_tone, sim_maskon_tone)

    # ------------------------------------------------
    # PLOTTING
    # #------------------------------------------------
    # plt.plot(maskoff_sentence, label='maskoff')
    # plt.plot(sim_maskon_sent, label='simulated mask')
    # plt.legend(bbox_to_anchor=(1, 1),
    #            bbox_transform=plt.gcf().transFigure)
    # plt.savefig('./plot/maskoff_sim_maskon.pdf', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
    # #------------------------------------------------
    # plt.plot(maskon_sentence, label='maskon')
    # plt.plot(sim_maskon_sent, label='simulated mask')
    # plt.legend(bbox_to_anchor=(1, 1),
    #            bbox_transform=plt.gcf().transFigure)
    # plt.savefig('./plot/maskon_sim_maskon.pdf', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
    # #------------------------------------------------
    # plt.plot(maskoff_sentence, label='maskoff')
    # plt.plot(maskon_sentence, label='maskon')
    # plt.plot(sim_maskon_sent, label='simulated mask')
    # plt.legend(bbox_to_anchor=(1, 1),
    #            bbox_transform=plt.gcf().transFigure)
    # plt.savefig('./plot/all_compared.pdf', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()

    # #------------------------------------------------

    # ---------------------------------------------------------------------------
    # TASK 10 - overlap - add
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # TASK 11 - mask simulation
    # ---------------------------------------------------------------------------
    # u 11 okienkove funkcie hodim na kazdy ramec pred dft

    # TODO odkomentovat
    # win_maskoff = np.array(maskoff_frames)
    # win_maskon = np.array(maskon_frames)
    # leng = len(win_maskon)
    # win_maskoff_copy = np.array(win_maskoff)
    # win_maskon_copy = np.array(win_maskon)
    # for i in range(0, leng):
    #     win_maskoff[i] = np.hanning(win_maskoff_copy[i])
    #     win_maskon[i] = np.hanning(win_maskon_copy[i])
    #
    # win_dft_maskoff, win_maskoff_spec_plot = dft_log_transformation(win_maskoff)
    # win_dft_maskon, win_maskon_spec_plot = dft_log_transformation(win_maskon)


    # ---------------------------------------------------------------------------
    # TASK 12 - mask simulation
    # ---------------------------------------------------------------------------
    # uloha s double lagom: zmenit hodnotu clipovania z 0.7 na 0.9
    # ak na double lag nenarazim, tak clipovanie zarovnavat len na 0 a 1
    # skusit na maskon aj maskoff


    # pri vlastnej dft: padding uz pred aplikovanim vzorca

    # ---------------------------------------------------------------------------
    # TASK 15 - frames shift
    # ---------------------------------------------------------------------------

    samplerate_maskoff_t_15, maskoff_tone_15 = wavfile.read('./extract/maskoff_tone_extract.wav')
    samplerate_maskon_t_15, maskon_tone_15 = wavfile.read('./extract/maskon_tone_extract.wav')

    maskoff_tone_15 = np.array(normalize_center(maskoff_tone_15))
    maskon_tone_15 = np.array(normalize_center(maskon_tone_15))

    # splits second into 25ms long frames
    maskoff_frames_15 = np.array(get_frames_15(maskoff_tone, samplerate_maskoff_t_15))
    maskon_frames_15 = np.array(get_frames_15(maskon_tone, samplerate_maskon_t_15))

    # fazovy posun
    maskoff_15_clipp = []
    maskon_15_clipp = []

    length = len(maskon_frames_15)

    maskoff_frames_15_tmp = np.array(maskoff_frames_15)
    maskon_frames_15_tmp = np.array(maskon_frames_15)

    # creates clipped frames
    for i in range(0, length):
        maskoff_15_clipp.append(center_clipping(maskoff_frames_15_tmp[i]))

    length2 = len(maskon_frames_15)
    for i in range(0, length2):
        maskon_15_clipp.append(center_clipping(maskon_frames_15_tmp[i]))

    moff_15_clipp = np.array(maskoff_15_clipp)
    mon_15_clipp = np.array(maskon_15_clipp)

    shift, lag_shift = calculate_shift(moff_15_clipp, mon_15_clipp)

    curr_len = len(maskon_frames_15[3])

    # currently we have 373 samples, which is equivalent to 373/16 = 23.31ms
    # we only need 320 which is equivalent to 20ms
    # faza nerouska, rouska vysla mensia, z nerouskoveho ramca odrezeme zaciatok a z ruskoveho koniec
    shifted_maskoff_tmp = []
    shifted_maskon_tmp = []
    for i in range(0, length):
        shifted_maskoff_tmp.append(maskoff_frames_15[i][lag_shift:])
        shifted_maskon_tmp.append(maskon_frames_15[i][0:curr_len-lag_shift])

    shifted_maskoff_long = np.array(shifted_maskoff_tmp)
    shifted_maskon_long = np.array(shifted_maskon_tmp)

    shifted_maskon_short = []
    shifted_maskoff_short = []
    for i in range(0, length):
        shifted_maskoff_short.append(np.array(shifted_maskoff_long[i][0:320]))
        shifted_maskon_short.append(np.array(shifted_maskon_long[i][0:320]))


    # final shifted second
    shifted_maskoff = np.array(shifted_maskoff_short)
    shifted_maskon = np.array(shifted_maskon_short)

    # ---------------------------------------------------------------------------
    # TASK 15 - plots the shift
    # ---------------------------------------------------------------------------
    # fig, axs = plt.subplots(2, 1, figsize=(18, 10))
    # axs[0].plot(maskoff_frames[3], label='maskoff')
    # axs[0].plot(maskon_frames[3], label='maskon')
    # axs[0].set_xlabel('samples')
    # axs[0].set_ylabel('y')
    # axs[0].set_title('Frames before shift')
    # axs[0].legend(loc="upper right")

    # axs[1].plot(shifted_maskoff[3], label='maskoff')
    # axs[1].plot(shifted_maskon[3], label='maskon')
    # axs[1].set_xlabel('samples')
    # axs[1].set_ylabel('y')
    # axs[1].set_title('Frames after shift')
    # axs[1].legend(loc="upper right")

    # plt.savefig('./plot/shift_before_after.pdf')
    # plt.show()

    # Applying this to the rest of the process
    # 15-5 dft
    np_maskon_shifted = np.array(shifted_maskon)
    dft_maskon_shifted, maskon_shifted_spec_plot = dft_log_transformation(np_maskon_shifted)

    np_maskoff_shifted = np.array(shifted_maskoff)
    dft_maskoff_shifted, maskoff_shifted_spec_plot = dft_log_transformation(np_maskoff_shifted)

    # 15-6 frequence char
    moff_shifted = np.array(shifted_maskoff)
    mon_shifted = np.array(shifted_maskon)

    dividend_shifted = np.array(dft_maskon_shifted / dft_maskoff_shifted)
    dividend_shifted = np.abs(dividend_shifted)

    dividend_shifted = dividend_shifted.T

    means_sh = []
    for i in range(0, 512):
        means_sh.append(np.array(np.average(dividend_shifted[i])))

    fchar_shifted = np.array(means_sh)

    # logarithmic form is used only for plotting
    plot_frequency_char_shifted = np.array(10*np.log10(np.square(np.abs(fchar_shifted))))

    # 15-7 impulse response
    fchar_half_shift = fchar_shifted[:512]
    impulse_response_shifted = np.array(np.fft.ifft(fchar_half_shift, 1024))
    impulse_response_half_shifted = impulse_response_shifted[:512]

    # 15-8 mask simulation

    sim_maskon_shifted_sent = scipy.signal.lfilter(b=impulse_response_half_shifted.real, a=1, x=maskoff_sentence)
    sim_maskon_shifted_tone = scipy.signal.lfilter(b=impulse_response_half_shifted.real, a=1, x=maskoff_tone)

    # TODO normalizovat
    write("../audio/sim_maskon_sentence_phase.wav", samplerate_maskon_tone, sim_maskon_sent)
    write("../audio/sim_maskon_tone_phase.wav", samplerate_maskon_tone, sim_maskon_tone)

    # --------- END OF CODE


