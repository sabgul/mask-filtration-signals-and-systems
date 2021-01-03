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


# 5: plots spectogram for mask-on / mask-off
def plot_spectogram(to_plot, label):
    plt.figure(figsize=(9, 5))
    plt.imshow(to_plot.T, extent=[0, 1, 0, 8000], aspect='auto', origin='lower')
    plt.gca().set_xlabel('time')
    plt.gca().set_ylabel('frequency')
    plt.gca().set_title(label)
    plt.colorbar()
    plt.show()


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

    #f0 = fs/index lagu

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
# abs - priemer - log
# log pouzit iba na vykreslenie
    # 5a calculate DFT spectrum from each frame with N = 1024

    # na kazdy ramec aplikujem dft a log
    # dft_maskoff = 10*np.log10(np.square(np.abs(np.fft.fft(maskoff_frames, 1024))) + 1e-20)

    np_maskon_frames = np.array(maskon_frames)
    dft_maskon, maskon_spec_plot = dft_log_transformation(np_maskon_frames)

    np_maskoff_frames = np.array(maskoff_frames)
    dft_maskoff, maskoff_spec_plot = dft_log_transformation(np_maskoff_frames)

    # ---------------------------------------------------------------------------
    # TASK 5 - output - spectogram
    # ---------------------------------------------------------------------------

    #plot_spectogram(maskon_spec_plot, 'mask-on spectogram')
    #plot_spectogram(maskoff_spec_plot, 'mask-off spectogram')

    # data z maskon_frames prezeniem dft, aplikujem vzorec s logaritmom a nasledne vykreslim spektogram
    # dft sa robi nad ustrednenym a normalizovanym signalom
    # ramce prevediem na spektogram pomocou imshow
    # dft robim nad maskoff_frames a maskon_frames

    # 5b implement own function for DFT and compare it with FFT

    # ---------------------------------------------------------------------------
    # TASK 6 - frequence characteristics
    # ---------------------------------------------------------------------------
    # zevraj tu mam dat ten logaritums
    #
    # dft(s
    # rouskou) / dft(bez
    # rousky), pak
    # zprumerujes
    # vsechny
    # radky
    # vyslednyho
    # 2
    # d
    # pole
    # mezi
    # sebou
    # a
    # vysledny
    # 1
    # d
    # pole
    # o
    # 1024
    # itemech
    # vykreslis
    # jo
    # a
    # jeste
    # pred
    # tim
    # vykreslenim
    # to
    # mam
    # v
    # logaritmu

    # do idft davat len 512
    moff_frames = np.array(maskoff_frames)
    mon_frames = np.array(maskon_frames)

    dividend = np.array(dft_maskoff / dft_maskon)
    dividend = np.abs(dividend)

    dividend = dividend.T
    #podla lemlaka dat do idft cele pole 1024 a potom vziat len polovicu
    means = []
    for i in range(0, 512):
        means.append(np.array(np.average(dividend[i])))

    fchar = np.array(means)
    frequence_char = np.array(10*np.log10(np.square(np.abs(fchar))))

    plt.plot(frequence_char)
    plt.xlabel('Frequence characteristics')
    plt.show()

    #tr = np.arange(ramce1[index1].size) / fs
    #tr je osa x -> namiesto vzoriek mi tam da cas

    # deleni -> abs -> mean -> log
    # dleeni -> abs -> mena -> log z dalsi abs

    # ---------------------------------------------------------------------------
    # TASK 7 - filtration
    # ---------------------------------------------------------------------------
    # x od 0 do 8000
    # DO IFFT LEN POLOVICU
    # frequence characteristics is transformed to impulse response via IDFT
    # tak spriemerujem
    # 100
    # rámcov = dostanem
    # 1024
    # hodnôt
    # a
    # použijem
    # iba
    # 0: 512
    # a
    # dám
    # logaritmus
    # a
    # vykreslím?
    # vysledok treba hadzat do log ci ne?

    idk = fchar[:512]
    impulse_response = np.array(np.fft.ifft(idk, 1024))
    noje = impulse_response[:512]
    plt.plot(impulse_response[:512])
    plt.xlabel('impulse response')
    plt.show()

    # impulzni = np.array(np.fft.ifft(fchar))
    #
    # plt.plot(impulzni)
    # plt.xlabel('impulse response')
    # plt.show()
    # ---------------------------------------------------------------------------
    # TASK 8 - mask simulation
    # ---------------------------------------------------------------------------
# simulated = (b=inverse_fft.imag, a=1, x=maskoff_sentence) ? Alebo za b dať reálnu zložku tej spätnej fft? Ta mi tam sedí ale nie som si istý
# *simulated = signal.lfilter(...)
    samplerate_maskoff_sentence, maskoff_sentence = wavfile.read('../audio/maskoff_sentence.wav')
    samplerate_maskon_sentence, maskon_sentence = wavfile.read('../audio/maskon_sentence.wav')

    test = scipy.signal.lfilter(b=noje.real, a = 1, x = maskoff_sentence)
    write("simulated.wav", samplerate_maskon_tone, test)

    plt.plot(maskoff_sentence, label = 'maskoff')
    plt.plot(test, label = 'Simulated')
    #plt.plot(maskon_sentence, label = 'Real')

    plt.legend(bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure)
    plt.show()

    #lfilter(filtr, [1], nahravka)
    # import soundfile as sf
    #
    # data, samplerate = sf.read('existing_file.wav')
    # sf.write('new_file.flac', data, samplerate)




    # uloha s double lagom: zmenit hodnotu clipovania z 0.7 na 0.9
    # pri vlastnej dft: padding uz pred aplikovanim vzorca