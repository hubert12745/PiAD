import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa
from scipy.signal import find_peaks


#zad1
def load_audio(file_path):
    signal, sample_rate = sf.read(file_path, dtype='float32')
    return signal, sample_rate


def plot_audio_signal(signal, sample_rate):
    duration = len(signal) / sample_rate
    channels = signal.shape[1] if len(signal.shape) > 1 else 1
    time = np.linspace(0, duration, len(signal))
    normalized_signal = (signal[:, 0] if channels > 1 else signal) / np.max(np.abs(signal))
    plt.figure(figsize=(18, 6))
    plt.plot(time * 1000, normalized_signal)
    plt.xlabel('Czas [ms]')
    plt.ylabel('Amplituda')
    plt.title('Sygnal audio')
    plt.show()


#zad2
def frame_analysis(signal, sample_rate, frame_length_ms=10):
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)

    def energy(frame):
        return np.sum(frame ** 2)

    def zero_crossing_rate(frame):
        return ((frame[:-1] * frame[1:]) < 0).sum()

    # maksima energii moga wskazywać na przykład na samogłoski, minima na przerwy w mowie lub szumy
    # maksima przejść przez zero mogą wskazywać na głoski bezdźwięczne, minima na dźwięczne
    # wysoka energia i niska liczba przejść przez zero mogą wskazywać na samogłoski
    # niska energia i wysoka liczba przejść przez zero mogą wskazywać na głoski bezdźwięczne
    # niska energia i niska liczba przejść przez zero mogą wskazywać na przerwy w mowie lub szumy
    # wysoka energia i wysoka liczba przejść przez zero mogą wskazywać na intensywne dźwięki i hałasy

    def plot_frame_analysis(signal, sample_rate, energies, zero_crossings, frame_times):
        duration = len(signal) / sample_rate
        time = np.linspace(0, duration, len(signal))
        normalized_signal = (signal[:, 0] if signal.ndim > 1 else signal) / np.max(np.abs(signal))
        plt.figure(figsize=(14, 8))
        plt.plot(time * 1000, normalized_signal, label='Znormalizowany sygnał', alpha=0.6)
        plt.plot(frame_times * 1000, energies, label='Energia', color='red')
        plt.plot(frame_times * 1000, zero_crossings, label='Przejścia przez zero', color='blue')
        plt.xlabel('Czas [ms]')
        plt.ylabel('Znormalizowane wartości')
        plt.title('Sygnał audio, energia oraz przejścia przez zero')
        plt.legend()
        plt.show()

    num_frames = len(signal) // frame_length_samples
    energies = []
    zero_crossings = []

    for i in range(num_frames):
        frame = signal[i * frame_length_samples:(i + 1) * frame_length_samples]
        energies.append(energy(frame))
        zero_crossings.append(zero_crossing_rate(frame))

    energies = np.array(energies) / max(energies)
    zero_crossings = np.array(zero_crossings) / max(zero_crossings)
    frame_times = np.linspace(0, len(signal) / sample_rate, num_frames)
    plot_frame_analysis(signal, sample_rate, energies, zero_crossings, frame_times)


#zad3, zad4
def analyze_vowel_segment(signal, sample_rate, start_sample, segment_length=2048):
    vowel_segment = signal[start_sample:start_sample + segment_length]
    window = np.hamming(len(vowel_segment))
    windowed_signal = vowel_segment * window

    yf = scipy.fftpack.fft(windowed_signal)
    log_spectrum = np.log(np.abs(yf))

    freqs = np.fft.fftfreq(len(windowed_signal), 1 / sample_rate)
    half_len = len(freqs) // 2
    freqs = freqs[:half_len]
    log_spectrum = log_spectrum[:half_len]
    plt.figure(figsize=(14, 8))
    plt.plot(freqs, log_spectrum)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.title('Widmo fragmentu samogłoski')
    plt.xlim(0, 10000)
    plt.grid(True)
    plt.show()
    f0 = freqs[np.argmax(np.abs(yf[:half_len]))]
    print(f"Częstotliwość podstawowa (F0): {f0:.2f} Hz")

    # Liniowe kodowanie predykcyjne (LPC) jest szeroko używane w przetwarzaniu sygnałów audio i kompresji mowy.
    # LPC działa poprzez oszacowanie współczynników filtru liniowego, który może przewidzieć bieżącą próbkę sygnału na podstawie przeszłych próbek.Przykładowe zastosowania LPC to synteza i kodowanie mowy.
    # Jest używane na przykład w bezstratnych kodekach dźwięku, takich jak FLAC.
    p = 20
    lpc_coeffs = librosa.lpc(vowel_segment, order=p)
    a = np.zeros(len(vowel_segment))
    a[:21] = lpc_coeffs
    widmoLPC = np.log(np.abs(scipy.fftpack.fft(a))) * -1
    widmoLPC = widmoLPC[:half_len] - 4.5

    peaks, _ = find_peaks(widmoLPC, height=None, distance=50)

    plt.figure(figsize=(14, 8))
    plt.plot(freqs, log_spectrum, label='Widmo właściwe')
    plt.plot(freqs, widmoLPC, label='Widmo LPC', color='red')

    for i, idx in enumerate(peaks):
        plt.plot(freqs[idx], widmoLPC[idx], 'ko')
        plt.text(freqs[idx], widmoLPC[idx], f'F{i + 1}', fontsize=12, color='black', ha='right')

    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.title('Wygładzone widmo LPC')
    plt.legend()
    plt.ylim(-10, 3)
    plt.grid(True)
    plt.show()


def main():
    file_path = 'audio.wav'
    signal, sample_rate = load_audio(file_path)
    print(f"Sample rate: {sample_rate} Hz")
    plot_audio_signal(signal, sample_rate)

    frame_analysis(signal, sample_rate)

    # 63600 - i - harmoniczna
    # 13500 - e - harmoniczna
    # 34800 - u
    # 78000 - o
    # 88200 - a - harmoniczna
    analyze_vowel_segment(signal, sample_rate, start_sample=88200)


if __name__ == "__main__":
    main()
