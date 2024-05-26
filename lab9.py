import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa
from scipy.signal import find_peaks
file_path = 'nagranie.wav'
s, fs = sf.read(file_path, dtype='float32')
duration = len(s) / fs
channels = s.shape[1] if len(s.shape) > 1 else 1
# sd.play(s, fs)
#status = sd.wait()
plt.figure(figsize=(18, 6))
print('Audio signal:', s)
print('Sample rate:', fs)
print('Audio signal size:', len(s))
time = np.linspace(0, duration, len(s))
normalized_signal = (s[:, 0] if channels > 1 else s) / np.max(np.abs(s))
plt.plot(time * 1000, normalized_signal)
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.title('Audio Signal')
plt.show()

frame_length_ms = 10
frame_length_samples = int(frame_length_ms * fs / 1000)

def energy(frame):
    return np.sum(frame ** 2)

def zero_crossing_rate(frame):
    return ((frame[:-1] * frame[1:]) < 0).sum()

num_frames = len(s) // frame_length_samples
energies = []
zero_crossings = []

for i in range(num_frames):
    frame = s[i * frame_length_samples:(i + 1) * frame_length_samples]
    energies.append(energy(frame))
    zero_crossings.append(zero_crossing_rate(frame))

energies = np.array(energies) / max(energies)
zero_crossings = np.array(zero_crossings) / max(zero_crossings)

#maksima energii moga wskazywać na przykład na samogłoski, minima na przerwy w mowie lub szumy
#maksima przejść przez zero mogą wskazywać na głoski bezdźwięczne, minima na dźwięczne
#wysoka energia i niska liczba przejść przez zero mogą wskazywać na samogłoski
#niska energia i wysoka liczba przejść przez zero mogą wskazywać na głoski bezdźwięczne
#niska energia i niska liczba przejść przez zero mogą wskazywać na przerwy w mowie lub szumy
#wysoka energia i wysoka liczba przejść przez zero mogą wskazywać na intensywne dźwięki i hałasy

# Create time axis for the frames
frame_times = np.linspace(0, duration, num_frames)

# Visualization
plt.figure(figsize=(14, 8))

# Plot normalized original signal, energy, and zero-crossing rate
time = np.linspace(0, duration, len(s))
plt.plot(time * 1000, normalized_signal, label='Normalized Signal', alpha=0.6)
plt.plot(frame_times * 1000, energies, label='Energy', color='red')
plt.plot(frame_times * 1000, zero_crossings, label='Zero Crossing Rate', color='blue')

plt.xlabel('Time [ms]')
plt.ylabel('Normalized Value')
plt.title('Audio Signal with Energy and Zero Crossing Rate')
plt.legend()
plt.show()

#zad3
#
#63600 - i - harmoniczna
#13500 - e - harmoniczna
#34800 - u
#78000 - o
#88200 - a - harmoniczna
vowel_segment = s[88200:88200+2048]

window = np.hamming(len(vowel_segment))
windowed_signal = vowel_segment * window

# Compute Fourier Transform and logarithmic amplitude spectrum
yf = scipy.fftpack.fft(windowed_signal)
log_spectrum = np.log(np.abs(yf))

# Frequency axis
freqs = np.fft.fftfreq(len(windowed_signal), 1 / fs)

half_len = len(freqs) // 2
freqs = freqs[:half_len]
log_spectrum = log_spectrum[:half_len]

# Identify fundamental frequency (F0)
f0 = freqs[np.argmax(np.abs(yf[:half_len]))]
print(f"Fundamental frequency (F0): {f0:.2f} Hz")

# Linear Predictive Coding (LPC)
p = 20  # Order of LPC
lpc_coeffs = librosa.lpc(vowel_segment, order=p)

# Compute smoothed spectrum
a = np.zeros(len(vowel_segment))
a[:21] = lpc_coeffs
lpc_spectrum = np.log(np.abs(scipy.fftpack.fft(a)))
lpc_spectrum = lpc_spectrum * -1

# Adjust the LPC spectrum for better visualization
lpc_spectrum = lpc_spectrum[:half_len] - 5

# Find local maximas in the LPC spectrum
peaks, _ = find_peaks(lpc_spectrum, height=None, distance=50)
print("Maximas indices:", peaks)

# Extract formant frequencies
formant_frequencies = freqs[peaks]

# Visualization
plt.figure(figsize=(14, 8))
plt.plot(freqs, log_spectrum, label='Original Spectrum')
plt.plot(freqs, lpc_spectrum, label='LPC Spectrum', color='red')

# Highlight local maximas (formants) on the plot
for i, idx in enumerate(peaks):
    plt.plot(freqs[idx], lpc_spectrum[idx], 'ko')  # Mark local maximas with black circles
    plt.text(freqs[idx], lpc_spectrum[idx], f'F{i+1}', fontsize=12, color='black', ha='right')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Log amplitude')
plt.title('LPC Smoothed Spectrum')
plt.legend()
plt.xlim(0, 10000)
plt.ylim(-10, 3)  # Adjusting y-axis limits for better visualization
plt.grid(True)
plt.show()

# Formant extraction (F1, F2, etc.)
print(f"Formant frequencies: {formant_frequencies[:5]} Hz")