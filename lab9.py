import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa

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

# Apply Hamming window
window = np.hamming(len(vowel_segment))
windowed_signal = vowel_segment * window

# Compute Fourier Transform and logarithmic amplitude spectrum
yf = scipy.fftpack.fft(windowed_signal)
log_spectrum = np.log(np.abs(yf))

# Frequency axis
freqs = np.fft.fftfreq(len(windowed_signal), 1 / fs)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(freqs[:len(freqs)//2], log_spectrum[:len(log_spectrum)//2])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Log amplitude')
plt.title('Logarithmic Amplitude Spectrum')
plt.xlim(0, 10000)
plt.show()

# Identify fundamental frequency (F0)
f0 = freqs[np.argmax(np.abs(yf[:len(yf)//2]))]
print(f"Fundamental frequency (F0): {f0:.2f} Hz")

p = 20  # order of LPC
lpc_coeffs = librosa.lpc(vowel_segment, order=p)

a = np.zeros(len(vowel_segment))
a[:21] = lpc_coeffs

# Compute smoothed spectrum
lpc_spectrum = np.log(np.abs(scipy.fftpack.fft(a)))
lpc_spectrum = lpc_spectrum * -1

# Frequency axis

# Visualization
plt.figure(figsize=(14, 8))


plt.plot(freqs[:len(freqs)//2], log_spectrum[:len(log_spectrum)//2], label='Original Spectrum')
plt.plot(freqs[:len(freqs)//2], lpc_spectrum[:len(lpc_spectrum)//2], label='LPC Spectrum', color='red')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Log amplitude')
plt.title('LPC Smoothed Spectrum')
plt.legend()
plt.show()

# Formant extraction (F1 and F2)
formants = freqs[np.argsort(np.abs(a))[:2]]
print(f"Formant frequencies: F1 = {formants[0]:.2f} Hz, F2 = {formants[1]:.2f} Hz")