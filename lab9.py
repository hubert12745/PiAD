import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

file_path = 'audio.wav'
s, fs = sf.read(file_path, dtype='float32')
# sd.play(s, fs)
status = sd.wait()

print('Audio signal:', s)
print('Sample rate:', fs)
print('Audio signal size:', len(s))
time = np.linspace(0, len(s), num=len(s))
plt.plot(time/fs*1000, s)
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.title('Audio Signal')
plt.show()

frame_size = fs * 0.01
frame_size = int(frame_size)
frames = len(s) / frame_size
frames = int(frames)
print('Frame size:', frame_size)
print('Frames:', frames)

energy = []
zero_crossings = []

for i in range(0, len(s), frame_size):
    frame = s[i:i + frame_size]
    if len(frame) < frame_size:
        break
    energy.append(np.sum(frame ** 2))
    crossing = 0
    for j in range(len(frame) - 1):
        if frame[j] * frame[j + 1] < 0:
            crossing += 1
    zero_crossings.append(crossing)
print('Energy_size:', len(energy))
print('Zero crossings size:', len(zero_crossings))

newSize = frame_size*frames


energy = np.array(energy) / max(energy)
zero_crossings = np.array(zero_crossings) / max(zero_crossings)
