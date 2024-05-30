import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


pi = np.pi

# Dot product
def dot_product(f, g, tmin, tmax):
    t = np.linspace(tmin, tmax, 1000)
    dt = t[1] - t[0]
    return np.dot(f(t), g(t))*dt

# Fourier image
def wave_fourier_image(func, t0, t1):
    image = lambda w: dot_product(func, lambda t: np.exp(-2*pi*1j*w*t), t0, t1)
    return np.vectorize(image)

# Plot graph
def visualize_result(wave_time, wave_img_abs, time):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))
    fig.suptitle('Музыкальное')
    
    for i in range(2):
        if i == 0:
            plt.sca(axes[i])
            t = np.linspace(0, time - 0.1, 10000)
            plt.plot(t, wave_time(t), color='black', linewidth=1.25, label='Wave')
            axes[i].set_xlabel('t')
            axes[i].set_ylabel('f(t)')
            axes[i].yaxis.label.set_rotation(0)
            plt.title('Waveform of Аккорд_20')
        else:
            plt.sca(axes[i])
            w = np.linspace(0, 5000, 5000)
            plt.plot(w, wave_img_abs(w), color='seagreen', linewidth=1.25, label='Wave image')
            axes[i].set_xlabel('w')
            axes[i].set_ylabel(r'$|\hat{f}(w)|$')
            axes[i].yaxis.label.set_rotation(0)
            plt.title('Fourier image of Аккорд_20')

        plt.legend(loc='upper right')
        plt.grid(color='black', linestyle='--', linewidth=0.5)


wave, sr = librosa.load('Аккорд_20.mp3')
wave_time = np.vectorize(lambda t: wave[int(t * sr)])
time = len(wave) / sr 
wave_image = wave_fourier_image(wave_time, 0, 0.15)
wave_image_abs = lambda t: abs(wave_image(t))
visualize_result(wave_time, wave_image_abs, time)
plt.show()





