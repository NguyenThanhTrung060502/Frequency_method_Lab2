import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

pi = np.pi

# Rectangle function
def rectangle(t, a, b):
    if abs(t) <= b:
        return a
    return 0

# Triangle function
def triangle(t, a, b):
    if abs(t) <= b:
        return a - abs(a*t/b)
    return 0

# sinc function
def sinc(t, a, b):
    return a*np.sinc(b*t)

# Gauss function
def gauss(t, a, b):
    return a*np.exp(-b*t**2)

# Bilateral Attenuation Function
def bila_atten(t, a, b):
    return  a*np.exp(-b*abs(t))

# Dot product
def dot_product(f, g, tmin, tmax):
    t = np.linspace(tmin, tmax, 1000)
    dt = t[1] - t[0]
    return np.dot(f(t), g(t))*dt

# Fourier image
def fourier_image(f, t0, t1):
    four_img = lambda w: 1/(np.sqrt(2*pi))*dot_product(f, lambda t: np.exp(-1j*w*t), t0, t1)
    return np.vectorize(four_img)

# Plot original function and complex function
def visualize_results(f, four_img, a, b, title):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    fig.suptitle(title + ' a = ' + str(float(a)) + ', b = ' + str(float(b)))
    
    plt.sca(axes[0])
    plt.plot(t, f(t).real, color='black', label='f(t)')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    
    plt.sca(axes[1])
    plt.plot(t, four_img(t).real, color='green', label='Real part')
    plt.plot(t, four_img(t).imag, color='red',  label='Imag part')
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()

# Parseval check
def parseval(f_original, four_img):
    four_image_abs = np.vectorize(lambda t: abs(four_img(t)))
    tmin = -50; tmax = 50
    parseval_1 = abs(dot_product(f_original, f_original, tmin, tmax))
    parseval_2 = dot_product(four_image_abs, four_image_abs, tmin, tmax)
    print(parseval_1, parseval_2, f'With (a, b) = {a, b}')


# Initial data
t0 = -15; t1 = 15             
Ts = 1000           
t = np.linspace(t0, t1, Ts)
a_s = np.array([2, 3])
b_s = np.array([1, 2])
c_s = np.array([-2, 2])

for i in range(len(a_s)):
    a = a_s[i]; b = b_s[i]
    
    # Rectangle function
    f_1 = np.vectorize(lambda t :rectangle(t, a, b), otypes=[np.complex_])
    four_img_1 = fourier_image(f_1, t0, t1)
    visualize_results(f_1, four_img_1, a, b, 'Rectangle function')
    parseval(f_1, four_img_1)
       
    # Triangle function
    # f_2 = np.vectorize(lambda t : triangle(t, a, b), otypes=([np.complex_]))
    # four_img_2 = fourier_image(f_2, t0, t1)
    # visualize_results(f_2, four_img_2, a, b, 'Triangle function')
    # print(parseval(f_2, four_img_2))
    
    # Sine function
    # f_3 = np.vectorize(lambda t: sinc(t, a, b), otypes=[np.complex_])
    # four_img_3 = fourier_image(f_3, t0, t1)
    # visualize_results(f_3, four_img_3, a, b, 'Cardinal sine')
    # print(parseval(f_3, four_img_3))

    # # Gauss function
    # f_4 = np.vectorize(lambda t: gauss(t, a, b), otypes=[np.complex_])
    # four_img_4 = fourier_image(f_4, t0, t1)
    # visualize_results(f_4, four_img_4, a, b, 'Gauss function')
    # print(parseval(f_4, four_img_4))

    # Bilateral attenuation function
    # f_5 = np.vectorize(lambda t: bila_atten(t, a, b), otypes=[np.complex_])
    # four_img_5 = fourier_image(f_5, t0, t1)
    # visualize_results(f_5, four_img_5, a, b, 'Bilateral Attenuation Function')
    # print(parseval(f_5, four_img_5))



