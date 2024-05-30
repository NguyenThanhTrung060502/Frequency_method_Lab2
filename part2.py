import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

pi = np.pi

# Rectangle function
def rectangle(t, a, b):
    if abs(t) <= b:
        return a
    return 0

# Sinc function
def sinc(t, a, b):
    return a*np.sinc(b*t)

# Dot product
def dot_product(f, g, tmin, tmax):
    t = np.linspace(tmin, tmax, 1000)
    dt = t[1] - t[0]
    return np.dot(f(t), g(t))*dt

# Fourier image
def fourier_image(f, t0, t1):
    four_img = lambda w: 1/(np.sqrt(2*pi))*dot_product(f, lambda t: np.e ** (-1j*w*t), t0, t1)
    return np.vectorize(four_img)

# Plot original function and complex function
def visualize_results(g, four_img_g, abs_g, a, b):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 15))
    fig.suptitle('Case: c = ' + str(float(c)))
    
    plt.sca(axes[0])
    plt.plot(t, g(t).real, color='black', label='g(t)')
    plt.xlabel('t')
    plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    
    plt.sca(axes[1])
    plt.plot(t, abs_g(t), color='black', linewidth=1.2, label='|' + r'$\hat{g}(\omega)$|' )  
    plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    plt.sca(axes[2])
    plt.plot(t, four_img_g(t).real, color='green', linewidth=0.75, label='Real part')
    plt.plot(t, four_img_g(t).imag, color='red', linewidth=0.75, label='Imag part')
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()

# Parseval check
def parseval(g, four_img_g, c):
    four_image_abs = np.vectorize(lambda t: abs(four_img_g(t)))
    tmin = -50; tmax = 50
    parseval_1 = abs(dot_product(g, g, tmin, tmax))
    parseval_2 = dot_product(four_image_abs, four_image_abs, tmin, tmax)
    print(parseval_1, parseval_2, f'With (a, b, c) = {a, b, c}')


# Initial data
t0 = -15; t1 = 15             
Ts = 1000           
t = np.linspace(t0, t1, Ts)
a = 2
b = 1
c_s = np.array([-5, -1, 5, 10])


## Задание 2. Комплексное.
for i in range(len(c_s)):
    c = c_s[i]
    
    g = np.vectorize(lambda t: rectangle(t + c, a, b), otypes=[np.complex_])
    four_img_g = fourier_image(g, t0, t1)
    abs_g = lambda t: abs(four_img_g(t))
    
    visualize_results(g, four_img_g, abs_g, a, b)
    parseval(g, four_img_g, c)
    
    
