import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift


def generator_sin(f, fs):
    t_total =  1  # Общее время для отображения 

    t = np.arange(0, t_total, 1/fs)
    y = np.sin(2 * np.pi * f * t)

    return y, t

def visual_graph(t, y, stem_flag):

    plt.figure(figsize=(10, 4))
    if stem_flag:    
        plt.stem(t, y)
    plt.plot(t, y, 'b-', alpha=0.5)
    plt.title('Дискретизация')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()
    plt.show()

def Hamming_window(k, N):
    return 0.54 + 0.46 * np.cos(np.pi * k / N)

def FIR_lowpass(k, L, N):
    return np.sinc(k / L) * Hamming_window(k, N)


def Upsampling(x, L):
    n = len(x)
    m = L*n
    y = np.zeros(m)
    x_i = 0
    for i in range(m):
        if i % L == 0:
            y[i] = x[x_i]
            x_i += 1

    return y


def filtering_signal(x, L):
    N = 2 * L
    m = len(x)
    # m = n * L
    y = np.zeros(m)

    for i in range(m):
        for k in range(-N, N + 1):
            if (i - k) >= 0 and (i-k) < m:
                h = FIR_lowpass(k, L, N)
                # h = FIR_lowpass(k, f_c, f_s)
                y[i] += h * x[i - k]

    return y


def interpolation(x, L):
    y = Upsampling(x, L)
    
    y = filtering_signal(y, L)
    
    return y
    
def DPF(x, N):
    X = fft(x, N) / N
    
    plt.figure
    k = np.arange(0, N)
    plt.stem(k, abs(X))
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")
    

def decimation(x, M):
    N = 2 * M
    m = len(x)
    n = m // M
    y = np.zeros(n)
    for i in range(n):
        for k in range(M):
            if (i - k) >= 0 and (i-k) < m:
                h = FIR_lowpass(k, M, N)
                # h = FIR_lowpass(k, f_c, f_s)
                y[i] += h * x[i*M - k]

    return y


def MSE(y_1, y_2):
    if len(y_1) != len(y_2):
        return -1
    
    N = len(y_1)
    summ = 0
    for n in range(N):
        summ += (y_1[n] - y_2[n]) ** 2
        
    return 1/N * summ
    

f = 10
fs = 100
L = 2
y, t = generator_sin(f, fs)

DPF(y, fs)

visual_graph(t, y, 0)
visual_graph(t, y, 1)


fs_2= fs * L
t2 = np.arange(0, 1, 1/fs_2)

y_2 = Upsampling(y, L)
DPF(y_2, fs_2)
visual_graph(t2, y_2, 1)



y_2 = filtering_signal(y_2, L)



visual_graph(t2, y_2, 0)
visual_graph(t2, y_2, 1)

DPF(y_2, fs_2)



M = 2

y_3 = decimation(y_2, M)


print(len(y_3))
visual_graph(t, y_3, 0)
visual_graph(t, y_3, 1)

DPF(y_3, fs)




mse = MSE(y, y_3)
print(mse)




mse_array = np.zeros(51)

for f in range(0, 50 + 1):
    y_1, t1 = generator_sin(f, fs)
    
    y_2 = interpolation(y_1, L)
    
    y_3 = decimation(y_2, M)
    mse_array[f] = MSE(y_1, y_3)
    
print(mse_array)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, 51), mse_array, marker='o', linestyle='-', color='b', label='MSE vs Frequency')
plt.xlabel('Частота (Гц)')
plt.ylabel('MSE')
plt.title('MSE в зависимости от частоты')
plt.grid(True)
plt.legend()
plt.show()





