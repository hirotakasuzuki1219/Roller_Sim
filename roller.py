# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math


# データのパラメータ
N = 256            # サンプル数
dt = 0.01          # サンプリング間隔
f1, f2 = 10, 50    # 周波数
t = np.arange(0, N*dt, dt) # 時間軸
freq = np.linspace(0, 1.0/dt, N) # 周波数軸

# 信号を生成（周波数10の正弦波+周波数20の正弦波+ランダムノイズ）
f = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 0.3 * np.random.randn(N)

# 高速フーリエ変換
F = np.fft.fft(f)

# 振幅スペクトルを計算
Amp = np.abs(F)

# グラフ表示
plt.figure()
plt.subplot(121)
plt.plot(t, f, label='f(n)')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.grid()
leg = plt.legend(loc=1)
leg.get_frame().set_alpha(1)
plt.subplot(122)
plt.plot(freq, Amp, label='|F(k)|')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid()
leg = plt.legend(loc=1)
leg.get_frame().set_alpha(1)
plt.show()

file_name = 'wave.jpg'
plt.savefig(file_name)

w = np.stack([t, f])
np.savetxt('python-csv.csv', w.T, delimiter=',')



# import numpy as np
# import matplotlib.pyplot as plt
# import math
 
# # パラメータ
# wave_number = 2     # 合成する波形の数
# dt = 0.01           # サンプリング周期 [sec]
# # time_max = 50
# data_number = 256 # データ数
   
# ampl        = np.random.rand(wave_number) # 0.0 以上 1.0 未満の乱数
# # freq        = np.random.rand(wave_number) # 0.0 以上 1.0 未満の乱数
# freq        = np.linspace(0, 1.0/dt, data_number) # 周波数軸
# t           = np.linspace(0, data_number*dt, data_number)
# # t           = np.linspace(0, data_number*dt, dt)
# # Freq = np.linspace(0, 1.0/dt, data_number) # 周波数軸
# print(t)

# for i in range(0, wave_number):
#     if i==0:
#         f  = ampl[i] * np.sin(2 * np.pi * freq[i] * t)
#     else:
#         f += ampl[i] * np.sin(2 * np.pi * freq[i] * t)

# F = np.fft.fft(f)
# print(F)
# Amp = np.abs(F)
# print(Amp)
# print(len(Amp))

# # plt.rcParams['font.size'] = 17

# # グラフ表示
# plt.plot(t, f)
# plt.grid(True)
# plt.title('Wave')
# plt.xlabel('time[sec]')
# plt.ylabel('amplitude')

# # グラフ出力
# file_name = 'wave.jpg'
# plt.savefig(file_name)


# plt.figure()
# plt.subplot(121)
# plt.plot(t, f, label='f(n)')
# plt.xlabel("Time")
# plt.ylabel("Signal")
# plt.grid()
# leg = plt.legend(loc=1)
# leg.get_frame().set_alpha(1)
# plt.subplot(122)
# # plt.plot(t, Amp, label='|F(k)|')
# plt.plot(t, Amp, label='|F(k)|')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.grid()
# leg = plt.legend(loc=1)
# leg.get_frame().set_alpha(1)

# plt.show()

# w = np.stack([t, f])
# np.savetxt('python-csv.csv', w.T, delimiter=',')