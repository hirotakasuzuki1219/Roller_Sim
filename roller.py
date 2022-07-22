import numpy as np
import matplotlib.pyplot as plt
import math
 
# パラメータ
wave_number = 1     # 合成する波形の数
dt = 0.005           # サンプリング周期 [sec]
time_max = 50
data_number = math.floor(time_max/dt) # データ数
   
ampl        = np.random.rand(wave_number) # 0.0 以上 1.0 未満の乱数
freq        = np.random.rand(wave_number) # 0.0 以上 1.0 未満の乱数
t           = np.arange(0, data_number*dt, dt)

for i in range(0, wave_number):
    if i==0:
        f  = ampl[i] * np.sin(2 * np.pi * freq[i] * t)
    else:
        f += ampl[i] * np.sin(2 * np.pi * freq[i] * t)

# グラフ表示
plt.plot(t, f)
plt.grid(True)
plt.title('Wave')
plt.xlabel('time[sec]')
plt.ylabel('amplitude')

# グラフ出力
file_name = 'wave.jpg'
plt.savefig(file_name)
plt.show()

w = np.stack([t, f])
np.savetxt('python-csv.csv', w.T, delimiter=',')
 
