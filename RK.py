import numpy as np
import matplotlib.pyplot as plt
import math

# パラメータ
dt = 0.0001
tmax = 1
N = math.ceil(tmax/dt)+1
freq = np.linspace(0, 1.0/dt, N) # 周波数軸

x = [[0],[0]]
v = [[0],[0]]

def ka_cal(t,x,v):

    freq_F = 20
    g = 9.81
    F_amp = 29400

    mass_1 = 500
    mass_2 = 1000
    stiff_1 = 10**6
    stiff_2 = 10 * 10**7
    damp_1= 2 * 0.1 * math.sqrt(mass_1*stiff_1)
    damp_2= 2 * 0.4 * math.sqrt(mass_2*stiff_2)


    
    M = [[mass_1,0],[0,mass_2]]
    K =[[stiff_1, -stiff_1],[-stiff_1, stiff_1+stiff_2]]
    C =[[damp_1,-damp_1],[-damp_1,damp_1+damp_2]]
    F = [[mass_1 * g],[mass_2 * g + F_amp * math.sin(2* math.pi * freq_F * t)]]
    M_inv = np.linalg.inv(M)

    acc_right = - np.dot(K,x) - np.dot(C,v) + F
    acc = np.dot(M_inv, acc_right)
    return  acc

def kv_cal(v):
    return  v

t_mat = []
acc_mat_1 = []
acc_mat_2 = []
x_mat_1 = []
x_mat_2 = []
t=0
k=0
while t<=tmax:
    k1_a = ka_cal(t,x,v)
    k1_v = kv_cal(v)
    k1_v  = np.array(k1_v)
    k2_a = ka_cal(t+dt/2, x+k1_v*dt/2, v+k1_a*dt/2)
    k2_v = kv_cal(v)
    k2_v  = np.array(k2_v)
    k3_a = ka_cal(t+dt/2, x+k2_v*dt/2, v+k2_a*dt/2)
    k3_v = kv_cal(v)   
    k3_v  = np.array(k3_v)
    k4_a = ka_cal(t+dt, x+k3_v*dt, v+k3_a*dt)
    k4_v = kv_cal(v)
    k4_v  = np.array(k4_v)

    x = x + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    v = v + dt/6 * (k1_a + 2*k2_a + 2*k3_a + k4_a)

    t_mat.append(t)
    acc_mat_1.append(k1_a[0])
    acc_mat_2.append(k1_a[1])
    x_mat_1.append(x[0])
    x_mat_2.append(x[1])
    
    t += dt
    k = k + 1

F = np.fft.fft(acc_mat_1)
# 正規化 + 交流成分2倍
F = F/(N/2)
F[0] = F[0]/2

plt.figure()
plt.plot(t_mat, x_mat_1,label='x1')
plt.plot(t_mat, x_mat_2,label='x2')
plt.xlabel("Time")
plt.ylabel("X")
plt.legend()

plt.figure()
plt.plot(t_mat, acc_mat_1,label='acc1')
plt.plot(t_mat, acc_mat_2,label='acc2')
plt.xlabel("Time")
plt.ylabel("acc")
plt.legend()

plt.figure()
# plt.plot(freq, F,label='F')
plt.plot(freq[0:N//2], np.abs(F[0:N//2])**2)
plt.xlabel("Freq.")
plt.ylabel("Amp.")
plt.xlim(0,100)

plt.show()