import numpy as np
import matplotlib.pyplot as plt
import math

# パラメータ
dt = 0.01
tmax = 50

x = [[0],[0]]
v = [[0],[0]]

def ka_cal(t,x,v):
    mass_1 = 1
    mass_2 = 1
    stiff_1 = 1
    stiff_2 = 1
    damp_1= 1
    damp_2 = 1

    frec = 5
    g = 9.81
    F_amp = 0
    
    M = [[mass_1,0],[0,mass_2]]
    K =[[stiff_1, -stiff_1],[-stiff_1, stiff_1+stiff_2]]
    C =[[damp_1,-damp_1],[-damp_1,damp_1+damp_2]]
    F = [[mass_1 * g],[mass_2 * g + F_amp * math.sin(2* math.pi * frec * t)]]
    M_inv = np.linalg.inv(M)

    acc_right = - np.dot(K,x) - np.dot(C,v) + F
    acc = np.dot(M_inv, acc_right)
    return  acc

def kv_cal(v):
    return  v


t=0
k=0
while t<=tmax:
    k1_a = ka_cal(t,x,v)
    k1_v = kv_cal(v)
    k1_v  = np.array(k1_v)
    # print(type(k1_v))
    # np.matrix(k1_v)
    # print(t+dt/2)
    # print(k1_v)
    # print(k1_v.dtype)
    # print(float(k1_v))
    # print(float(k1_v)*dt)
    # print(x+k1_v*dt/2)
    # print(v+k1_a*dt/2)
    # k1_v*dt/2
    
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

    print(x)
    print(v)
    
    t += dt
    k = k + 1