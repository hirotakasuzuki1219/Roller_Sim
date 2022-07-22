# ルンゲクッタの方がいいな

import numpy as np
from matplotlib import pyplot
import math
from scipy.integrate import odeint #odeint
# param
mass = 1
stiff = 10
damp = 1
## 解析条件
dt = 0.01
time_max = 10
## 初期値
time = np.linspace(0,time_max,math.ceil(time_max/dt))
x0 = [0.5, 0]

# 運動方程式の定義
def func(s,time,stiff,damp,mass):
    x,vel = s
    dxdt = [vel, (-stiff *x - damp * vel) / mass]
    return dxdt

# 運動方程式の解
sol = odeint(func,x0,time,args=(stiff,damp,mass))

# プロット
pyplot.plot(time,sol[:,0,],label='x')
pyplot.plot(time,sol[:,1],label='v')
pyplot.legend(loc='best')#レジェンドを付ける
pyplot.xlabel('t')
pyplot.show()

# 運動方程式

# スペクトラム解析

# 剛性計算