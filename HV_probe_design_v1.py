import numpy as np
import math as math
import matplotlib.pyplot as plt
#2024/11/16 Ryouta Nagata.
#High Voltage Probe の多段RCリードフィルタの設計プログラム.
#コンソール出力はLTspice のためと、step応答のシミュレーションプログラム（HV_probe_design_v1_step_responce.py）のためのもの.
#warburg impedanceの計算は以下の論文をもとに実装.
#Juraj VALSA et.al., "Network Model of the CPE," RADIOENGINEERING, VOL. 20, NO. 3, 2011.
#lossy線路のカットオフ周波数をfreq_minにすると少し調整が必要だったため、それはgssで解かせている.

#黄金分割法で誤差を最小化.
def gss_len_sec(f, a, b, tolerance=1e-4):
    """
    https://en.wikipedia.org/wiki/Golden-section_search
    Golden-section search
    to find the minimum of f on [a,b]

    * f: a strictly unimodal function on [a,b]

    Example:
    >>> def f(x): return (x - 2) ** 2
    >>> x = gss(f, 1, 5)
    >>> print(f"{x:.5f}")
    2.00000
    """
    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        if f(c) < f(d):
            b = d
        else:  # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2

def f(freq_shift):
    global res, cap, Rp, Cp, freq, Gain, Z_RCladder, Z_Rhv, freq_max

    freq_max=freq_max_def*freq_shift

    Rp_design=R_hv/(divide_ratio-1)

    R0=R_hv/len_R
    C0=2*math.pi*eps_r*8.854e-12/np.log(dia_shield/dia_R)
    Omega_min=1/(len_R**2*R0*C0)
    Omega_min=Omega_min*freq_shift
    T=np.sqrt(C0/R0)*divide_ratio
    C1=T

    Omega_max=2*math.pi*freq_max
    freq_min=Omega_min/(2*math.pi)
    P=0.5#warburg impedance
    ab=(Omega_min/Omega_max)**(1/n)
    tau1=1/Omega_min
    R1=tau1/C1
    phi = P*90
    alpha=phi/90
    a=ab**alpha
    b=ab/a

    Rp=R1*(1-a)/a
    Cp=C1*b**n/(1-b)
    exp=np.array(np.linspace(0,n-1,n))
    res=R1*a**exp
    cap=C1*b**exp

    k=int(n/2)
    Omega=np.sqrt(Omega_min*Omega_max)
    inv_Z_RCladder = []
    inv_Z_RCladder.append(1/Rp)
    for i in range(n):
        Rs = res[i]
        Cs = cap[i]
        y=1/(Rs+1/(1j*Omega*Cs))
        inv_Z_RCladder.append(y)
    inv_Z_RCladder.append(1j*Omega*Cp)
    Z_RCladder=np.abs(1/np.sum(inv_Z_RCladder))

    Zcpe = 1/(T*(1j*Omega)**P)
    abs_Zcpe=np.abs(Zcpe)
    D=abs_Zcpe/Z_RCladder#インピーダンスの絶対値を指定のそれと合わせる.

    #D=Rp_design/Rp
    Rp=Rp_design
    res=res*D
    Cp=Cp/D
    cap=cap/D

    if key==0:
        freq = np.logspace(np.log10(freq_min)-2, np.log10(freq_max)-np.log10(freq_max/freq_min)/5, freq_num)
    else:
        freq = np.logspace(np.log10(freq_min)-2, np.log10(freq_max)+2, freq_num)
    Omega = 2*math.pi*freq
    Z_RCladder = []
    for k in range(freq_num):
        inv_Z_RCladder = []
        inv_Z_RCladder.append(1/Rp)
        for i in range(n):
            Rs = res[i]
            Cs = cap[i]
            y=1/(Rs+1/(1j*Omega[k]*Cs))
            inv_Z_RCladder.append(y)
        inv_Z_RCladder.append(1j*Omega[k]*Cp)
        Z_RCladder.append(1/np.sum(inv_Z_RCladder))

    Z_RCladder = np.array(Z_RCladder)

    theta=(1+1j)/np.sqrt(2)*np.sqrt(Omega*C0*R0)*len_R
    y=[]
    for x in theta:
        if np.abs(x)<10:
            y.append(np.tanh(x))
        else:
            y.append(1)
    y=np.array(y)
    Z_Rhv=(R0/(1j*Omega*C0))**0.5*y#RC線路の真のインピーダンス.
    Gain=Z_RCladder/(Z_RCladder+Z_Rhv)
    abs_Gain = np.abs(Gain)

    #error=np.abs(np.sum(1-abs_Gain*divide_ratio))/freq_num
    error=np.max(np.abs(1-abs_Gain*divide_ratio))#真のインピーダンスとラダー回路との誤差.
    return error

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi

freq_max_def=100e6#最高周波数[Hz].
n=6 #RCの数.
divide_ratio=1000#1/分圧比.
R_hv=100e6#高電圧抵抗の抵抗値.
len_R=123.7e-3#高電圧抵抗の長さ.
dia_R=8.2e-3#高電圧抵抗の直径.
dia_shield=50e-3#高電圧抵抗の周りのシールドの直径.
eps_r=3#高電圧抵抗のと周りのシールドの間を満たす絶縁物の誘電率.

freq_num=100#plot用.


key=0
freq_shift=gss_len_sec(f, 0.1, 10, tolerance=1e-4)
key=1#plt用に周波数範囲を広げる.
f(freq_shift)

Gain=Z_RCladder/(Z_RCladder+Z_Rhv)
abs_Gain = np.abs(Gain)
phase_Gain = np.arctan2(np.imag(Gain),np.real(Gain))

print(R_hv,len_R,dia_R,dia_shield,eps_r,freq_max_def,n,divide_ratio,Rp,Cp,freq_shift,sep=", ")
for i in range(n):
    print(f'{res[i]:e}',end=", ")
for i in range(n-1):
    print(f'{cap[i]:e}',end=", ")
print(f'{cap[i+1]:e}')

print(".param ",end="")
print("Rhv=",end="")
print(f'{R_hv:e}',end=", ")
print("len=",end="")
print(f'{len_R:e}',end=", ")
print("dia_R=",end="")
print(f'{dia_R:e}',end=", ")
print("dia_shield=",end="")
print(f'{dia_shield:e}',end=", ")
print("eps_r=",end="")
print(f'{eps_r:e}',end=", ")
for i in range(n):
    print("R"+str(i)+"=",end="")
    print(f'{res[i]:e}',end=", ")
for i in range(n):
    print("C"+str(i)+"=",end="")
    print(f'{cap[i]:e}',end=", ")

print("Rp=",end="")
print(f'{Rp:e}',end=", ")

print("Cp=",end="")
print(f'{Cp:e}',end="")

fig, ax = plt.subplots(nrows=2, ncols=1, squeeze=False, tight_layout=True, figsize=[8,6], sharex = "col")
ax[0,0].plot(freq,abs_Gain,"k-")
ax[0,0].plot([freq[0],freq[-1]],[1/divide_ratio,1/divide_ratio],"r--")
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')

ax[1,0].plot(freq,phase_Gain*180/math.pi,"k-")
ax[1,0].plot([freq[0],freq[-1]],[0,0],"r--")
ax[1,0].set_xscale('log')

plt.show()
