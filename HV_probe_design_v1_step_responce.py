import numpy as np
import math as math
import matplotlib.pyplot as plt
#2024/11/16 Ryouta Nagata.
#High Voltage Probe の多段RCリードフィルタのstep応答のシミュレーション.

def f(freq_shift):
    global res, cap, Rp, Cp, freq, Gain, Z_RCladder, Z_Rhv, freq_max

    freq_max=freq_max_def*freq_shift

    R0=R_hv/len_R
    C0=2*math.pi*eps_r*8.854e-12/np.log(dia_shield/dia_R)
    Omega_min=1/(len_R**2*R0*C0)
    Omega_min=Omega_min*freq_shift
    T=np.sqrt(C0/R0)*divide_ratio

    freq_min=Omega_min/(2*math.pi)
    P=0.5

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
    Z_Rhv=(R0/(1j*Omega*C0))**0.5*y
    Gain=Z_RCladder/(Z_RCladder+Z_Rhv)
    abs_Gain = np.abs(Gain)

    #error=np.abs(np.sum(1-abs_Gain*divide_ratio))/freq_num
    error=np.max(np.abs(1-abs_Gain*divide_ratio))
    return error

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi

R_hv,len_R,dia_R,dia_shield,eps_r,freq_max_def,n,divide_ratio,Rp,Cp,freq_shift=[100000000.0, 0.1237, 0.0082, 0.05, 3, 100000000.0, 6, 1000, 100100.1001001001, 4.885236106639213e-12, 8.141852268682936]
rc=[5.022041e+04, 1.632659e+04, 5.307751e+03, 1.725542e+03, 5.609715e+02, 1.823711e+02, 2.792770e-09, 9.079256e-10, 2.951653e-10, 9.595785e-11, 3.119576e-11, 1.014170e-11]

res=rc[:n]
cap=rc[n:]
freq_num=1000

#step 応答のシミュレーションの設定.
t_end=10e-3
t_delta=100e-9

key=1
f(freq_shift)

Gain=Z_RCladder/(Z_RCladder+Z_Rhv)
abs_Gain = np.abs(Gain)
phase_Gain = np.arctan2(np.imag(Gain),np.real(Gain))

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


segment_size=int(t_end/t_delta)
time=np.linspace(0,t_end,segment_size)
vin=np.concatenate([np.zeros(int(segment_size/2)),np.ones(segment_size-int(segment_size/2))])
fft_vin=np.fft.fft(vin)

freq_min=0
freq_max=1/t_delta/2
freq_num=int(segment_size/2)

freq = np.linspace(freq_min, freq_max, freq_num)
Omega = 2*math.pi*freq
Z_RCladder = []
Z_RCladder.append(Rp)
for k in range(freq_num-1):
    inv_Z_RCladder = []
    inv_Z_RCladder.append(1/Rp)
    for i in range(n):
        Rs = res[i]
        Cs = cap[i]
        y=1/(Rs+1/(1j*Omega[k+1]*Cs))
        inv_Z_RCladder.append(y)
    inv_Z_RCladder.append(1j*Omega[k+1]*Cp)
    Z_RCladder.append(1/np.sum(inv_Z_RCladder))
Z_RCladder = np.array(Z_RCladder)

R0=R_hv/len_R
C0=2*math.pi*eps_r*8.854e-12/np.log(dia_shield/dia_R)

theta=(1+1j)/np.sqrt(2)*np.sqrt(Omega*C0*R0)*len_R
y=[]
for x in theta:
    if np.abs(x)<10:
        y.append(np.tanh(x))
    else:
        y.append(1)
y=np.array(y)
Z_Rhv=[]
Z_Rhv.append(R0*len_R)
for i in range(freq_num-1):
    Z_Rhv.append((R0/(1j*Omega[i+1]*C0))**0.5*y[i+1])
Z_Rhv=np.array(Z_Rhv)
Gain=Z_RCladder/(Z_RCladder+Z_Rhv)

fft_Gain=np.concatenate([Gain,np.flip(Gain)])

fft_vout=fft_Gain*fft_vin
vout=np.fft.ifft(fft_vout)
fig2, ax2 = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True, figsize=[8,4], sharex = "col")
ax2[0,0].plot(time-time[-1]/2,np.abs(vin),"r--")
ax2[0,0].plot(time-time[-1]/2,np.real(vout)*divide_ratio,"k-")

plt.show()