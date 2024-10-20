

#执行NARMA2任务

import numpy as np
from scipy.linalg import pinv  #伪逆矩函数
from scipy.linalg import inv  #逆矩函数
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%% NARMA2时间序列模型

def NARMA2(n):
    N = n*2
    np.random.seed(1)
    w = np.random.uniform(0,0.5,N)
    y = np.zeros(N)

    for i in range(2, N):
        y[i] = 0.4*y[i-1] + 0.4*y[i-1]*y[i-2] + 0.6*np.power(w[i-1], 3) + 0.1

    return y[-n:],w[-n-1:-1]

#%% RC模型

def RC_circuit(
               input_voltage,
               pulse_width,
               output_voltage_initial,
               resistance,
               capacitance, 
               pulse_period,
               ):
    
    tau = resistance * capacitance
    V1_state = np.array([])

    V = input_voltage * (1 - np.exp(-pulse_width / tau)) * np.exp(-(pulse_period - pulse_width) / tau) + output_voltage_initial * np.exp(-pulse_period / tau)
    output_voltage_initial = V
    V1_state = np.append(V1_state, V)

    #绘图
    #plt.figure(1)
    #plt.plot(V1_all, 'r-', label='V1(t)')
    #plt.xlabel('Time Step')
    #plt.ylabel('Vout')
    #plt.legend()
    #plt.title('Time Evolution')
    #plt.show()

    return V1_state

#%% 确定虚拟节点的个数

Vnode_num = 5

#%% 确定RC电路个数

device_num = 8

#%% 确定脉冲周期,pulse width + pulse interval

pulse_period = 4E-3

#%% 确定脉宽与输入电压幅值

pulse_width_max = pulse_period     #确定最大脉宽
pulse_width_min = 0          #确定最小脉宽
input_voltage = 3.3            #确定输入电压幅值

#%% 确定电阻

resistance = 40000

#%% 确定电容,是否需要D2D

if_D2D = 2

if if_D2D == 0:
    tau = np.ones(device_num)*pulse_period*1
    capacitance = tau/resistance

if if_D2D == 1:
    tau_ratio_start = 0.5     #tau是pulse_period的多少倍
    tau_ratio_step = 0.3      #tau是pulse_period的多少倍        
    tau = np.arange(tau_ratio_start, tau_ratio_start+device_num*tau_ratio_step, tau_ratio_step)*pulse_period          
    capacitance = tau/resistance

if if_D2D == 2:
    capacitance = np.array([10E-9,25E-9,40E-9,55E-9,70E-9,85E-9,100E-9,115E-9])

#%% 展示tau

tau_pulse_period_show = capacitance*resistance/pulse_period

#%% 岭回归(1)还是线性回归(0)

fit_type = 0

alpha = 0        #岭回归正则化参数

#%% washout_num

washout_num = 50

#%% NARMA2信号生成

#生成
n = 500       #时间序列的长度
data_1,data_2 = NARMA2(n)

#划分
data_input = data_2
data_target = data_1

train_step = 250
initialization_step = 250

data_train = data_input[0:train_step:1]
data_test = data_input[train_step::1]

data_test_init = data_test[0:initialization_step:1]
data_test_pred = data_test[initialization_step::1]

target_train = data_target[Vnode_num-1:train_step:1]
target_test = data_target[train_step::1]

#%% train

#train运行
reservoir_state = np.zeros((device_num,len(data_train)))
output_voltage_initial = np.zeros(device_num)

for k in range(0,len(data_train),1):
    UL = np.max(data_input)
    DL = np.min(data_input)
    pulse_width = (data_train[k]-DL)/(UL-DL)*(pulse_width_max-pulse_width_min)+pulse_width_min
    
    for l in range (0,device_num,1):
        output_voltage = RC_circuit(
                                    input_voltage,
                                    pulse_width,
                                    output_voltage_initial[l],
                                    resistance,
                                    capacitance[l],                                     
                                    pulse_period,        
                                    )
    
        reservoir_state[l,k] =  output_voltage[0]        #筛选输出电压信号
        #reservoir_state[l,k] =  pulse_width        #筛选输出电压信号
        
        output_voltage_initial[l] = output_voltage[-1]

states = np.zeros((device_num*Vnode_num,len(data_train)-Vnode_num+1))
for m in range (0,len(data_train)-Vnode_num+1,1):
    a = reservoir_state[:,m:m+Vnode_num]
    states[:,m] = a.flatten()
    
state_train = states
b = np.ones((1,len(data_train)-Vnode_num+1))
state_train = np.concatenate((b,state_train), axis=0)

if fit_type == 0:
#线性回归
    #Wout = np.dot(target_train,pinv(state_train))
    Wout = np.dot(np.dot(target_train,state_train.T),pinv(np.dot(state_train,state_train.T)))

if fit_type == 1:
#岭回归
    Wout = np.dot(np.dot(target_train,state_train.T),inv(np.dot(state_train,state_train.T)+alpha*np.eye(len(np.dot(state_train,state_train.T)))))

#train-test
train_test = np.dot(Wout, state_train) 

#train_test展示

#NRMSE: Dynamic memristor-based reservoir computing for high-efficiency temporal signal processing
train_NRMSE = np.sqrt(np.mean(np.square(train_test[washout_num:]-target_train[washout_num:]))/np.var(target_train[washout_num:]))
print('train_NRMSE: '+ str(train_NRMSE))

#Prediction error: Edge-of-chaos learning achieved by ion-electron–coupled dynamics in an ion-gating reservoir
train_prediction_error = np.sum(np.square(train_test[washout_num:]-target_train[washout_num:]))/np.sum(np.square(target_train[washout_num:]))
print('train_prediction_error: '+ str(train_prediction_error))

#NMSE: Edge-of-chaos learning achieved by ion-electron–coupled dynamics in an ion-gating reservoir
train_NMSE = np.sum(np.square(train_test[washout_num:]-target_train[washout_num:]))/(np.var(target_train[washout_num:])*len(data_train))
print('train_NMSE: '+ str(train_NMSE))

fig_1, ax_1 = plt.subplots(figsize=(50, 20))
ax_1.plot(target_train, 'y', lw=15,label="target")
ax_1.plot(train_test, 'b', lw=10,label="prediction")
ax_1.set_xlabel('Time Step',fontsize=120)
ax_1.set_ylabel('Prediction',fontsize=120)
ax_1.tick_params(axis='x', labelsize= 80)
ax_1.tick_params(axis='y', labelsize= 80)
ax_1.legend()
ax_1.grid(linewidth = 5)
plt.tight_layout()

#%% test

#initialization运行
test_test = np.array([])
#output_voltage_initial = np.zeros(device_num)

for k in range(0,len(data_test_init),1):
    UL = np.max(data_input)
    DL = np.min(data_input)
    pulse_width = (data_test_init[k]-DL)/(UL-DL)*(pulse_width_max-pulse_width_min)+pulse_width_min
    
    reservoir_state_init = np.zeros(device_num)
    
    for l in range (0,device_num,1):
        output_voltage = RC_circuit(
                                    input_voltage,
                                    pulse_width,
                                    output_voltage_initial[l],
                                    resistance,
                                    capacitance[l],                                     
                                    pulse_period,        
                                    )
    
        reservoir_state_init[l] =  output_voltage[0]        #筛选输出电压信号
        #reservoir_state_init[l] =  pulse_width        #筛选输出电压信号
        
        output_voltage_initial[l] = output_voltage[-1]
        
    #initialization-test
    reservoir_state = np.concatenate((reservoir_state,reservoir_state_init.reshape(-1,1)), axis=1)
    states = reservoir_state[:,-Vnode_num:].flatten().reshape(-1,1)
    
    state_test = states
    b = np.ones((1,1))
    state_test = np.concatenate((b,state_test), axis=0)
    initialization_test = np.dot(Wout, state_test)  
    test_test = np.append(test_test,initialization_test)  
    
#autonomous prediction运行
for k in range(0,len(data_test_pred),1):
    UL = np.max(data_input)
    DL = np.min(data_input)
    pulse_width = (test_test[-1]-DL)/(UL-DL)*(pulse_width_max-pulse_width_min)+pulse_width_min
    
    reservoir_state_pred = np.zeros(device_num)
    
    for l in range (0,device_num,1):
        output_voltage = RC_circuit(
                                    input_voltage,
                                    pulse_width,
                                    output_voltage_initial[l],
                                    resistance,
                                    capacitance[l],                                     
                                    pulse_period,        
                                    )
    
        reservoir_state_pred[l] =  output_voltage[0]        #筛选输出电压信号
        output_voltage_initial[l] = output_voltage[-1]
        
    #autonomous prediction-test
    reservoir_state = np.concatenate((reservoir_state,reservoir_state_pred.reshape(-1,1)), axis=1)
    states = reservoir_state[:,-Vnode_num:].flatten().reshape(-1,1)
    
    state_test = states
    b = np.ones((1,1))
    state_test = np.concatenate((b,state_test), axis=0)
    autonomous_prediction_test = np.dot(Wout, state_test)  
    test_test = np.append(test_test,autonomous_prediction_test)
    
#test_test展示

#NRMSE: Dynamic memristor-based reservoir computing for high-efficiency temporal signal processing
test_NRMSE = np.sqrt(np.mean(np.square(test_test[washout_num:]-target_test[washout_num:]))/np.var(target_test[washout_num:]))
print('test_NRMSE: '+ str(test_NRMSE))

#Prediction error: Edge-of-chaos learning achieved by ion-electron–coupled dynamics in an ion-gating reservoir
test_prediction_error = np.sum(np.square(test_test[washout_num:]-target_test[washout_num:]))/np.sum(np.square(target_test[washout_num:]))
print('test_prediction_error: '+ str(test_prediction_error))

#NMSE: Edge-of-chaos learning achieved by ion-electron–coupled dynamics in an ion-gating reservoir
test_NMSE = np.sum(np.square(test_test[washout_num:]-target_test[washout_num:]))/(np.var(target_test[washout_num:])*len(data_test))
print('test_NMSE: '+ str(test_NMSE))

fig_2, ax_2 = plt.subplots(figsize=(50, 20))
ax_2.plot(target_test, 'y', lw=15,label="target")
ax_2.plot(test_test, 'r', lw=10,label="prediction")
ax_2.plot(test_test[0:initialization_step], 'b', lw=10,label="initialization")
ax_2.set_xlabel('Time Step',fontsize=120)
ax_2.set_ylabel('Prediction',fontsize=120)
ax_2.tick_params(axis='x', labelsize= 80)
ax_2.tick_params(axis='y', labelsize= 80)
plt.rcParams.update({'font.size':80})
ax_2.legend()
ax_2.grid(linewidth = 5)
#plt.ylim(0.18,0.32)
#plt.xlim(20,)
plt.tight_layout()
