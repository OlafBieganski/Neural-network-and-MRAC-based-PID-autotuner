from pid import PIDController
from process_2tanks import TwoTankProcess
from autotuner_NN import NNPIDAutotuner
from reference_model import ReferenceModel
import matplotlib.pyplot as plt
import numpy as np
import copy

def ref_signal(t):
    # Generate the square wave signal with amplitude 0.6 and period 500 s
    usq = 0.6 * (1 + np.sign(np.sin(2 * np.pi * t / 500)))
    # Generate the sinusoidal signal with amplitude 0.6 and frequency corresponding to a period of 50 s
    sinusoidal = 0.6 * np.sin(0.02 * t)
    # Generate the reference signal uc(t) by adding the square wave and sinusoidal signals, and adding a constant offset of 5
    return usq + sinusoidal + 5

def squareWave(t):
    # Generate the square wave signal with amplitude 2 and period 2000 s
    return 2 * (1 + np.sign(np.sin(2 * np.pi * t / 2000))) + 5

'''
def predictNextModelErr(refModel: ReferenceModel, current_y, dy_du):
    model_cp = copy.deepcopy(refModel)
    y_next = model_cp.system_output()
'''

#simulation of a PID control for 2 tanks system

STEPS_AMT = 10000
TIME_STEP = 3
Y_0 = 0 #initial system state
H_D = 5 # desired height of liquid in 2nd tank (that is desired output of the system)
AUTOTUNING_ON = False

tanksSys = TwoTankProcess() # default data from the article
pid = PIDController(Kp=0.1,Ki=0.01,Kd=1) #random settings,  data from the article Kp=4.95,Ki=0.01,Kd=10.02
tunerNN = NNPIDAutotuner(2*10**(-7), 10**(-7), 0.0001) # data from paper
refModel = ReferenceModel()

time_n = np.arange(0, STEPS_AMT, TIME_STEP) # amount of a time steps for the simulation
system_y = [] # system output
refSystem_ym = [] # reference model output
controlVal_u = []
systemErr_e = []
referenceSignal_uc = []
systemJacobian_dYdu = []

#simulation
E = [0, 0, 0] # model errors at n, n-1, n-2
u_p = 0 # previous u
y_p = 0 # previous y
uc_p = 0 # previous u_c
# initial conditions
y = Y_0
u = 0
u_c = 0
for t in time_n:
    # predict K coeffcients updates
    if AUTOTUNING_ON: delta_Kp, delta_Ki, delta_Kd = tunerNN.predict([u, u_p, y, y_p, u_c, uc_p])
    else: delta_Kp, delta_Ki, delta_Kd = (0, 0, 0)
    pid.update_coefficients(pid.Kp + delta_Kp, pid.Ki + delta_Ki, pid.Kd + delta_Kd)
    u_c = squareWave(t) # reference signal, tank nr 2 liquid height
    u = pid.update_controller(u_c, y)
    if u > 100: u = 100 # control value is 0% - 100% and corresponds to opening of the valve
    if u < 0: u = 0 # same as ^^^
    y = tanksSys.process_run(u, TIME_STEP)
    dU = (0.003*TIME_STEP)/(u+0.00000001/100)
    dY_du = tanksSys.get_dY_dU(TIME_STEP, dU)
    y_m = refModel.system_output(u_c, TIME_STEP)
    e = y - y_m # model error
    # update errors arr
    E[2] = E[1]
    E[1] = E[0]
    E[0] = e
    if AUTOTUNING_ON: tunerNN.train(E, dY_du)
    # for plotting
    system_y.append(y)
    controlVal_u.append(u)
    refSystem_ym.append(y_m)
    systemErr_e.append(e)
    referenceSignal_uc.append(u_c)
    systemJacobian_dYdu.append(dY_du)
    # update previous data
    u_p = u
    y_p = y
    uc_p = u_c

# Plotting all collected signals on one graph
plt.figure(figsize=(10, 6))

plt.plot(time_n, system_y, label='liquid level - h_2', color='blue', linestyle='-')
plt.plot(time_n, controlVal_u, label='control value - v', color='red', linestyle='--')
plt.plot(time_n, refSystem_ym, label='reference model output - y_m', color='green', linestyle='-.')
plt.plot(time_n, systemErr_e, label='model error - e', color='orange', linestyle=':')
plt.plot(time_n, referenceSignal_uc, label='reference signal - u_c', color='purple', linestyle='-')
#plt.plot(time_n, systemJacobian_dYdu, label='system Jacobian - dY_du', color='brown', linestyle='--')

plt.xlabel('time')
plt.ylabel('values')
plt.legend()
plt.grid(True)
plt.title('Signals Over Time')

plt.tight_layout()
plt.show()

# Plotting all collected signals seperately
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(time_n, system_y, label='liquid level - h_2')
plt.xlabel('time')
plt.ylabel('liquid level')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(time_n, controlVal_u, label='control value - v')
plt.xlabel('time')
plt.ylabel('control value')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(time_n, refSystem_ym, label='reference model output - y_m')
plt.xlabel('time')
plt.ylabel('reference model output')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(time_n, systemErr_e, label='model error - e')
plt.xlabel('time')
plt.ylabel('model error')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(time_n, referenceSignal_uc, label='reference signal - u_c')
plt.xlabel('time')
plt.ylabel('reference signal')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(time_n, systemJacobian_dYdu, label='system Jacobian - dY_du')
plt.xlabel('time')
plt.ylabel('system Jacobian')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()