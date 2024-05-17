import control as ct
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_AMT = 20000

class ReferenceModel:
    def __init__(self, natural_freq=0.03, damping=0.7) -> None:
        self.omega = natural_freq
        self.eta = damping
        self.prev_y = 0
        self.prev_prev_y = 0

    def system_output(self, input, time_step):
        y = 1/(1 + 1/(self.omega**2 * time_step**2) + (2*self.eta)/(self.omega*time_step)) * (input + (2/(self.omega**2 * time_step**2) + (2*self.eta)/(self.omega*time_step))*self.prev_y - 1/(self.omega**2 * time_step**2)*self.prev_prev_y)
        self.prev_y = y
        self.prev_prev_y = self.prev_y
        return y
        

        
model = ReferenceModel()
time = list(range(SAMPLE_AMT))
y = list()
u = 10
u_arr = [u] * SAMPLE_AMT
plt.plot(time, u_arr)
for t in time:
    if t == 6000: u = 35
    y.append(model.system_output(u, 1))

plt.plot(time, y)
plt.show()