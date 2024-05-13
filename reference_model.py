import control as ct
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_AMT = 3000

class ReferenceModel:
    def __init__(self, natural_freq=0.02, damping=0.7) -> None:
        self.natural_freq = natural_freq
        self.damping = damping
        # Reference model transfer function numerator coefficients
        self.num = [natural_freq**2]
        # Reference model transfer function denominator coefficients
        self.den = [1, 2*damping*natural_freq, natural_freq**2]
        # Model transfer function
        self.G_m = ct.tf(self.num, self.den)

    def system_output(self, input):
        # Simulate one time step for the system and obtain the output
        t, y = ct.forced_response(self.G_m, T=np.array([0, 1]), U=np.array([0, input]))
        # Return the output at the end of the time step
        return y[-1]

        
model = ReferenceModel()
time = list(range(3000))
y = list()
u = 10
u_arr = [u] * 3000
plt.plot(time, u_arr)
for t in time:
    if t == 1500: u = 35
    y.append(model.system_output(u))

plt.plot(time, y)
plt.show()