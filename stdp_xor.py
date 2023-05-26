import os
os.environ['ANNARCHY_CONFIG_FILE'] = 'annarchy.json'

from ANNarchy import *
import numpy as np

# Parameters
F = 15.0 # Poisson distribution at 15 Hz
N = 2 # 2 Poisson inputs for XOR
gmax = 0.01 # Maximum weight
duration = 100000.0 # Simulation for 100 seconds

# XOR input patterns
patterns = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# Definition of the neuron
IF = Neuron(
    parameters = """
        tau_m = 10.0
        tau_e = 5.0
        vt = -54.0
        vr = -60.0
        El = -74.0
        Ee = 0.0
    """,
    equations = """
        tau_m * dv/dt = El - v + g_exc * (Ee - vr) : init = -60.0
        tau_e * dg_exc/dt = - g_exc
    """,
    spike = """
        v > vt
    """,
    reset = """
        v = vr
    """
)

# Input population
Input = PoissonPopulation(name='Input', geometry=N, rates=F)

# Output neuron
Output = Population(name='Output', geometry=1, neuron=IF)

# Projection learned using STDP
proj = Projection(
    pre=Input,
    post=Output,
    target='exc',
    synapse=STDP(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.0105, w_max=0.01)
)
proj.connect_all_to_all(weights=Uniform(0.0, gmax))

# Compile the network
compile()

# Start recording
Mi = Monitor(Input, 'spike')
Mo = Monitor(Output, 'spike')

# Simulation loop
for _ in range(int(duration / 1000)):
    for pattern in patterns:
        # Set input rates based on pattern
        Input.rates = np.array(pattern) * F

        # Simulate for 1 second
        simulate(1000.0)

# Retrieve the recordings
input_spikes = Mi.get('spike')
output_spikes = Mo.get('spike')

# Compute the mean firing rates during the simulation
print('Mean firing rate in the input population: ' + str(Mi.mean_fr(input_spikes)))
print('Mean firing rate of the output neuron: ' + str(Mo.mean_fr(output_spikes)))

# Compute the instantaneous firing rate of the output neuron
output_rate = Mo.smoothed_rate(output_spikes, 100.0)

# Receptive field after simulation
weights = proj.w[0]

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
plt.subplot(3, 1, 1)
plt.plot(output_rate[0, :])
plt.subplot(3, 1, 2)
plt.plot(weights, '.')
plt.subplot(3, 1, 3)
plt.hist(weights, bins=20)
plt.show()
