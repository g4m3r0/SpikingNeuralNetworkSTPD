from ANNarchy import *

dt = 0.1
setup(dt=dt)

LIF = Neuron(
    parameters = """
    tau_m = 20.0 # Membrane time constant (ms)
    R = 1.0 # Membrane resistance (MOhm)
    V_rest = -70.0 # Resting potential (mV)
    V_reset = -80.0 # Reset potential (mV)
    V_th = -50.0 # Threshold potential (mV)
    I = 0.0 # External current (nA)
    """,

    equations = """
    # Membrane equation
    tau_m * dV/dt = -(V - V_rest) + R * I : init = -70.0, midpoint
    """,

    spike = """
    V >= V_th
    """,

    reset = """
    V = V_reset
    """
)

pop = Population(neuron=LIF, geometry=1)
pop.V = -70.0

compile()

m = Monitor(pop, ['spike', 'V'])

# Simulation
for i in range(2000):
    if i >= 200 and i <= 800:
        pop.I = 5.0
    else:
        pop.I = 0.0
    simulate(dt)

data = m.get()

import matplotlib.pyplot as plt
plt.plot(dt*np.arange(len(data['V'])), data['V'])
plt.title('V')
plt.show()
