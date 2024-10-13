import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
N = 32   # Number of oscillators
alpha = 0.25  # Nonlinearity parameter
dt = 0.01
t_max = 10000

# Initial conditions
def initial_conditions(N):
    q0 = np.sin(np.pi * np.arange(1, N+1) / (N + 1))  # Mode-1 initial displacement
    p0 = np.zeros(N)  # Zero initial momentum
    return np.concatenate((q0, p0))

# Equations of motion
def fpu_system(y, t, alpha, N):
    q = y[:N]
    p = y[N:]
    
    dqdt = p
    dpdt = np.zeros(N)
    
    # Boundary conditions (q_0 = q_N+1 = 0)
    for i in range(N):
        if i == 0:
            dpdt[i] = q[i+1] - 2 * q[i] + alpha * ((q[i+1] - q[i]) ** 2)
        elif i == N-1:
            dpdt[i] = q[i-1] - 2 * q[i] + alpha * ((q[i] - q[i-1]) ** 2)
        else:
            dpdt[i] = (q[i+1] - 2 * q[i] + q[i-1] +
                       alpha * ((q[i+1] - q[i]) ** 2 - (q[i] - q[i-1]) ** 2))
    
    return np.concatenate((dqdt, dpdt))

# Time evolution
t = np.arange(0, t_max, dt)
y0 = initial_conditions(N)
sol = odeint(fpu_system, y0, t, args=(alpha, N))
q_t = sol[:, :N]
p_t = sol[:, N:]

# Compute the energies of each mode
def mode_energy(q_t, p_t, N):
    energies = []
    
    # Fourier transform to get modes
    for i in range(len(q_t)):
        q_k = np.fft.fft(q_t[i])[:N//2]  # Displacement in modes
        p_k = np.fft.fft(p_t[i])[:N//2]  # Momentum in modes
        
        # Mode energies
        omega_k = 2 * np.sin(np.pi * np.arange(1, N//2+1) / (2*(N+1)))  # Mode frequencies
        energy_k = 0.5 * (np.abs(p_k)**2 + (omega_k**2) * np.abs(q_k)**2)
        energies.append(energy_k)
    
    return np.array(energies)

energies = mode_energy(q_t, p_t, N)

plt.figure(figsize=(10, 6))
for mode in range(4):  # Plot first 5 modes
    plt.plot(t, energies[:, mode], label=f'Mode {mode+1}')

plt.title('Energy of Each Mode Over Time (FPU Paradox)')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.grid()

# Save the plot to a file
plt.savefig('fpu_energy_modes.png')

# To make sure no plot is left open
plt.close()

