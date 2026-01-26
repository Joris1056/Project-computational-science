from Main import ParkinsonSim
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


number_runs = 2

neurons_alive_total = []
years_total = []
for run in range(number_runs):
    print(run)
    sim = ParkinsonSim(visualize=False)
    sim.reset()

    done = False
    while not done:
        done = sim.step()
        if sim.t >= 2000:
            break
    neuron_alive = [100- p for p in sim.neuron_death]

    neurons_alive_total.append(neuron_alive)
    years_total.append(sim.time_years)

t_min = 0
t_max = min(max(y) for y in years_total)
n_points = 200

common_time = np.linspace(t_min, t_max, n_points)

neurons_interp = []
for years, neurons in zip(years_total,neurons_alive_total):
    f = interp1d(
        years,
        neurons,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    neurons_interp.append(f(common_time))

neurons_interp = np.array(neurons_interp)

mean_neurons_alive = np.nanmean(neurons_interp, axis=0)
std_neurons_alive = np.nanstd(neurons_interp, axis=0)
CI = [mean_neurons_alive - (1.96 * (std_neurons_alive/np.sqrt(number_runs))), mean_neurons_alive + (1.96 * (std_neurons_alive/np.sqrt(number_runs)))]

index_70 = np.where(mean_neurons_alive <= 70)[0][0]

plt.plot(common_time-common_time[index_70], mean_neurons_alive, label = 'mean')
plt.fill_between(np.array(common_time)-common_time[index_70], CI[0], CI[1], alpha = 0.3, label = '95% CI')
plt.axvline(x = common_time[index_70]-common_time[index_70], color='red', linestyle='--', label='70% neurons alive')
plt.xlabel('years')
plt.ylabel('% neurons alive')
plt.legend()
plt.show()
    


