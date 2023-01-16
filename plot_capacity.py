"""
Plot the capacity results

@author: Yaqi
"""
import matplotlib.pyplot as plt
import pickle

data_root = "./capacity_result/180m_nlos_test2/"
save_fig = False

with open(data_root+'capacity_fc_0.pkl', 'rb') as handle:
    capacities = pickle.load(handle)

legend = []
x = [-180, -165, -150, -135, -120, -105, -90,-75, -60, -45, -30, -15, 0, 
     15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
for key in capacities:
    plt.plot(x, capacities[key])
    legend.append(key)
plt.xlim([-180,180])
plt.ylim([0,42])
plt.legend(legend, loc='upper right')
plt.xlabel('rotation (deg)')
plt.ylabel('capacity (bits/s/hertz)')
plt.grid()
plt.title('center frequency capacity with Tx rotation')
plt.savefig(data_root+'plots/capacity_fc.png', dpi=300)