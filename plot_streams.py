"""
Plot the capacity results

@author: Yaqi
"""
import matplotlib.pyplot as plt
import pickle

data_root = "./capacity_result/180m_nlos/"
save_fig = False

with open(data_root+'streams_no_rnd_180_speed_correct.pkl', 'rb') as handle:
    streams = pickle.load(handle)

legend = []
x = [-180, -165, -150, -135, -120, -105, -90,-75, -60, -45, -30, -15, 0, 
     15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
for key in streams:
    plt.plot(x, streams[key])
    legend.append(key)
plt.xlim([-180,180])
plt.ylim([0,26])
plt.legend(legend, loc='upper right')
plt.xlabel('rotation (deg)')
plt.ylabel('number of streams')
plt.grid()
plt.title('Test (b) NLOS')
plt.savefig(data_root+'plots/streams_with_tx_rotation_no_rnd_180_speed_correct.png', dpi=300)