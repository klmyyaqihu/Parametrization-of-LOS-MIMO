"""
Plot the capacity results

@author: Yaqi
"""
import matplotlib.pyplot as plt
import pickle

data_root = "./capacity_result/180m_nlos_test2/"
save_fig = False

files = []
freq_offset_ls = range(int(-1e9), int(1e9 + 1e8), int(2e8))
for offset in freq_offset_ls:
    offset_str = str(int(offset / 1e8))
    files.append("capacity_fc_" + offset_str + ".pkl")


with open(data_root + files[0], 'rb') as handle:
    capacities = pickle.load(handle)

for f in files[1:]:
    with open(data_root + f, 'rb') as handle:
        temp = pickle.load(handle)
        for key in capacities:
            for idx in range(len(temp[key])):
                capacities[key][idx] += temp[key][idx]
                if f == files[-1]:
                    capacities[key][idx] /= len(files)


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
plt.title('spectral efficiency with Tx rotation')
plt.savefig(data_root+'plots/capacity_average.png', dpi=300)