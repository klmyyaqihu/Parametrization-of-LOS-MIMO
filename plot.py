# -*- coding: utf-8 -*-
"""
Plot the relative error result

Created on Fri May  6 11:48:07 2022

@author: Yaqi
"""
from model.mpchan import MPChan
from model.constants import DataConfig

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_root = "./data/42_pair/"
save_fig = False


"""
28GHz
"""
fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True,
                        figsize=(10,5.5))
fig.suptitle('(a) 28GHz')
# clear subplots
for ax in axs:
    ax.remove()
# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

# if case_name == "no_foliage_no_diffraction":
#     f.suptitle("Without Foliage and Diffraction")  
# if case_name == "add_foliage_add_diffraction":
#     f.suptitle("Add Foliage and Diffraction")  

subfigs[0].suptitle('Without Foliage and Diffraction')
axs = subfigs[0].subplots(nrows=1, ncols=4, sharex=True, sharey=True)

case_name = "no_foliage_no_diffraction"
fc = "28GHz"
ref_df_root = data_root+fc+"/"+case_name+"/Beijing_ref_fix.csv"
disp_1_df_root = data_root+fc+"/"+case_name+"/Beijing_1.0_fix.csv"
disp_2_df_root = data_root+fc+"/"+case_name+"/Beijing_2.0_fix.csv"
ref_df = pd.read_csv(ref_df_root)
disp_1_df = pd.read_csv(disp_1_df_root)
disp_2_df = pd.read_csv(disp_2_df_root)

lambda_value = ['5.0', '10.0', '50.0', '100.0']
disp_dist_ls = ['5 cm', '10 cm', '50 cm', '100 cm']
n_test = 43 # the number of Tx-Rx pairs

for i in range(len(lambda_value)):
    test_df_root = data_root+fc+"/"+case_name+"/Beijing_" + lambda_value[i] +"_fix.csv"
    test_df = pd.read_csv(test_df_root)
    rm_res = []
    pwa_res = []
    const_res = []
    for i_test in range(n_test):
        test_channel_num = int(i_test)
        
        ref_channel_df = ref_df.iloc[[test_channel_num]]
        ref_channel_df = ref_channel_df.reset_index()
        disp_1_channel_df = disp_1_df.iloc[[test_channel_num]]
        disp_1_channel_df = disp_1_channel_df.reset_index()
        disp_2_channel_df = disp_2_df.iloc[[test_channel_num]]
        disp_2_channel_df = disp_2_channel_df.reset_index()
        test_channel_df = test_df.iloc[[test_channel_num]]
        test_channel_df = test_channel_df.reset_index()
        
        if (np.isnan(ref_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_1_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_2_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(test_channel_df.at[0, DataConfig.paths_number])):
            continue
        
        # Test fit RM 
        ref_chan = MPChan(ref_channel_df)
        ref_chan.fit_RM(MPChan(disp_1_channel_df), MPChan(disp_2_channel_df))
        
        # Test generate three error list
        RM_ls, PWA_ls, const_ls = ref_chan.generate_error_res(
            MPChan(test_channel_df), ntest = 10, use_zero = True)
        
        rm_res.extend(RM_ls)
        pwa_res.extend(PWA_ls)
        const_res.extend(const_ls)
        
    x = np.sort(rm_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'r')
    x = np.sort(pwa_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'blue')
    x = np.sort(const_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'green')
    axs[i].set_title(disp_dist_ls[i])
    axs[i].grid()
    axs[i].set_xlim([1e-5, 1e-0])
    axs[i].set_ylim([0, 1])
    
axs[0].set_ylabel("eCDF")

subfigs[1].suptitle('Add Foliage and Diffraction')
axs = subfigs[1].subplots(nrows=1, ncols=4, sharex=True, sharey=True)

case_name = "add_foliage_add_diffraction"
fc = "28GHz"
ref_df_root = data_root+fc+"/"+case_name+"/Beijing_ref_fix.csv"
disp_1_df_root = data_root+fc+"/"+case_name+"/Beijing_1.0_fix.csv"
disp_2_df_root = data_root+fc+"/"+case_name+"/Beijing_2.0_fix.csv"
ref_df = pd.read_csv(ref_df_root)
disp_1_df = pd.read_csv(disp_1_df_root)
disp_2_df = pd.read_csv(disp_2_df_root)

lambda_value = ['5.0', '10.0', '50.0', '100.0']
n_test = 43 # the number of Tx-Rx pairs

for i in range(len(lambda_value)):
    test_df_root = data_root+fc+"/"+case_name+"/Beijing_" + lambda_value[i] +"_fix.csv"
    test_df = pd.read_csv(test_df_root)
    rm_res = []
    pwa_res = []
    const_res = []
    for i_test in range(n_test):
        test_channel_num = int(i_test)
        
        ref_channel_df = ref_df.iloc[[test_channel_num]]
        ref_channel_df = ref_channel_df.reset_index()
        disp_1_channel_df = disp_1_df.iloc[[test_channel_num]]
        disp_1_channel_df = disp_1_channel_df.reset_index()
        disp_2_channel_df = disp_2_df.iloc[[test_channel_num]]
        disp_2_channel_df = disp_2_channel_df.reset_index()
        test_channel_df = test_df.iloc[[test_channel_num]]
        test_channel_df = test_channel_df.reset_index()
        
        if (np.isnan(ref_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_1_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_2_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(test_channel_df.at[0, DataConfig.paths_number])):
            continue
        
        # Test fit RM 
        ref_chan = MPChan(ref_channel_df)
        ref_chan.fit_RM(MPChan(disp_1_channel_df), MPChan(disp_2_channel_df))
        
        # Test generate three error list
        RM_ls, PWA_ls, const_ls = ref_chan.generate_error_res(
            MPChan(test_channel_df), ntest = 10, use_zero = True)
        
        rm_res.extend(RM_ls)
        pwa_res.extend(PWA_ls)
        const_res.extend(const_ls)
        
    x = np.sort(rm_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'r')
    x = np.sort(pwa_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'blue')
    x = np.sort(const_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'green')   
    axs[i].set_title(disp_dist_ls[i])
    axs[i].grid()
    axs[i].set_xlim([1e-5, 1e-0])
    axs[i].set_ylim([0, 1])

axs[i].legend(["RM", "PWA", "Const"])
axs[0].set_ylabel("eCDF")
fig.supxlabel("Error")
if save_fig:
    fig.savefig("./plots/28GHz.png", dpi = 300)



"""
140 GHz
"""   
fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True,
                        figsize=(10,5.5))
fig.suptitle('(b) 140GHz')
# clear subplots
for ax in axs:
    ax.remove()
# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

subfigs[0].suptitle('Without Foliage and Diffraction')
axs = subfigs[0].subplots(nrows=1, ncols=4, sharex=True, sharey=True)

case_name = "no_foliage_no_diffraction"
fc = "140GHz"
ref_df_root = data_root+fc+"/"+case_name+"/Beijing_ref_fix.csv"
disp_1_df_root = data_root+fc+"/"+case_name+"/Beijing_1.0_fix.csv"
disp_2_df_root = data_root+fc+"/"+case_name+"/Beijing_2.0_fix.csv"
ref_df = pd.read_csv(ref_df_root)
disp_1_df = pd.read_csv(disp_1_df_root)
disp_2_df = pd.read_csv(disp_2_df_root)

lambda_value = ['5.0', '10.0', '50.0', '100.0']
disp_dist_ls = ['5 cm', '10 cm', '50 cm', '100 cm']
n_test = 43 # the number of Tx-Rx pairs

for i in range(len(lambda_value)):
    test_df_root = data_root+fc+"/"+case_name+"/Beijing_" + lambda_value[i] +"_fix.csv"
    test_df = pd.read_csv(test_df_root)
    rm_res = []
    pwa_res = []
    const_res = []
    for i_test in range(n_test):
        test_channel_num = int(i_test)
        
        ref_channel_df = ref_df.iloc[[test_channel_num]]
        ref_channel_df = ref_channel_df.reset_index()
        disp_1_channel_df = disp_1_df.iloc[[test_channel_num]]
        disp_1_channel_df = disp_1_channel_df.reset_index()
        disp_2_channel_df = disp_2_df.iloc[[test_channel_num]]
        disp_2_channel_df = disp_2_channel_df.reset_index()
        test_channel_df = test_df.iloc[[test_channel_num]]
        test_channel_df = test_channel_df.reset_index()
        
        if (np.isnan(ref_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_1_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_2_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(test_channel_df.at[0, DataConfig.paths_number])):
            continue
        
        # Test fit RM 
        ref_chan = MPChan(ref_channel_df, fc = 140e9, bw = 1e9)
        ref_chan.fit_RM(MPChan(disp_1_channel_df), MPChan(disp_2_channel_df))
        
        # Test generate three error list
        RM_ls, PWA_ls, const_ls = ref_chan.generate_error_res(
            MPChan(test_channel_df), ntest = 10, use_zero = True)
        
        rm_res.extend(RM_ls)
        pwa_res.extend(PWA_ls)
        const_res.extend(const_ls)
        
    x = np.sort(rm_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'r')
    x = np.sort(pwa_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'blue')
    x = np.sort(const_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'green')
    axs[i].set_title(disp_dist_ls[i])
    axs[i].grid()
    axs[i].set_xlim([1e-5, 1e-0])
    axs[i].set_ylim([0, 1])
    
axs[0].set_ylabel("eCDF")

subfigs[1].suptitle('Add Foliage and Diffraction')
axs = subfigs[1].subplots(nrows=1, ncols=4, sharex=True, sharey=True)

case_name = "add_foliage_add_diffraction"
fc = "140GHz"
ref_df_root = data_root+fc+"/"+case_name+"/Beijing_ref_fix.csv"
disp_1_df_root = data_root+fc+"/"+case_name+"/Beijing_1.0_fix.csv"
disp_2_df_root = data_root+fc+"/"+case_name+"/Beijing_2.0_fix.csv"
ref_df = pd.read_csv(ref_df_root)
disp_1_df = pd.read_csv(disp_1_df_root)
disp_2_df = pd.read_csv(disp_2_df_root)

lambda_value = ['5.0', '10.0', '50.0', '100.0']
n_test = 43 # the number of Tx-Rx pairs

for i in range(len(lambda_value)):
    test_df_root = data_root+fc+"/"+case_name+"/Beijing_" + lambda_value[i] +"_fix.csv"
    test_df = pd.read_csv(test_df_root)
    rm_res = []
    pwa_res = []
    const_res = []
    for i_test in range(n_test):
        test_channel_num = int(i_test)
        
        ref_channel_df = ref_df.iloc[[test_channel_num]]
        ref_channel_df = ref_channel_df.reset_index()
        disp_1_channel_df = disp_1_df.iloc[[test_channel_num]]
        disp_1_channel_df = disp_1_channel_df.reset_index()
        disp_2_channel_df = disp_2_df.iloc[[test_channel_num]]
        disp_2_channel_df = disp_2_channel_df.reset_index()
        test_channel_df = test_df.iloc[[test_channel_num]]
        test_channel_df = test_channel_df.reset_index()
        
        if (np.isnan(ref_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_1_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(disp_2_channel_df.at[0, DataConfig.paths_number]) or
            np.isnan(test_channel_df.at[0, DataConfig.paths_number])):
            continue
        
        # Test fit RM 
        ref_chan = MPChan(ref_channel_df, fc = 140e9, bw = 1e9)
        ref_chan.fit_RM(MPChan(disp_1_channel_df), MPChan(disp_2_channel_df))
        
        # Test generate three error list
        RM_ls, PWA_ls, const_ls = ref_chan.generate_error_res(
            MPChan(test_channel_df), ntest = 10, use_zero = True)
        
        rm_res.extend(RM_ls)
        pwa_res.extend(PWA_ls)
        const_res.extend(const_ls)
        
    x = np.sort(rm_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'r')
    x = np.sort(pwa_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'blue')
    x = np.sort(const_res)
    y = np.arange(len(x))/float(len(x))
    axs[i].semilogx(x, y, c = 'green')   
    axs[i].set_title(disp_dist_ls[i])
    axs[i].grid()
    axs[i].set_xlim([1e-5, 1e-0])
    axs[i].set_ylim([0, 1])
    
axs[i].legend(["RM", "PWA", "Const"])
axs[0].set_ylabel("eCDF")
fig.supxlabel("Error")
if save_fig:
    fig.savefig("./plots/140GHz.png", dpi = 300)