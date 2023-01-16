# -*- coding: utf-8 -*-
"""
Test a single case

@author: Yaqi and Mingsheng
"""
from model.mimochan import MIMOChan
import pandas as pd
import pickle
import os

data_root = "./data/mimo_180m_NLOS_test2/"

ref_df_root = data_root + "Beijing_ref_fix.csv"
disp_1_df_root = data_root + "Beijing_disp7_fix.csv"
disp_2_df_root = data_root + "Beijing_disp6_fix.csv"

ref_df = pd.read_csv(ref_df_root)
ref_channel_df = ref_df.iloc[[0]]
ref_channel_df = ref_channel_df.reset_index()

disp_1_df = pd.read_csv(disp_1_df_root)
disp_1_channel_df = disp_1_df.iloc[[0]]
disp_1_channel_df = disp_1_channel_df.reset_index()

disp_2_df = pd.read_csv(disp_2_df_root)
disp_2_channel_df = disp_2_df.iloc[[0]]
disp_2_channel_df = disp_2_channel_df.reset_index()


# rotations = ['-90', '-75', '-60', '-45', '-30', '-15', '+0', '+15', '+30', '+45', '+60', '+75', '+90']
rotations = ['-180','-165', '-150', '-135', '-120', '-105', '-90', '-75', '-60', '-45', '-30', '-15', '+0', 
              '+15', '+30', '+45', '+60', '+75', '+90', '+105', '+120', '+135', '+150', '+165', '+180']

# rotations = ['+0']

freq_offset_ls = range(int(-1e9), int(1e9 + 1e8), int(2e8))
# freq_offset_ls = range(int(-1e9), int(1e9 + 1e8), int(1e9))
for freq_offset in freq_offset_ls:
    capacity = {
        'true': [],
        'RM': [],
        'PWA': [],
        'const': []
        }
    
    streams = {
        'true': [],
        'RM': [],
        'PWA': [],
        'const': []
        }
    
    for rotation in rotations:
        print(f'current rotation: {rotation}')
        # Test compute the error
        valid_df_root = data_root + "fix/Beijing_8x8mimo_apg"+rotation+"_fix.csv"
        valid_df = pd.read_csv(valid_df_root)
        MPChan_df_ls = []
        for i_test in range(4096):
        
            valid_channel_df = valid_df.iloc[[i_test]]
            valid_channel_df = valid_channel_df.reset_index()
            
            MPChan_df_ls.append(valid_channel_df)
        
            
        mimo_chan = MIMOChan(ref_channel_df, fc = 140e9, bw = 1e9, 
                             tx_rotate_ang = float(rotation), 
                             freq_offset = freq_offset,
                             if_print = False)
        mimo_chan.init_MPChan_arr(MPChan_df_ls)
        
        chan = mimo_chan.mpchan_arr[0][0]
        mimo_chan.channel_matrix_true()
        
        mimo_chan.fit_ref_RM_model(disp_1_channel_df, disp_2_channel_df)
        mimo_chan.channel_matrix_rm_est()
        mimo_chan.channel_matrix_pwa_est()
        mimo_chan.channel_matrix_const_est()
        
        true_capacity, true_s, RM_capacity, RM_s, PWA_capacity, PWA_s, const_capacity, const_s = mimo_chan.mimo_capacity()
        capacity['true'].append(true_capacity)
        capacity['RM'].append(RM_capacity)
        capacity['PWA'].append(PWA_capacity)
        capacity['const'].append(const_capacity)
        streams['true'].append(true_s)
        streams['RM'].append(RM_s)
        streams['PWA'].append(PWA_s)
        streams['const'].append(const_s)
    
    str_freq_offset = str(int(freq_offset / 1e8))
    save_dir = "./capacity_result/180m_nlos_test2/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    f = open(save_dir+"capacity_fc_" + str_freq_offset + ".pkl", "wb")
    pickle.dump(capacity, f)
    f.close()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    f = open(save_dir+"streams_fc_" + str_freq_offset + ".pkl", "wb")
    pickle.dump(streams, f)
    f.close()