# -*- coding: utf-8 -*-
"""
MIMOChan object

"""
from model.constants import DataConfig
from model.functions import comp_cmplx_gain, compute_capacity
from model.mpchan import MPChan
import numpy as np
import random


class MIMOChan(object):
    
    
    
    def __init__(self, ref_MPChan_df, tx_arr_size = [8,8], rx_arr_size = [8,8],
                 fc = 140e9, bw = 1e9, tx_rotate_ang = 0, freq_offset = None,
                 if_print = False):
        
        self.if_print = if_print
        self.tx_arr_size = tx_arr_size
        self.rx_arr_size = rx_arr_size
        self.nrx = self.rx_arr_size[0]*self.rx_arr_size[1]
        self.ntx = self.tx_arr_size[0]*self.tx_arr_size[1]
        self.fc = fc
        self.bw = bw
        # self.rnd_fc = random.sample(range(int(self.fc-self.bw), int(self.fc+self.bw)),1)[0]
        self.rnd_fc = fc+freq_offset
        
        self.ref_MPChan = MPChan(ref_MPChan_df, 
                                 fc = self.fc, bw = self.bw, 
                                 if_print = self.if_print)
        self.ref_MPChan.get_rx_arr_normal_vector_angs()
        self.ref_MPChan.get_tx_arr_normal_vector_angs(tx_rotate_ang)
        
    
    def init_MPChan_arr(self, MPChan_df_ls):
        """
        Parameters
        ----------
        MPChan_ls : list
            a list of MPChan objects in order of the dataframes

        Returns
        -------
        None.
        The matrix is in size of (Nrx, Ntx), and Nrx=Ntx=n*n.
        One dataframe is a Tx-Rx link, take n=2 as an example. We will put MPChan
        described by dataframe from row 1 to row 16, by the following order,
        [
         [16 12 8 4]
         [15 11 7 3]
         [14 10 6 2]
         [13 9  5 1]
         ]

        """
        MPChan_ls = []
        for i_test in range(self.ntx*self.nrx):
            MPChan_ls.append(MPChan(MPChan_df_ls[i_test], 
                                    fc = self.fc, 
                                    bw = self.bw))
            
        # init MPChan array with size(Nrx Ntx)
        mpchan_arr = [[0]*self.ntx for i in range(self.nrx)]
        
        current_Chan = 0
        
        for tx in range(self.ntx-1, -1, -1):
            for rx in range(self.nrx-1, -1, -1):
                mpchan_arr[rx][tx] = MPChan_ls[current_Chan]
                current_Chan += 1      
        self.mpchan_arr = mpchan_arr
                
        
        
        
    def fit_ref_RM_model(self, disp_MPChan_1_df, disp_MPChan_2_df):
        """

        Parameters
        ----------
        disp_MPChan_1 : MPChan
            a displaced multipath channel
        disp_MPChan_2 : MPChan
            a displaced multipath channel

        Returns
        -------
        None.
        """
        
        # fit RM model in ref_MPChan
        self.ref_MPChan.fit_RM(MPChan(disp_MPChan_1_df, 
                                      fc = self.fc, 
                                      bw = self.bw), 
                               MPChan(disp_MPChan_2_df, 
                                      fc = self.fc, 
                                      bw = self.bw))       
    
    
    def channel_matrix_true(self):
        chan_mx = np.empty([self.nrx,self.ntx], dtype = 'complex_')
        Etrue_mx = np.empty([self.nrx,self.ntx])
        # compute MPChan response matrix with size(Nrx Ntx)
        for rx in range(self.nrx):
            for tx in range(self.ntx):
                chan_mx[rx][tx], Etrue_mx[rx][tx] = comp_cmplx_gain(
                    -self.mpchan_arr[rx][tx].pl, 
                    np.deg2rad(self.mpchan_arr[rx][tx].phase), 
                    self.mpchan_arr[rx][tx].dly, 
                    # self.ref_MPChan.fc, self.ref_MPChan.fc,
                    # Use random fc
                    self.rnd_fc, self.ref_MPChan.fc,
                    self.mpchan_arr[rx][tx].ang[:,DataConfig.aoa_phi_ind], # aoa_az
                    self.mpchan_arr[rx][tx].ang[:,DataConfig.aoa_theta_ind], # aoa_el
                    self.ref_MPChan.normal_az_el,
                    self.mpchan_arr[rx][tx].ang[:, DataConfig.aod_phi_ind],
                    self.mpchan_arr[rx][tx].ang[:, DataConfig.aod_theta_ind],
                    self.ref_MPChan.normal_az_el_tx)
                                                               
        self.chan_mx_true = chan_mx
        self.Etrue_mx_true = Etrue_mx
    
    
    def channel_matrix_rm_est(self):   
        chan_mx = np.empty([self.nrx,self.ntx], dtype = 'complex_')
        Etrue_mx = np.empty([self.nrx,self.ntx])
        # compute MPChan response matrix with size(Nrx Ntx)
        for rx in range(self.nrx):
            for tx in range(self.ntx):
                chan_mx[rx][tx], Etrue_mx[rx][tx] = (
                    self.ref_MPChan.compute_Hest_from_RM(
                        self.mpchan_arr[rx][tx], rnd_freq=self.rnd_fc, use_rotate=True))
        
        # compute MPChan response matrix with size(Nrx Ntx)
        self.chan_mx_rm = chan_mx
        self.Eest_mx_rm = Etrue_mx
    
    def channel_matrix_pwa_est(self):    
        chan_mx = np.empty([self.nrx,self.ntx], dtype = 'complex_')
        Etrue_mx = np.empty([self.nrx,self.ntx])
        # compute MPChan response matrix with size(Nrx Ntx)
        for rx in range(self.nrx):
            for tx in range(self.ntx):
                chan_mx[rx][tx], Etrue_mx[rx][tx] = (
                    self.ref_MPChan.compute_Hest_from_PWA(
                        self.mpchan_arr[rx][tx], rnd_freq=self.rnd_fc, use_rotate=True))
        
        # compute MPChan response matrix with size(Nrx Ntx)
        self.chan_mx_pwa = chan_mx
        self.Eest_mx_pwa = Etrue_mx
        
        
    def channel_matrix_const_est(self):
        chan_mx = np.empty([self.nrx,self.ntx], dtype = 'complex_')
        Etrue_mx = np.empty([self.nrx,self.ntx])
        # compute MPChan response matrix with size(Nrx Ntx)
        for rx in range(self.nrx):
            for tx in range(self.ntx):
                chan_mx[rx][tx], Etrue_mx[rx][tx] = (
                    self.ref_MPChan.compute_Hest_from_const(
                        self.mpchan_arr[rx][tx], rnd_freq=self.rnd_fc, use_rotate=True))
        
        # compute MPChan response matrix with size(Nrx Ntx)
        self.chan_mx_const = chan_mx
        self.Eest_mx_const = Etrue_mx
    
    def mimo_capacity(self):
        print(DataConfig.tx_pow_dbm)
        Ex = 10**(0.1*(DataConfig.tx_pow_dbm))
        N0 = 10**(0.1*(-174 + 10*np.log10(2e9) + 3))
        print(f'[Ex/N0 (dB)] {10*np.log10(Ex/N0)}')
        print(f'[N0] (dB) {10*np.log10(N0)}')
        print(f'[Ex/N0] {Ex/N0}')
        print(f'[N0] {N0}')
        print(f'{DataConfig.tx_pow_dbm - 30 + 174 - 10*np.log10(1e9) - 3}')
        c=4
        print('[RM]')
        self.RM_capacity, RM_s = compute_capacity(Ex/N0, self.chan_mx_rm, c)
        print('[PWA]')
        self.PWA_capacity, PWA_s = compute_capacity(Ex/N0,self.chan_mx_pwa, c)
        print('[const]')
        self.const_capacity, const_s = compute_capacity(Ex/N0, self.chan_mx_const, c)
        print('[true]')
        self.true_capacity, true_s = compute_capacity(Ex/N0, self.chan_mx_true, c)
        
        
        print(f'[True] {self.true_capacity}, [RM] {self.RM_capacity}, \n[PWA] {self.PWA_capacity}, [Const] {self.const_capacity}')
        return self.true_capacity, true_s, self.RM_capacity, RM_s, self.PWA_capacity, PWA_s, self.const_capacity, const_s
        
        
        