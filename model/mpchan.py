# -*- coding: utf-8 -*-
"""
Multi-path Channel object
With functions to 
    Fit the reflection model    
        Paths matching
        Compute gamma angles by TWO displaced pairs
    Reflection model complex channel gain error
    PWA model complex channel gain error
    Constant model complex channel gain error

@author: Yaqi, Mingsheng
@date: May 05 2022
"""

from model.constants import LinkState, DataConfig, PhyConst
from model.spherical import cart_to_sph_scalar
from model.functions import comp_cmplx_gain, d0, am, Am, Bm, Cm, u
from model.functions import est_dist_by_RM, est_dist_by_PWA, compute_arbitrary_phase

import numpy as np
import random

class MPChan(object):
    
    """
    Class for a multi-path channel from ray tracing
    
    With functions to 
        fit_RM() : Fit the reflection model    
                   path_match() : Paths matching
                   find_gamma() : Compute gamma angles by TWO displaced pairs
        compute_RM_error() : Reflection model complex channel gain error
        compute_PWA_error() : PWA model complex channel gain error
        compute_const_error() : Constant model complex channel gain error
    
    """
    nangle = DataConfig.nangle
    aoa_phi_ind = DataConfig.aoa_phi_ind    #[-pi, pi] azimuth
    aoa_theta_ind = DataConfig.aoa_theta_ind   #[-pi/2, pi/2] elevation
    aod_phi_ind = DataConfig.aod_phi_ind
    aod_theta_ind = DataConfig.aod_theta_ind
    ang_name = DataConfig.ang_name
    
    def __init__(self, data_frame, fc = 28e9, bw = 200e6, is_outage = False,
                 if_print = False):
        """
        MPChan Constructor from data_frame
        
        -----------
        npath : number of path of this MPChan
        tx_coord and rx_coord: np.array like [x, y, z]
        dly : second
        pl : dBm
        phase : degree
        angles : radian
        interact : string like "Tx-R-R-Rx"
        num_interaction : int number of interaction, where LOS = 0
        link_state : LinkState values
        """
        
        self.if_print = if_print
        # Parameters for each path
        if is_outage:
            self.link_state = LinkState.no_link
        else:
            self.fc = fc
            self.bw = bw
            self.npath = int(data_frame.at[0, DataConfig.paths_number])
            self.tx_coord = data_frame.at[0, DataConfig.tx_coord][1:-1]
            self.tx_coord = np.fromstring(self.tx_coord, sep=' ',dtype = float)
            self.rx_coord = data_frame.at[0, DataConfig.rx_coord][1:-1]
            self.rx_coord = np.fromstring(self.rx_coord, sep=' ',dtype = float)
            self.pl = np.zeros((self.npath)) # db
            self.dly = np.zeros((self.npath)) # second
            self.phase = np.zeros((self.npath))
            self.ang = np.zeros((self.npath, MPChan.nangle)) #rad
            self.interact = [] # interaction list wiht string
            self.num_interact = np.zeros((self.npath))
            
            for ipath in range(self.npath):
                path_id = ipath + 1 # path id starts from 1
                self.pl[ipath] = DataConfig.tx_pow_dbm - \
                    10*np.log10(1000*data_frame.at[0, str(path_id)+
                                                   DataConfig.path_rcvpow]) # dBm
                self.dly[ipath] = data_frame.at[0, 
                        str(path_id)+DataConfig.path_dly]
                self.phase[ipath] = data_frame.at[0, 
                                  str(path_id)+DataConfig.path_phase]
                self.ang[ipath, 0] = np.deg2rad(\
                        data_frame.at[0, str(path_id)+DataConfig.path_aoa_phi])
                self.ang[ipath, 1] = np.deg2rad(90 
                        -data_frame.at[0, str(path_id)+DataConfig.path_aoa_theta])
                self.ang[ipath, 2] = np.deg2rad(
                        data_frame.at[0, str(path_id)+DataConfig.path_aod_phi])
                self.ang[ipath, 3] = np.deg2rad(90 
                        -data_frame.at[0, str(path_id)+DataConfig.path_aod_theta])
                self.interact.append(\
                         data_frame.at[0, str(path_id)+DataConfig.path_interact])
                self.num_interact[ipath] = data_frame.at[0, 
                         str(path_id)+DataConfig.path_num_interact]
            
            if 0 in self.num_interact: # LOS path has 0 interactions
                self.link_state = LinkState.los_link
            else:
                self.link_state = LinkState.nlos_link
    
    
    def get_rx_arr_normal_vector_angs(self):
        normal = self.tx_coord - self.rx_coord
        # print(normal)
        _, az, inc = cart_to_sph_scalar(normal)
        el = 90-inc
        self.normal_az_el = [az, el]
        if self.if_print:
            print(f'[Rx Normal V] Az = {az}, El = {el}')
        
    def get_tx_arr_normal_vector_angs(self, rotate_ang):
        normal = self.rx_coord - self.tx_coord
        # print(normal)
        _, az, inc = cart_to_sph_scalar(normal)
        el = 90-inc 
        az = az+rotate_ang
        self.normal_az_el_tx = [az, el]
        if self.if_print:
            print(f'[Tx Normal V] Az = {az}, El = {el}')
        
    
    def fit_RM(self, displaced_MPChan_1, displaced_MPChan_2):
        """
        Fit the current reference MPChan in Reflection Model.
        The output is the gamma value of current all paths.
        If the value of gamma = -1, means this path does not have path to match
        in displaces MPChan

        Parameters
        ----------
        displaced_MPChan_1 : MPChan
            The first displaced MPChan uses to compute gamma.
        displaced_MPChan_2 : MPChan
            The second displaced MPChan uses to compute gamma.

        Returns
        -------
        gamma list

        """
        
        # Path match: indicate the path number in the another MPChan
        path_map_1 = self.path_match(displaced_MPChan_1)
        path_map_2 = self.path_match(displaced_MPChan_2)
        if self.if_print:
            print("[Start] Path Match")
            print(f"[Res]   {path_map_1} and {path_map_2}")
        
        s_value = [-1, 1]
        self.gamma = np.zeros([self.npath])
        # indicate if gamma is valid
        self.gamma_valid = np.zeros([self.npath])
        # save S value
        self.s_ls = np.zeros([self.npath])
        
        # Compute gamma angles for each path
        for ipath in range(self.npath):
            interact = self.interact[ipath].split('-')
            s_id = interact.count('R')%2
            self.s_ls[ipath] = s_value[s_id]
            
            if path_map_1[ipath] == -1 and path_map_2[ipath] == -1: continue
            

            
            self.gamma[ipath] = self.find_gamma(ipath,
                                                s_value[s_id],
                                                path_map_1[ipath],
                                                displaced_MPChan_1,
                                                path_map_2[ipath],
                                                displaced_MPChan_2)
            self.gamma_valid[ipath] = 1
        if self.if_print:
            print("[Start] Compute Gamma")
            print(f"[Res]   Gamma Value : {self.gamma}")
            print(f"[Res]   S     Value : {self.s_ls}")
            print(f"[Res]   Gamma Valid : {self.gamma_valid}")
            
            
    def find_gamma(self, ipath,
                   s, path_id_1, displaced_MPChan_1,
                   path_id_2, displaced_MPChan_2):
        """
        Find gamma value for a matched path
    
        Parameters
        ----------
        ref_chan
        
        path_id
        
        ref_path_id
    
        Returns
        -------
        Gamma angles value in radian.
    
        """
        # aoa and aod
        aoa_az = self.ang[ipath, DataConfig.aoa_phi_ind] # receiver 
        aoa_el = self.ang[ipath, DataConfig.aoa_theta_ind]
        aod_az = self.ang[ipath, DataConfig.aod_phi_ind] # transimitter
        aod_el = self.ang[ipath, DataConfig.aod_theta_ind]
        
        # direction of arrival
        u_r = u(aoa_az, aoa_el)

        # direction of departure
        u_t = u(aod_az, aod_el)
        
        A_ls = []
        B_ls = []
        C_ls = []
        
        # compute params from displaced_MPChan_1
        if path_id_1 != -1:
            am_r = am(aoa_el, aoa_az, self.rx_coord, 
                      displaced_MPChan_1.rx_coord)
            am_t = am(aod_el, aod_az, self.tx_coord, 
                      displaced_MPChan_1.tx_coord)
            d_sq = (displaced_MPChan_1.dly[path_id_1] * 
                    PhyConst.light_speed_remcom) ** 2
            d_0 = d0(self.dly[ipath], self.rx_coord, self.tx_coord,
                     displaced_MPChan_1.rx_coord, displaced_MPChan_1.tx_coord, 
                     u_t, u_r)
            A_ls.append(Am(am_r, am_t, s))
            B_ls.append(Bm(am_r, am_t, s))
            C_ls.append(Cm(am_r, am_t, d_sq, d_0))
            
        # compute params from displaced_MPChan_2
        if path_id_2 != -1:
            am_r = am(aoa_el, aoa_az, self.rx_coord, 
                      displaced_MPChan_2.rx_coord)
            am_t = am(aod_el, aod_az, self.tx_coord, 
                      displaced_MPChan_2.tx_coord)
            d_sq = (displaced_MPChan_2.dly[path_id_2] * 
                    PhyConst.light_speed_remcom) ** 2
            d_0 = d0(self.dly[ipath], self.rx_coord, self.tx_coord,
                      displaced_MPChan_2.rx_coord, displaced_MPChan_2.tx_coord, 
                      u_t, u_r)
            A_ls.append(Am(am_r, am_t, s))
            B_ls.append(Bm(am_r, am_t, s))
            C_ls.append(Cm(am_r, am_t, d_sq, d_0))
    
    
        cos, sin = np.linalg.lstsq(np.vstack([A_ls,B_ls]).T, C_ls, rcond=None)[0]
        gamma = np.arctan2(sin, cos)

        return gamma
        
        
    def path_match(self, mp_chan, TH = 3):
        """
        Path match with another MPChan object.

        Parameters
        ----------
        mp_chan : MPChan object
            Another MPChan. 
            It can be the displaced TX-RX channel.
            (for computing gamma angle)
            And it also can be another displaced validation TX-RX channel 
            (for computing error)
        TH : threshold

        Returns
        -------
        path map : -1 means not match

        """
        
        # list indicate that if the path already matched with other
        path_number_used = [0]*mp_chan.npath 
        # draft choice from angles matching 
        # (this is because maybe more paths matched the same path)
        # then we need to match again by delay
        draft_id = []
        
        for i_path in range(self.npath):
            # Match by angles
            angs = self.ang[i_path, :].squeeze() # (4,) path's angles
            # calculate the angles abs difference between this path with 
            # all paths in other MPChan.
            angs_diff = np.sum(np.abs(angs-mp_chan.ang), axis=1)
            # find delay diff
            dly_diff = np.abs(self.dly[i_path] - mp_chan.dly)
            
            idx_by_angs = np.argmin(angs_diff)
            ref_path_interaction = mp_chan.interact[idx_by_angs]
            
            # check if interaction ==
            if (self.interact[i_path] == ref_path_interaction) and\
                (abs(np.argmin(angs_diff) - np.argmin(dly_diff)) < TH):
                # get the id
                draft_id.append(np.argmin(angs_diff)) # path start from 1
            else:
                # append -1
                draft_id.append(-1)

        # check if we matched two paths with the same reference tx-rx path
        for i_path in range(self.npath):
            # if -1 means not match any path
            if draft_id[i_path] != -1:
                path_number_used[draft_id[i_path]] += 1

        if any(x > 1 for x in path_number_used):
            # weaker path go -1
            path_number_used = [0]*mp_chan.npath
            for i_path in range(self.npath):
                if draft_id[i_path] != -1:
                    if path_number_used[draft_id[i_path]] == 0:
                        path_number_used[draft_id[i_path]] = 1
                    else:
                        draft_id[i_path] = -1
            if self.if_print:
                print("[Match] Matched two paths with the same reference tx-rx path")
                            
        else:
            if self.if_print:
                print("[Match] No problem for matching paths")
                print(f"[Res]   {draft_id}")
                
        return draft_id
    
    
    def compute_Hest_from_RM(self, test_MPChan, use_rnd = False, rnd_freq = None,
                         use_zero = True, use_rotate = False):
        """
        Compute RM estimation error for test_MPChan

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        use_rnd : boolean
            if true use random frequency, if not use self.fc
        rnd_freq : float,
            randomized frequency around center frequency, in Hz
        use_zero : bool, optional
            If true, using gamma = zero when gamma is not available.
            else using PWA
            The default is True.

        Returns
        -------
        Hest : complex channel gain         
        Eest : Eavg 

        """
        phase_ls = []
        gain_ls = []
        dly_ls = []
        
        
        # Loop in path of ref_chan(self)
        for i_path in range(self.npath):
            aoa_az = self.ang[i_path, DataConfig.aoa_phi_ind] # receiver 
            aoa_el = self.ang[i_path, DataConfig.aoa_theta_ind]
            aod_az = self.ang[i_path, DataConfig.aod_phi_ind] # transimitter
            aod_el = self.ang[i_path, DataConfig.aod_theta_ind]
            # Use gamma = 0 when path has no match
            if use_zero:
                # compute estimated distance by RM eq(13)
                est_dist = est_dist_by_RM(self.dly[i_path], aoa_el, 
                                          aoa_az, self.rx_coord, 
                                          test_MPChan.rx_coord, aod_el, 
                                          aod_az, self.tx_coord, 
                                          test_MPChan.tx_coord, self.s_ls[i_path], 
                                          self.gamma[i_path])
            
            # if not use_zero then we use PWA when no match
            if not use_zero:
                # if valid gamma
                if self.gamma_valid[i_path]:
                    # compute estimated distance by (13)
                    est_dist = est_dist_by_RM(self.dly[i_path], aoa_el, 
                                              aoa_az, self.rx_coord, 
                                              test_MPChan.rx_coord, aod_el, 
                                              aod_az, self.tx_coord, 
                                              test_MPChan.tx_coord, self.s_ls[i_path], 
                                              self.gamma[i_path])  
                        
                # else use PWA
                else:
                    u_r = u(aoa_az, aoa_el)
                    u_t = u(aod_az, aod_el)
                    
                    # compute estimated distance by PWA eq(5)
                    est_dist =  est_dist_by_PWA(self.dly[i_path], u_r, u_t, 
                                                self.rx_coord, test_MPChan.rx_coord,
                                                self.tx_coord, test_MPChan.tx_coord)
            
            # compute arbitrary phase
            arbitrary_phase = compute_arbitrary_phase(self.fc, self.dly[i_path], 
                                                      self.phase[i_path])
            
            phase_from_est_dist = 2*np.pi - (
                2*np.pi*est_dist*self.fc/PhyConst.light_speed_remcom) % (2*np.pi)
            phase_from_est_dist = phase_from_est_dist + arbitrary_phase
            dly_from_est_dist = est_dist/PhyConst.light_speed_remcom
            
            # store result
            phase_ls.append(phase_from_est_dist)
            gain_ls.append(-self.pl[i_path])
            dly_ls.append(dly_from_est_dist)
        
        if use_rotate:
            Hest, Eest = comp_cmplx_gain(gain_ls, phase_ls, dly_ls, rnd_freq, self.fc,
                                         self.ang[:, DataConfig.aoa_phi_ind],
                                         self.ang[:, DataConfig.aoa_theta_ind],
                                         self.normal_az_el,
                                         self.ang[:, DataConfig.aod_phi_ind],
                                         self.ang[:, DataConfig.aod_theta_ind],
                                         self.normal_az_el_tx
                                        )
        else:
            # Compute Hest, Eest
            Hest, Eest = comp_cmplx_gain(gain_ls, phase_ls, dly_ls, rnd_freq, self.fc)
        return Hest, Eest
    
    
    def compute_RM_error(self, test_MPChan, use_rnd = False, rnd_freq = None,
                         use_zero = True):
        """
        Compute RM estimation error for test_MPChan

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        use_rnd : boolean
            if true use random frequency, if not use self.fc
        rnd_freq : float,
            randomized frequency around center frequency, in Hz
        use_zero : bool, optional
            If true, using gamma = zero when gamma is not available.
            else using PWA
            The default is True.

        Returns
        -------
        Relative error. Float 
            np.abs(Hest-Htrue)**2/Eest

        """
        # compute random 
        if not use_rnd:
            rnd_freq = self.fc
         
        # Compute Hest, Eest
        Hest, Eest = self.compute_Hest_from_RM(test_MPChan, use_rnd, rnd_freq,
                             use_zero)
        
        
        # Compute Htrue, Etrue
        Htrue, Etrue = comp_cmplx_gain(-test_MPChan.pl, np.deg2rad(test_MPChan.phase), 
                                       test_MPChan.dly, rnd_freq, self.fc)
        # Compute Relative Err
        rel_err = np.abs(Hest-Htrue)**2/Eest
        if self.if_print:
            print(f'[Res]   RM  Rel_err = {rel_err}')
        
        return rel_err


    def compute_Hest_from_PWA(self, test_MPChan, use_rnd = False, rnd_freq = None, 
                              use_rotate = False):
        """
        Compute PWA estimation error for test_MPChan

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        use_rnd : boolean
            if true use random frequency, if not use self.fc
        rnd_freq : float,
            randomized frequency around center frequency, in Hz

        Returns
        -------
        Hest : complex channel gain         
        Eest : Eavg 
        """
        phase_ls = []
        gain_ls = []
        dly_ls = []
        
        # Loop in path of ref_chan(self)
        for i_path in range(self.npath):
            aoa_az = self.ang[i_path, DataConfig.aoa_phi_ind] # receiver 
            aoa_el = self.ang[i_path, DataConfig.aoa_theta_ind]
            aod_az = self.ang[i_path, DataConfig.aod_phi_ind] # transimitter
            aod_el = self.ang[i_path, DataConfig.aod_theta_ind]
            
            u_r = u(aoa_az, aoa_el)
            u_t = u(aod_az, aod_el)
            
            # compute estimated distance by PWA eq(5)
            est_dist =  est_dist_by_PWA(self.dly[i_path], u_r, u_t, 
                                        self.rx_coord, test_MPChan.rx_coord,
                                        self.tx_coord, test_MPChan.tx_coord)
            
            # compute arbitrary phase
            arbitrary_phase = compute_arbitrary_phase(self.fc, self.dly[i_path], 
                                                      self.phase[i_path])
            
            phase_from_est_dist = 2*np.pi - (
                2*np.pi*est_dist*self.fc/PhyConst.light_speed_remcom) % (2*np.pi)
            phase_from_est_dist = phase_from_est_dist + arbitrary_phase
            dly_from_est_dist = est_dist/PhyConst.light_speed_remcom
            
            # store result
            phase_ls.append(phase_from_est_dist)
            gain_ls.append(-self.pl[i_path])
            dly_ls.append(dly_from_est_dist)
        
        if use_rotate:
            Hest, Eest = comp_cmplx_gain(gain_ls, phase_ls, dly_ls, rnd_freq, self.fc,
                                         self.ang[:, DataConfig.aoa_phi_ind],
                                         self.ang[:, DataConfig.aoa_theta_ind],
                                         self.normal_az_el,
                                         self.ang[:, DataConfig.aod_phi_ind],
                                         self.ang[:, DataConfig.aod_theta_ind],
                                         self.normal_az_el_tx
                                        )    
        else:
            # Compute Hest, Eest
            Hest, Eest = comp_cmplx_gain(gain_ls, phase_ls, dly_ls, rnd_freq, self.fc)
        return Hest, Eest
        


    def compute_PWA_error(self, test_MPChan, use_rnd = False, rnd_freq = None):
        """
        Compute PWA estimation error for test_MPChan

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        use_rnd : boolean
            if true use random frequency, if not use self.fc
        rnd_freq : float,
            randomized frequency around center frequency, in Hz

        Returns
        -------
        Relative error. Float 
            np.abs(Hest-Htrue)**2/Eest
        """
        # compute random 
        if not use_rnd:
            rnd_freq = self.fc
            
        Hest, Eest = self.compute_Hest_from_PWA(test_MPChan, use_rnd, rnd_freq)
        # Compute Htrue, Etrue
        Htrue, Etrue = comp_cmplx_gain(-test_MPChan.pl, 
                                       np.deg2rad(test_MPChan.phase), 
                                       test_MPChan.dly, rnd_freq, self.fc)
        # Compute Relative Err
        rel_err = np.abs(Hest-Htrue)**2/Eest
        
        if self.if_print:
            print(f'[Res]   PWA Rel_err = {rel_err}')
        
        return rel_err
                 
    
    def compute_Hest_from_const(self, test_MPChan, use_rnd = False, 
                                rnd_freq = None, use_rotate = False):
        """
        Compute constant model error for test_MPChan
        Constant model is that we directly use ref_channel h as our estmated h

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        use_rnd : boolean
            if true use random frequency, if not use self.fc
        rnd_freq : float,
            randomized frequency around center frequency, in Hz

        Returns
        -------
        Hest : complex channel gain         
        Eest : Eavg 
        """
        if use_rotate:
            Hest, Eest = comp_cmplx_gain(-self.pl, self.phase, self.dly,
                                         rnd_freq, self.fc,
                                         self.ang[:, DataConfig.aoa_phi_ind],
                                         self.ang[:, DataConfig.aoa_theta_ind],
                                         self.normal_az_el,
                                         self.ang[:, DataConfig.aod_phi_ind],
                                         self.ang[:, DataConfig.aod_theta_ind],
                                         self.normal_az_el_tx
                                         )  
        else:
            # Compute Hest, Eest
            Hest, Eest = comp_cmplx_gain(-self.pl, self.phase, self.dly,
                                         rnd_freq, self.fc)
        return Hest, Eest
        
    
    def compute_const_error(self, test_MPChan, use_rnd = False, rnd_freq = None):
        """
        Compute constant model error for test_MPChan
        Constant model is that we directly use ref_channel h as our estmated h

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        use_rnd : boolean
            if true use random frequency, if not use self.fc
        rnd_freq : float,
            randomized frequency around center frequency, in Hz

        Returns
        -------
        Relative error. Float 
            np.abs(Hest-Htrue)**2/Eest
        """
        # compute random 
        if not use_rnd:
            rnd_freq = self.fc
        
        Hest, Eest = self.compute_Hest_from_const(test_MPChan, use_rnd, rnd_freq)
        # Compute Htrue, Etrue
        Htrue, Etrue = comp_cmplx_gain(
            -test_MPChan.pl, np.deg2rad(test_MPChan.phase), 
            test_MPChan.dly, rnd_freq, self.fc)
        # Compute Relative Err
        rel_err = np.abs(Hest-Htrue)**2/Eest
        if self.if_print:
            print(f'[Res]   Constant Rel_err = {rel_err}')
        
        return rel_err
    
    
    def generate_error_res(self, test_MPChan, ntest = 10, use_zero = True):
        """

        Parameters
        ----------
        test_MPChan : MPChan
            Validation MPChan.
        ntest : int, optional
            number of random frequencies. The default is 10.
        use_zero : bool, optional
            If true, using gamma = zero when gamma is not available.
            else using PWA
            The default is True.

        Returns
        -------
        Three error list. Numpy array (ntest,)

        """
        RM_ls = np.empty([ntest,])
        PWA_ls = np.empty([ntest,])
        const_ls = np.empty([ntest,])       
        
        rnd_freq_arr = np.array(random.sample(range(int(self.fc-self.bw), 
                                                    int(self.fc+self.bw)), 
                                              ntest), dtype=float)
        for ifc, fc in enumerate(rnd_freq_arr):
            RM_ls[ifc] = self.compute_RM_error(test_MPChan, True, fc, use_zero)
            PWA_ls[ifc] = self.compute_PWA_error(test_MPChan, True, fc)
            const_ls[ifc] = self.compute_const_error(test_MPChan, True, fc)
        
        return RM_ls, PWA_ls, const_ls
        
