# -*- coding: utf-8 -*-
"""
Common functions support the MPChan object

@author: Yaqi and Mingsheng
"""
from model.constants import PhyConst, Rx, Ry, Rz, Qz
import numpy as np
import model.antenna as ant
from model.spherical import rotate_by_los_x_arr

def comp_cmplx_gain(gain, phase, delay, rnd_freq, center_freq, 
                    aoa_az=None, aoa_el=None, rx_ref_angs=None,
                    aod_az=None, aod_el=None, tx_ref_angs=None):
        
    """
    Calculate complex channel gain and Eavg
    
    Returns
    -------
    complex channel gain
    Eavg 
    """
    h = 0
    Eavg = 0
    delta = rnd_freq - center_freq
    if aoa_az is None:
        ant_gain = np.zeros(len(gain))
    else:
        rx_ref_az, rx_ref_el = rotate_by_los_x_arr(rx_ref_angs[0], rx_ref_angs[1], 
                                           (aoa_az*180)/np.pi, (aoa_el*180)/np.pi)
        tx_ref_az, tx_ref_el = rotate_by_los_x_arr(tx_ref_angs[0], tx_ref_angs[1], 
                                           (aod_az*180)/np.pi, (aod_el*180)/np.pi)

        elem = ant.Elem3GPP()
        ant_gain = elem.response(tx_ref_az, tx_ref_el) + elem.response(rx_ref_az, rx_ref_el)

    # g_complex = []
    for i in range(len(gain)):
        g = np.sqrt(10 ** ((gain[i]+ant_gain[i])/10)) # convert dB to linear
        # g_complex.append(g * np.exp((phase[i]+2*np.pi*delta*delay[i])*1j))
        
        h += g * np.exp((phase[i]+2*np.pi*delta*delay[i])*1j) 
        Eavg += g**2
    
    # g_complex_norm = g_complex / np.linalg.norm(g_complex, 2)
    # h = np.sum(g_complex_norm)

    return h, Eavg


def d0(tau, x0_r, x0_t, xm_r, xm_t, u_t, u_r):
    """
    Compute d0 equation (53)
    
    Parameters
    ----------
    tau : float
        time of flight, in second
    x0_r: numpy array
        reference 3D coordinate of receiver, (3,)
    x0_t: numpy array
        reference 3D coordinate of transmitter, (3,)
    xm_r: numpy array
        m-th displaced 3D coordinate of receiver, (3,)
    xm_t: numpy array
        m-th displaced 3D coordinate of transmitter, (3,)
    u_t: numpy array
        transmitter unit vector, (3,)
    u_r: numpy array
        receiver unit vector, (3,)    
    
    Returns
    -------
    d0: float 
    
    """
    term1 = -(PhyConst.light_speed_remcom * tau)**2
    term2 = np.sum((x0_r - xm_r + PhyConst.light_speed_remcom*tau*u_r)**2)
    term3 = np.sum((x0_t - xm_t + PhyConst.light_speed_remcom*tau*u_t)**2)
    
    return term1+term2+term3


def am(theta, phi, x0, xm):
    """
    Compute am_r/ am_t, equation (48)
    
    Parameters
    ----------
    theta : float
        elevation, radian
    phi: float
        azimuth, radian
    x0: numpy array
        reference 3D coordinate, (3,)
    xm: numpy array
        m-th displaced 3D coordinate, (3,) 
    
    Returns
    -------
    am: numpy array
        (3,)
    
    """
    return Ry(theta).dot(Rz(-phi)).dot(x0-xm)


def Am(am_r, am_t, s):
    """
    Compute Coefficient Am (57a)
    
    Parameters
    ----------
    am_r: numpy array
        equation (48a), (3,)
    am_t: numpy array
        equation (48b), (3,)
    s: +-1
        equation (12)
    
    Returns
    -------
    Am: float
    
    """
    return 2*(am_r[1]*am_t[1] + s*am_r[2]*am_t[2])


def Bm(am_r, am_t, s):
    """
    Compute Coefficient Bm (57b)
    
    Parameters
    ----------
    am_r: numpy array
        equation (48a), (3,)
    am_t: numpy array
        equation (48b), (3,)
    s: +-1
        equation (12)
    
    Returns
    -------
    Bm: float
    
    """
    return 2*(s*am_r[2]*am_t[1] - am_r[1]*am_t[2])


def Cm(am_r, am_t, d_sq, d0):
    """
    Compute Coefficient Cm (57c)
    
    Parameters
    ----------
    am_r: numpy array
        equation (48a), (3,)
    am_t: numpy array
        equation (48b), (3,)
    d_sq: float
        the squared distance of the matched path in the m-th displaced location
    d0: float
        equation (53)
    
    Returns
    -------
    Cm: float
    
    """
    return d_sq - d0 - 2*am_r[0]*am_t[0]


def u(az, el):
    """
    unit vector of direction, equation (7)
    
    Parameters
    ----------
    az: float
        azimuth, radian
    el: float
        elevation, radian
    
    Returns
    -------
    u: numpy array
        (3,)
    
    """
    return np.array([np.cos(az)*np.cos(el), 
                  np.sin(az)*np.cos(el),
                  np.sin(el)]) 


def est_dist_by_RM(ref_dly, aoa_el, aoa_az, ref_rx_coord, disp_rx_coord, 
                   aod_el, aod_az, ref_tx_coord, disp_tx_coord, s, gamma):
    """
    estimate the distance of a path by reflection model, equation (13)
    
    Parameters
    ----------
    ref_dly: float
        reference path delay, in second
    aoa_el: float
        angle of arrival elevation, radian
    aoa_az: float
        angle of arrival azimuth, radian
    ref_rx_coord: numpy array
        reference receiver coordinate, (3,)
    disp_rx_coord: numpy array
        displaced reciever coordinate. (3,)
    aod_el: float
        angle of departure elevation, radian
    aod_az: float
        angle of departure azimuth, radian
    ref_tx_coord: numpy array
        reference transmitter coordinate, (3,)
    disp_tx_coord: numpy array
        displaced transmitter coordinate. (3,)
    s: int
        +-1
    gamma: float
        angle gamma, radian
    
    Returns
    -------
    est_dist: float
        the estimated path distance
    
    """ 
    ref_path_distance = ref_dly * PhyConst.light_speed_remcom
    item_1 = np.array([ref_path_distance, 0, 0])
    item_2 = Ry(aoa_el).dot(Rz(-aoa_az)).dot(ref_rx_coord - disp_rx_coord)
    item_3 = Qz(s).dot(Rx(gamma)).dot(
        Ry(aod_el)).dot(Rz(-aod_az)).dot(ref_tx_coord - disp_tx_coord)
    eq_right_vector = item_1 + item_2 + item_3
    
    est_dist = np.sqrt(np.sum(eq_right_vector**2)) 
    return est_dist


def est_dist_by_PWA(ref_dly, u_r, u_t, ref_rx_coord, 
                    disp_rx_coord, ref_tx_coord, disp_tx_coord):
    """
    estimate the distance of a path by PWA model, equation (5)
    
    Parameters
    ----------
    ref_dly: float
        reference path delay, in second
    u_r: numpy array
        unit vector of receiver, equation(7a), (3,)
    u_t: numpy array
        unit vector of transmitter, equation(7b), (3,)    
    ref_rx_coord: numpy array
        reference receiver coordinate, (3,)
    disp_rx_coord: numpy array
        displaced reciever coordinate. (3,)
    ref_tx_coord: numpy array
        reference transmitter coordinate, (3,)
    disp_tx_coord: numpy array
        displaced transmitter coordinate. (3,)
    
    Returns
    -------
    est_dist: float
        the estimated path distance
    
    """ 
    est_dist = PhyConst.light_speed_remcom * ref_dly +\
                    u_r.dot(ref_rx_coord- disp_rx_coord) +\
                    u_t.dot(ref_tx_coord - disp_tx_coord) 
    return est_dist


def compute_arbitrary_phase(center_freq, ref_dly, phase):
    """
    estimate the distance of a path by PWA model, equation (5)
    
    Parameters
    ----------
    center_freq: float
        center frequency in Hz
    ref_dly: float
        reference path delay, in second
    phase: phase of reference path
    
    Returns
    -------
    arbitrary_phase: float
        radian
    
    """ 
    phase_ = 2*np.pi -(2*np.pi*center_freq*ref_dly)%(2*np.pi)
    arbitrary_phase = np.deg2rad(phase) - phase_
    return arbitrary_phase


def compute_capacity(ExN0, H, c):
    """
    Parameters
    ----------
    Ex : float
        Total power, linear
    H : numpy array
        channel matrix
    N0 : float
        Noise power, linear
    c: float
        Waterfilling bar

    Returns
    -------
    mimo capacity: float
    number of streams: int

    """
    # H = np.sqrt(4*4)*H / np.linalg.norm(H,2)
    
    # Sigular vector dicomposition
    # u, s, v = np.linalg.svd(H.dot(H))
    u, s, v = np.linalg.svd(H)
    # SNR
    gamma = s**2 * ExN0
    
    ans = 0
    s_op = 0
    for i in range(s.shape[0]):
        power = 1/(i+1)
        # Capacity = np.sum(np.log2(1+gamma[0:i+1]*power))
        Capacity = np.sum( np.minimum(4.8, 0.6*np.log2(1 +gamma[0:i+1]*power)))
        if Capacity > ans:
            ans = Capacity
            s_op = i+1
    
    # Power allocation: waterfilling
    # power = np.maximum([0]*s.shape[0], c-(1/gamma)) 
    # power = 0.25
    
    # print(f'[s] {s}')
    # print(f'[s**2 (dB)] {10*np.log10(s**2)}')
    # print(f'[gamma] {gamma}')
    # print(f'[power allocation] {1/s_op}')
    
    return ans, s_op



