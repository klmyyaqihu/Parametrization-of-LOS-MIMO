"""
spherical.py:  Methods for spherical coordinates
"""

import numpy as np


def cart_to_sph(d):
    """
    Cartesian to spherical coordinates.

    Parameters
    ----------
    d : (n,3) array
        vector of positions

    Returns
    -------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees
    """

    # Compute radius
    r = np.sqrt(np.sum(d ** 2, axis=1))
    r = np.maximum(r, 1e-8)

    # Compute angle of departure
    phi = np.arctan2(d[:, 1], d[:, 0]) * 180 / np.pi
    theta = np.arccos(d[:, 2] / r) * 180 / np.pi

    return r, phi, theta

def cart_to_sph_scalar(d):
    """
    Cartesian to spherical coordinates.

    Parameters
    ----------
    d : (n,3) array
        vector of positions

    Returns
    -------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees
    """

    # Compute radius
    r = np.sqrt(np.sum(d ** 2))
    r = np.maximum(r, 1e-8)

    # Compute angle of departure
    phi = np.arctan2(d[1], d[0]) * 180 / np.pi
    theta = np.arccos(d[2] / r) * 180 / np.pi

    return r, phi, theta

def sph_to_cart(r, phi, theta):
    """
    Spherical coordinates to cartesian coordinates

    Parameters
    ----------
    r:  (n,) array
        radius of each point
    phi, theta:  (n,) arrays
        azimuth and inclination angles in degrees

    Returns
    -------
    d : (n,3) array
        vector of positions

    """

    # Convert to radians
    phi = phi * np.pi / 180
    theta = theta * np.pi / 180

    # Convert to cartesian
    d0 = r * np.cos(phi) * np.sin(theta)
    d1 = r * np.sin(phi) * np.sin(theta)
    d2 = r * np.cos(theta)
    d = np.stack((d0, d1, d2), axis=-1)

    return d



def spherical_add_sub(phi0, theta0, phi1, theta1, sub=True):
    """
    Angular addition and subtraction in spherical coordinates

    For addition, we start with a vector at (phi0,theta0), then rotate by
    theta1 in the (x1,x3) plance and then by phi1 in the (x1,x2) plane.
    For subtraction, we start with a vector at (phi0,theta0), then rotate by
    -phi1 in the (x1,x2) plane and then by -theta1 in the (x1,x3) plane.


    Parameters
    ----------
    phi0, theta0 : arrays of same size
        (azimuth,inclination) angle of the initial vector in degrees
    phi1, theta1 : arrays of same size
        (azimuth,inclination) angle of the rotation
    sub:  boolean
        if true, the angles are subtracted.  otherwise, they are added

    Returns
    -------
    phi2, theta2 : arrays of same size as input
        (azimuth,inclination) angle of the rotated vector

    """

    # Convert to radians
    theta0 = np.pi / 180 * theta0
    theta1 = np.pi / 180 * theta1
    phi0 = np.pi / 180 * phi0
    phi1 = np.pi / 180 * phi1

    if sub:
        # Find unit vector in direction of (theta0,phi0)
        x1 = np.sin(theta0) * np.cos(phi0)
        x2 = np.sin(theta0) * np.sin(phi0)
        x3 = np.cos(theta0)

        # Rotate by -phi1.
        y1 = x1 * np.cos(phi1) + x2 * np.sin(phi1)
        y2 = -x1 * np.sin(phi1) + x2 * np.cos(phi1)
        y3 = x3

        # Rotate by -theta1
        z1 = y1 * np.cos(theta1) - y3 * np.sin(theta1)
        z3 = y1 * np.sin(theta1) + y3 * np.cos(theta1)
        z2 = y2
        z1 = np.minimum(1, np.maximum(-1, z1))
 
        # Compute the angle of the transformed vector
        # we use the (z3,z2,z1) coordinate system
        phi2 = np.arctan2(z2, z3) * 180 / np.pi
        theta2 = np.arcsin(z1) * 180 / np.pi
    else:

        # Find unit vector in direction of (theta0,phi0)
        x3 = np.cos(theta0) * np.cos(phi0)
        x2 = np.cos(theta0) * np.sin(phi0)
        x1 = np.sin(theta0)

        # Rotate by theta1
        y1 = x1 * np.cos(theta1) + x3 * np.sin(theta1)
        y3 = -x1 * np.sin(theta1) + x3 * np.cos(theta1)
        y2 = x2

        # Rotate by phi1.
        z1 = y1 * np.cos(phi1) - y2 * np.sin(phi1)
        z2 = y1 * np.sin(phi1) + y2 * np.cos(phi1)
        z3 = y3
        z3 = np.minimum(1, np.maximum(-1, z3))

        # Compute angles
        phi2 = np.arctan2(z2, z1) * 180 / np.pi
        theta2 = np.arccos(z3) * 180 / np.pi

    return phi2, theta2


def rotate_by_los_x_arr( phi_los, theta_los, phi_arr, theta_arr):
    
        
    n = np.shape(phi_arr)[0]
    
    res_phi_arr = []
    res_theta_arr = []
    phi_los = np.pi/180*phi_los
    theta_los = np.pi/180*theta_los
    
    for i in range(n):
        # elevation
        theta = theta_arr[i]
        phi = phi_arr[i]
        
        
        # if i < 3:
        #     print(f'ref_0={phi_los}, ref_1={theta_los}')
        #     print(f'az = [{phi}], el = [{theta}]')
        
        # Convert to radians 
        theta = np.pi/180*theta
        phi = np.pi/180*phi


        x3 = np.array([np.cos(theta)*np.cos(phi), 
                       np.cos(theta)*np.sin(phi),
                       np.sin(theta)])

        phi_matrix = np.array([[np.cos(-phi_los), -np.sin(-phi_los), 0],
                             [np.sin(-phi_los), np.cos(-phi_los), 0],
                             [0, 0, 1]])
        x3 = np.dot(phi_matrix,x3)

    #     print(f'{x3[0]}, {x3[1]}, {x3[2]}')
        r = np.sqrt(np.sum(x3**2))
        phi = np.arctan2(x3[1], x3[0])*180/np.pi
        theta = np.arctan2(x3[2], np.sqrt(np.sum(x3[0]**2 + x3[1]**2)))*180/np.pi

    #     print(f'{phi}, {theta}')

        theta_matrix = np.array([[np.cos(-theta_los), 0, -np.sin(-theta_los)],
                             [0, 1, 0],
                             [np.sin(-theta_los), 0, np.cos(-theta_los)]])
        x3 = np.dot(theta_matrix, x3)

        r = np.sqrt(np.sum(x3**2))
        phi = np.arctan2(x3[1], x3[0])*180/np.pi
        theta = np.arctan2(x3[2], np.sqrt(np.sum(x3[0]**2 + x3[1]**2)))*180/np.pi
    #     print(f'{x3[0]}, {x3[1]}, {x3[2]}')
    
        # if i < 3:
        #     print(f'REL ref_az = [{phi}], ref_el = [{theta}]')
    
        res_phi_arr.append(phi)
        res_theta_arr.append(theta)
    return np.array(res_phi_arr), np.array(res_theta_arr)