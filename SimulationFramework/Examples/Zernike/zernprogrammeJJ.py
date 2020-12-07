# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:26:48 2020

@author: fdz57121
"""


import numpy as np
import matplotlib.pyplot as plt

def zernike_nm(n, m, N):
#This multiplies the Rmn part of the function with the cos/sin part
    """
     Creates the Zernike polynomial with radial index, n, and azimuthal index, m.

     Args:
        n (int): The radial order of the zernike mode
        m (int): The azimuthal order of the zernike mode
        N (int): The diameter of the zernike more in pixels
     Returns:
        ndarray: The Zernike mode
     """

    coords = np.linspace(-1,1,N)
#These are all calculated with a grid of -1 to 1 because zernikes are calculated over a unit circle

    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)


    if m==0:
        Z = zernikeRadialFunc(n, 0, R)
    else:
        if m > 0: # j is even
            Z = zernikeRadialFunc(n, m, R) * np.cos(m*theta)
        else:   #i is odd
            m = abs(m)
            Z = zernikeRadialFunc(n, m, R) * np.sin(m * theta)

    #Applying a circular mask
    Z = Z*np.less_equal(R, 1.0)

    return Z


def zernikeRadialFunc(n, m, r):
#Makes the Rmn part
    """
    Fucntion to calculate the Zernike radial function

    Parameters:
        n (int): Zernike radial order
        m (int): Zernike azimuthal order
        r (ndarray): 2-d array of radii from the centre the array

    Returns:
        ndarray: The Zernike radial function
    """

    R = np.zeros(r.shape)
    for i in range(0, int((n - m) / 2) + 1):

        R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                         np.math.factorial(n - i)) /
                         (np.math.factorial(i) *
                          np.math.factorial(0.5 * (n + m) - i) *
                          np.math.factorial(0.5 * (n - m) - i)),
                         dtype='float')
#These functions come from the Rnm(rho) function you see in the defintion section on the zernike wikipedia
    return R

def zerngenerator(j, N):

    """
    Find the [n,m] list giving the radial order n and azimuthal order
    of the Zernike polynomial of Noll index j.

    Parameters:
        j (int): The Noll index for Zernike polynomials

    Returns:
        list: n, m values
    """

    n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
    p = (j-(n*(n+1))/2.)
    k = n%2
    m = int((p+k)/2.)*2 - k

    if m!=0:
        if j%2==0:
            s=1
        else:
            s=-1
        m *= s

    if m == 0:

        normalization = ((n+1)/np.pi)**0.5
    else:
        normalization = ((2)*(n+1)/np.pi)**0.5



    z = zernike_nm(n,m,N)*normalization

    return(z)


def reconstruction(a, N):
#N = size of image
#This function takes a list, a, however long of zernike coefficients (in order according to Noll index)
#And generates an image with that combination of zernikes
#Note Noll starts at 1 so anything in a[0] does nothing
#a[1] is the piston term so adding this will increase the amplitude in the image

    recon = np.zeros([N,N])
    for i in range(len(a)):
        if a[i] != 0:
            im = zerngenerator(i,N)
            recon += im*a[i]#You just add up the components multiplied by their coefficients
    return(recon)


if __name__ == "main":
    data = reconstruction([0,1,0,1], 1000)

    from PIL import Image

    data /= np.max(data) / 255
    image = Image.fromarray(data)
    image = image.convert('L')
    image.save('test.bmp')
