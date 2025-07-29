import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from numpy.fft import fftshift
from tqdm import tqdm # for progress bars
from statistics import mean, stdev
import torch
import frft
# import frft_gpu as frft_g
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import time
from numpy.linalg import svd
from numpy.polynomial.polynomial import Polynomial
from skimage.restoration import unwrap_phase
# from kamui import unwrap_dimensional
from scipy.fftpack import dct, idct,dctn, idctn

def Propagator(M, N, lambda_, area1, area2, z):
    p = np.zeros((M, N), dtype=np.complex128)
    for ii in range(M):
        for jj in range(N):
            alpha = lambda_ * (ii - M / 2 - 1) / area1
            beta = lambda_ * (jj - N / 2 - 1) / area2
            if (alpha ** 2 + beta ** 2) <= 1:
                p[ii, jj] = np.exp(-2 * np.pi * 1j * z * np.sqrt(1 - alpha ** 2 - beta ** 2) / lambda_)
    return p


def funcAutoFocusAMS(I):
    M, N = I.shape
    recon_sharpness = np.sum(np.abs(I))
    return recon_sharpness

def funcAutoFocusDFS(I):
    M, N = I.shape
    FX, FY = np.gradient(np.abs(I))
    temp = FX**2 + FY**2
    recon_sharpness = np.var(temp)
    return recon_sharpness

def funcAutoFocusGRA(I):
    M, N = I.shape
    FX, FY = np.gradient(np.abs(I))
    temp = FX**2 + FY**2
    recon_sharpness = np.mean(temp)
    return recon_sharpness

def PCA_aberration_comp(spectrum3_upd1):
    IFFT_ROI = IFT2Dc( spectrum3_upd1 )
    # Get the exponential term
    ConjPhase = np.exp(1j * np.angle(IFFT_ROI))

    # Singular Value Decomposition
    U, S, Vh = svd(ConjPhase)
    V = Vh.T

    # Take the first 'num' principal components
    M, N = spectrum3_upd1.shape
    num = 1
    SS = np.zeros((M, N), dtype=complex)
    for i in range(num):
        SS[i, i] = S[i]

    # Least-squares fitting for U
    Unwrap_U = np.unwrap(np.angle(U[:, :2]), axis=0)
    SF_U1 = Polynomial.fit(range(1, M + 1), Unwrap_U[:, 0], 2)
    SF_U2 = Polynomial.fit(range(1, M + 1), Unwrap_U[:, 1], 2)

    EstimatedSF_U1 = SF_U1(range(1, M + 1))
    EstimatedSF_U2 = SF_U2(range(1, M + 1))

    New_U1 = np.exp(1j * EstimatedSF_U1)
    New_U2 = np.exp(1j * EstimatedSF_U2)
    U[:, :2] = np.column_stack([New_U1, New_U2])

    # Least-squares fitting for V
    Unwrap_V = np.unwrap(np.angle(V[:, :2]), axis=0)
    SF_V1 = Polynomial.fit(range(1, N + 1), Unwrap_V[:, 0], 2)
    SF_V2 = Polynomial.fit(range(1, N + 1), Unwrap_V[:, 1], 2)

    EstimatedSF_V1 = SF_V1(range(1, N + 1))
    EstimatedSF_V2 = SF_V2(range(1, N + 1))

    New_V1 = np.exp(1j * EstimatedSF_V1)
    New_V2 = np.exp(1j * EstimatedSF_V2)
    V[:, :2] = np.column_stack([New_V1, New_V2])

    # Get the aberration term
    Z = U @ SS @ V.T
    # plt.figure()
    # plt.imshow(np.angle(Z), cmap='gray')
    # plt.title('Angle of Z')
    # plt.colorbar()

    # FFT and replace the corresponding original region of the spectrum
    Psi_pca = fftshift(fft2(fftshift(IFFT_ROI * np.conj(Z))))

    # plt.figure()
    # plt.imshow(np.absolute(Psi_pca), cmap='gray')
    # plt.title('Absolute Value of Psi_pca')
    # plt.colorbar()

    return Psi_pca

def zernike_func_cartesian_coordinate(A1):
    M, N = A1.shape
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, M)
    X, Y = np.meshgrid(x, y)

    F = np.zeros((M, N, 11))
    F[:, :, 0] = 1
    F[:, :, 1] = 2 * X
    F[:, :, 2] = 2 * Y
    F[:, :, 3] = np.sqrt(3) * (2 * X**2 + 2 * Y**2 - 1)
    F[:, :, 4] = np.sqrt(6) * (2 * X * Y)
    F[:, :, 5] = np.sqrt(6) * (X**2 - Y**2)
    F[:, :, 6] = np.sqrt(8) * (3 * X**2 * Y + 3 * Y**3 - 2 * Y)
    F[:, :, 7] = np.sqrt(8) * (3 * X**3 + 3 * X * Y**2 - 2 * X)
    F[:, :, 8] = np.sqrt(8) * (3 * X**2 * Y - Y**3)
    F[:, :, 9] = np.sqrt(8) * (X**3 - 3 * X * Y**2)
    F[:, :, 10] = np.sqrt(10) * (4 * X**3 * Y + 4 * X * Y**3)
    return F

def zernike_coeffs_cartesian(PW2, Z, mask, ncoeff):
    """
    Compute Zernike coefficients for phase aberration.
    ncoeff = number of coefficient (11-ncoeff) out of 11 to be used.
    """
    P1 = Z.shape[2]
    P1 = P1 - ncoeff
    M, N = PW2.shape

    phi = PW2.flatten()
    mask1 = mask.flatten()

    Z1 = np.zeros((M * N, P1))
    Z11 = []

    for i in range(P1):
        temp = Z[:, :, i].flatten()
        Z1[:, i] = temp
        Z11.append(temp[mask1 == 1])

    Z11 = np.array(Z11).T
    phi1 = phi[mask1 == 1]

    a1 = np.linalg.pinv(Z11) @ phi1
    return a1, Z1



def wrapToPi(x):
    """
    Wrap angle to [-π, π] interval while preserving complex type
    """
    xwrap = np.remainder(x, 2*np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2*np.pi * np.sign(xwrap[mask])
    return xwrap + 0j  # Ensure complex output

def dct2(x):
    """
    2D DCT transform
    """
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2(x):
    """
    2D inverse DCT transform
    """
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def solvePoisson(rho):
    """
    Solve the Poisson equation using DCT
    """
    dctRho = dct2(rho)
    N, M = rho.shape
    
    J, I = np.meshgrid(np.arange(M), np.arange(N))
    denom = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
    dctPhi = dctRho / denom
    dctPhi[0, 0] = 0
    
    return idct2(dctPhi)

def unwrap_TIE(phase_wrap):
    """
    Unwrap phase using Transport of Intensity Equation (TIE)
    """
    psi = np.exp(1j * phase_wrap)
    
    # Calculate derivatives while preserving complex values
    psi_diff_x = np.diff(psi, axis=1)
    # Preserve complex values by using psi_diff_x directly instead of its angle
    edx = np.column_stack([
        np.zeros((psi.shape[0], 1), dtype=complex),
        psi_diff_x,
        np.zeros((psi.shape[0], 1), dtype=complex)
    ])
    
    psi_diff_y = np.diff(psi, axis=0)
    edy = np.vstack([
        np.zeros((1, psi.shape[1]), dtype=complex),
        psi_diff_y,
        np.zeros((1, psi.shape[1]), dtype=complex)
    ])
    
    # Calculate Laplacian using finite difference
    lap = np.diff(edx, axis=1) + np.diff(edy, axis=0)
    
    # Calculate right hand side
    rho = np.imag(np.conj(psi) * lap)
    
    return solvePoisson(rho)

# Define 2D Fourier Transform functions
def FT2Dc(image):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

def IFT2Dc(spectrum):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(spectrum)))