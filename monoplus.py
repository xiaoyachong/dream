import numpy as np


#----------------------------------------------------------------
# Propagators in 1D
#----------------------------------------------------------------
#Fresnel Transfer Function (TF) Propagator in 1D
def propTF(uin_V_m, L_m, lambda_m, z_m):
    M = uin_V_m.size
    dx_m = L_m/M
    k_1_m = 2*np.pi/lambda_m
    
    #Frequency coordinates
    fx_1_m = np.linspace(-1/(2*dx_m), 1/(2*dx_m) - (1/L_m), M)
    
    #Transfer function
    H = np.exp(-1j * np.pi * lambda_m * z_m * (fx_1_m**2))
    H = np.fft.fftshift(H)
    Uin_V_m = np.fft.fft(np.fft.fftshift(uin_V_m))
    Uout_V_m = H * Uin_V_m
    uout_V_m = np.fft.ifftshift(np.fft.ifft(Uout_V_m))
    return uout_V_m

#Huygens-Fresnel Propagator in 1D
def propHF(xo_m, xi_m, Eo, k_1_m, z_m):
    M = np.size(xo_m)
    N = np.size(xi_m)
    Eii = np.zeros(M, dtype = "complex")
    Ei = np.zeros(N, dtype = "complex")
    #Loop for computing Ei for xi
    for j in range(N):
        #Loop for computing Ei at xo
        for i in range(M):
            roi_m = np.sign(z_m) * np.sqrt((xo_m[i] - xi_m[j])**2 + z_m**2)
            Eii[i] = Eo[i] * np.exp(+1j * k_1_m * roi_m) / roi_m
        #Allocate the calculated Ei(xi) = sum Ei_i
        Ei[j] = np.sum(Eii[:])
    return Ei[:]

#----------------------------------------------------------------
# Gaussian generator in 1D
#----------------------------------------------------------------
#Gaussian function in 1D
def gaussfunc(x, mean_x, sigma_x):
    gaussF = np.exp(-((x-mean_x)/(np.sqrt(2)*sigma_x))**2)
    return gaussF

#----------------------------------------------------------------
# Grating and lens
#----------------------------------------------------------------
#VLS grating equation
def VLS(alpha_rad, m, lambda_m, a_m, k_lpm, p_m, q_m, w_m):
    beta_rad = np.arcsin(np.sin(alpha_rad) - m *lambda_m/a_m)
    b2 = -((np.cos(alpha_rad)**2)/p_m + (np.cos(beta_rad)**2)/q_m) / (2 * k_lpm * lambda_m) 
    b3 = -(np.cos(alpha_rad)**2 * np.sin(alpha_rad)/p_m**2 - np.cos(beta_rad)**2 * np.sin(beta_rad)/q_m**2) / (2 * k_lpm * lambda_m) 
    n = k_lpm * (w_m + b2 * w_m**2 + b3* w_m**3)
    return n

#VLS grating equation for reflection
def VLSrefl(alpha_rad, m, lambda_m, a_m, k_lpm, p_m, q_m, w_m):
    beta_rad = np.arcsin(m *lambda_m/a_m - np.sin(alpha_rad))
    b2 = -((np.cos(alpha_rad)**2)/p_m + (np.cos(beta_rad)**2)/q_m) / (2 * k_lpm * lambda_m) 
    b3 = -(np.cos(alpha_rad)**2 * np.sin(alpha_rad)/p_m**2 - np.cos(beta_rad)**2 * np.sin(beta_rad)/q_m**2) / (2 * k_lpm * lambda_m) 
    n = k_lpm * (w_m + b2 * w_m**2 + b3* w_m**3)
    return n


#Focus in 1D
def focus(uin_V_m, L_m, lambda_m, zf_m):
    M = uin_V_m.size
    dx_m = L_m/M
    k_1_m = 2 * np.pi/lambda_m
    
    x_m = np.linspace(-L_m/2, L_m/2 - dx_m, M)
    uout_V_m = uin_V_m * np.exp(-1j * k_1_m/(2*zf_m)*(x_m**2))
    return uout_V_m

#Polynomial function of n degree with random generated number
def polyfunc(x, Dx, degree):
    shape_wave = x * 0
    for i_p in range (degree):
        shape_wave = shape_wave + np.random.rand(1) * (x/Dx) ** (i_p)
    return shape_wave

#Measurement the beam size:
# Z - number of points
# x_m - coordinates
#E - field
#METHOD 1: Second moment of area of Gaussian
def secondmomt(Z, x_m, E):
    sigma_mrms = np.zeros(Z)
    for i_z in range(Z):
        I_z = np.abs(E[:,i_z] **2)/np.max(np.abs(E[:,i_z] **2))
        mu_m = np.sum(x_m * I_z)/np.sum(I_z)
        sigma_mrms[i_z]= np.sqrt(np.sum((x_m - mu_m)**2 * I_z)/np.sum(I_z))
    return sigma_mrms             

#METHOD 2: Full Width Half Maximum
def fwhm(Z, x_m, E):
    sigma2_mrms = np.zeros(Z)
    px_m = (x_m[1] - x_m[0]) 
    for i_z in range(Z):
        Iz = np.abs(E[:,i_z] **2)
        sigma2_mrms[i_z] = (np.size(np.where(Iz >= np.max(Iz/2))))/2.35 * px_m
    return sigma2_mrms

#Divergence
def divg(sigma_rms, z):
    div = np.zeros(np.size(z))
    for i_z in range(np.size(z)):
        div[i_z] = sigma_rms[i_z]/z[i_z]
    return div