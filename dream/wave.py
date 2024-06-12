import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, sproot

###################################################Wavefront Propogation######################################################
def set_x(Dx_m = 1e-3, dx_m = 1e-6):
    """
    set the x-axis for similuation

    : param Dx_m : total length (default value: 1 mm)
    : param dx_m : unit lenght (default value: 1 um)

    :return: number of units and the defined x-axis
    """
    Nx = int(Dx_m/dx_m)
    # range of x-axis: -0.5mm to 0.5 mm
    x_m = np.linspace(-0.5, 0.5, num = Nx) * Dx_m

    return Dx_m, dx_m, Nx, x_m


#generate a gaussian source
def generate_gaussian_source(x_px, mean_px, fwhm_px):
    '''
    generate a 1D gaussian
 
    :param x_px: x-axis for simulation
    :param mean_px: mean value of the gaussian signal
    :param fwhm_px: fwhm of the gaussian signal
    :return: gaussian source
    '''
    sigma_x = fwhm_px/(2*np.sqrt(2*np.log(2)))
    return np.exp(-((x_px-mean_px)/(np.sqrt(2)*sigma_x))**2)


# generate a source point
def generate_point_source(Nx):
    '''
    generate a point source with center intensity = 1
 
    :param Nx: number of units
    :return: point source
    '''
    source = np.zeros(Nx)
    source[int(Nx/2)] = 1
    return source

# generate a sin source
def generate_sin_source(amplitude,frequency,phase,horizontal_shift,vertical_shift, num_samples,sampling_rate):
    '''
    generate a sin source
 
    :param amplitude
    :param frequency
    :param phase
    :param vertical_shift
    :param num_samples
    :param sampling_rate
    :return: point source
    '''
    time = np.linspace(0+horizontal_shift,num_samples/sampling_rate+horizontal_shift,num_samples)
    signal = amplitude*np.sin(2*np.pi*frequency*time+phase)+vertical_shift
    return time, signal

def get_lens(f_m, wavelength_m,x_m,Nx,aber=0):
    #define lens
    lens = np.exp(-1j*2*np.pi/(wavelength_m*2*f_m)*(x_m**2))
    #define the aberrations of the lens
    aberrations = np.exp(1j*2*np.pi*np.random.randn(Nx)*aber) # change aberrations from 0 to 1 
    return lens,aberrations


#definbe the propagation functions
def ft(t):
    return np.fft.fftshift( np.fft.fft(np.fft.ifftshift(t)))

def ift(t):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(t)))

def fs(t):
    N = len(t)
    df_cpm = 1/(t[-1]-t[0])
    f_cpm = np.fft.fftshift(np.array([df_cpm*n if n<N/2 else df_cpm*(n-N) for n in range(N)]))
    return f_cpm

def propTF(E_in,L_m,lambda_m,z_m):
    #get input field array size
    Nx = len(E_in)
    dx = L_m/Nx; #sample interval

    #(dx<lambda.*z/L)

    fx = fs(np.arange(Nx)*dx);

    H=np.exp(-1j*np.pi*lambda_m*z_m*(fx**2))

    E_out = ft(ft(E_in)*H);
    
    return E_out

def get_prop_list(source, Dx_m, z_start_m=0.1, z_end_m=1.0, num_plots=25, wavelength_m= 1e-9, norm= False):
    #E1_list: saving propagation results for different distances (z_start_m, z_end_m)
    E1_list=[]
    # propagation distance list
    z_m_array = np.linspace(z_start_m,z_end_m,num_plots)
    #propagation
    for z_m_i in z_m_array:
        E1_i = propTF(source, Dx_m, wavelength_m, z_m_i) 
        if norm ==True:
            E1_i = E1_i/np.max(abs(E1_i))
        E1_list.append(E1_i)

    return E1_list


def cal_fwhm(source_gaussian, x_m, method="equation"):
    if method == "equation":
        #by equation
        fwhm = np.sqrt(np.sum(source_gaussian*x_m**2)/np.sum(source_gaussian))*(2*np.sqrt(2*np.log(2)))
    elif method =="twoindex":
        #find the left and right index, then we have (FWHM = right index - left index)
        half_max = max(source_gaussian) / 2.
        #find when function crosses line half_max (when sign of diff flips)
        #take the 'derivative' of signum(half_max - Y[])
        d = np.sign(half_max - np.array(source_gaussian[0:-1])) - np.sign(half_max - np.array(source_gaussian[1:]))
        #find the left and right most indexes
        left_idx = np.where(d > 0)[0]
        right_idx = np.where(d < 0)[-1]
        fwhm = x_m[right_idx] - x_m[left_idx] #return the difference (full width)
    elif method =="rough":
        #For monotonic functions with many data points and if there's no need for perfect accuracy:
        deltax = x_m[1] - x_m[0]
        half_max = max(source_gaussian) / 2.
        l = np.where(source_gaussian > half_max, 1, 0)
        fwhm = np.sum(l) * deltax
    elif method =="inter":
        #interolation
        spline = UnivariateSpline(x_m*1e3, source_gaussian-np.max(source_gaussian)/2, s=0)
        r1, r2 = spline.roots() # find the roots
        fwhm = r2-r1
    elif method =="peaks":
        """
        Determine full-with-half-maximum of a peaked set of points, x and y.

        Assumes that there is only one peak present in the datasset.  The function
        uses a spline interpolation of order k.
        """
        class MultiplePeaks(Exception): pass
        class NoPeaksFound(Exception): pass
        half_max = np.max(source_gaussian)/2.0
        s = splrep(x_m, source_gaussian - half_max, k=3)
        roots = sproot(s)

        if len(roots) > 2:
            raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                    "thus the FWHM can't be determined.")
        elif len(roots) < 2:
            raise NoPeaksFound("No proper peaks were found in the data set; likely "
                    "the dataset is flat (e.g. all zeros).")
        else:
            fwhm = abs(roots[1] - roots[0])
    return fwhm

def correct_beam(E,q_m,z_m,wavelength_m, x_m):
    dt = np.exp(1j*2*np.pi/(wavelength_m*2*(q_m-z_m))*(x_m**2))
    #dt = np.exp(1j*2*np.pi/(wavelength_m*2*(q_m-0.01))*(x_m**2))
    E_corr = E * np.exp(-1j*np.angle(E*dt))
    return E_corr


###################################################Wavefront sensing######################################################

def get_grating(x_m,pitch_m):
    grating=(np.sign(np.sin(2*np.pi*x_m/pitch_m))+1)/2
    return grating

#Antoine's code to get phase
def get_phase_AW(Ia, x_m, pitch_m):
  f_pm = fs(x_m)
  #select the last sample in E_list, then use fft

  Sa = ft(Ia)

  #create a mask
  f0_pm = 1/pitch_m
  mask= np.abs(f_pm- f0_pm)<0.6*f0_pm

  #select the signal around 10 mm by masking
  Sa_mask = Sa*mask

  Ia_mask = ift(Sa_mask)

  phi_a_rad = np.unwrap(np.angle(Ia_mask))

  return phi_a_rad


## Ken's code to get phase
def get_phase_Ken(a, wide=5., dc=5):
    # first, search for the first order peak
    f = np.fft.fft(a)
    fabs = np.abs(f)
    imax = len(fabs)//2 - 2  # look only at the left half because of the FFT. Integer division

    dc0 = np.short(dc)  # this is the minimum frequency where we can have a peak
    w = dc0 + np.squeeze(np.where(fabs[dc0:imax] == np.max(fabs[dc0:imax]))) # locate the 1st order peak

    x = np.arange(0,len(f))  # create a linear array to support the filtering
    filt = np.exp( -0.5 * ((x-w)/wide)**2 ) # filter the Fourier domain
    c = np.fft.ifft(filt * f)
    phi1 = np.arctan2(np.imag(c), np.real(c)) # extract the wrapped phase and the unwrap
    phi2 = np.unwrap(phi1)

    return phi2

  # Get the amplitude from the data
  # we may have to zero-pad this and then remove padding in the end
def ExtractAmplitude(a, wide=7.):
    f1 = np.fft.fftshift(np.fft.fft(a))  # centered fft
    N = len(f1)
    wmiddle = N//2
    x = np.arange(0,N)
    filt = np.exp( -0.5 * ((x-wmiddle)/wide)**2 )  # Gaussian filter
    amp = np.abs(np.fft.ifft(np.fft.ifftshift(filt*f1)))
    return amp

# unwrapping from the center
def unwrapc(s):
  Nx = len(s)
  right = np.unwrap(s[int(Nx/2):])
  left  = np.flip(np.unwrap(s[(int(Nx/2)-1):-1:]))
  return np.concatenate((left,right))

# cumulative sum from the center
def cumsumc(s):
    Nx = len(s)
    right = np.cumsum(s[int(Nx/2):])
    left  = np.flip(np.cumsum(s[(int(Nx/2)-1):-1:]))
    return np.concatenate((left,right))

def cumsum(phi_rad,Nx,dx_m,pitch_m,wavelength_m):
    wavefront_rad = np.cumsum(phi_rad-phi_rad[int(Nx/2)])*(dx_m)*(pitch_m)/wavelength_m/2
    return wavefront_rad
