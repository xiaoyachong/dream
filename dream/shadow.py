import Shadow
import numpy as np
import numpy

# Function designed for BL531. Calculate the corresponding beam size (FWHM, unit: um) with given x_rot (pitch) and y_rot (roll) values.
def toroid(X_ROT,Y_ROT):
    beam = Shadow.Beam()
    oe0 = Shadow.Source()
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()

    oe1.X_ROT = X_ROT
    oe1.Y_ROT = Y_ROT

    

    oe0.FDISTR = 3
    oe0.F_PHOT = 0
    oe0.HDIV1 = 0.0
    oe0.HDIV2 = 0.0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.ISTAR1 = 5676561
    oe0.PH1 = 1000.0
    oe0.SIGDIX = 5e-05
    oe0.SIGDIZ = 5e-05
    oe0.SIGMAX = 1e-05
    oe0.SIGMAZ = 1e-05
    oe0.VDIV1 = 0.0
    oe0.VDIV2 = 0.0
    oe1.DUMMY = 100.0
    oe1.FMIRR = 3
    oe1.FWRITE = 1
    oe1.F_EXT = 1
    oe1.F_MOVE = 1
    oe1.R_MAJ = 305.3065
    oe1.R_MIN = 0.655
    oe1.T_IMAGE = 0.0

    oe2.ALPHA = 90.0
    oe2.DUMMY = 100.0
    oe2.FCYL = 1
    oe2.FMIRR = 2
    oe2.FWRITE = 1
    oe2.F_DEFAULT = 0
    oe2.SIMAG = 5.0
    oe2.SSOUR = 1000000.0
    oe2.THETA = 88.0
    oe2.T_IMAGE = 5.0
    oe2.T_SOURCE = 5.0

    beam.genSource(oe0)

    beam.traceOE(oe1,1)

    beam.traceOE(oe2,2)

    x_fhwm_m = beam.histo1(1)['fwhm']
    y_fwhm_m = beam.histo1(3)['fwhm']

    a=x_fhwm_m*1e6
    b=y_fwhm_m*1e6
    size_um=np.sqrt(a**2+b**2)
   
    return size_um

# Function designed for oasys (BL631). Calculate the corresponding histogram, bins, center, size, intensity lists for a given mirror file (.mat)
def get_histo(filepath):
    beam = Shadow.Beam()
    oe0 = Shadow.Source()
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()
    oe3 = Shadow.OE()
    oe4 = Shadow.OE()
    oe5 = Shadow.OE()

    #
    # Define variables. See meaning of variables in: 
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml 
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #

    oe0.FDISTR = 3
    oe0.F_COLOR = 2
    oe0.F_PHOT = 0
    oe0.F_POLAR = 0
    oe0.HDIV1 = 0.0
    oe0.HDIV2 = 0.0
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.ISTAR1 = 0
    oe0.NPOINT = 1000000
    #oe0.N_COLOR = 3
    oe0.N_COLOR = 1
    oe0.PH1 = 860.0
    #oe0.PH2 = 860.43
    #oe0.PH3 = 859.57
    oe0.SIGDIX = 0.0024
    oe0.SIGDIZ = 0.0003
    oe0.SIGMAX = 4e-05
    oe0.SIGMAZ = 7e-06
    oe0.VDIV1 = 0.0
    oe0.VDIV2 = 0.0

    oe1.ALPHA = 90.0
    oe1.DUMMY = 100.0
    oe1.FMIRR = 1
    oe1.FWRITE = 1
    oe1.F_DEFAULT = 0
    oe1.SIMAG = 15.25
    oe1.SSOUR = 10.95
    oe1.THETA = 88.5
    oe1.T_IMAGE = 0.0
    oe1.T_INCIDENCE = 88.5
    oe1.T_REFLECTION = 88.5
    oe1.T_SOURCE = 10.95

    #oe2.CX_SLIT = numpy.array([p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe2.CX_SLIT = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe2.DUMMY = 100.0
    oe2.FWRITE = 3
    oe2.F_REFRAC = 2
    oe2.F_SCREEN = 1
    oe2.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    oe2.N_SCREEN = 1
    oe2.RX_SLIT = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe2.RZ_SLIT = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe2.T_IMAGE = 0.0
    oe2.T_INCIDENCE = 0.0
    oe2.T_REFLECTION = 180.0
    oe2.T_SOURCE = 5.0

    oe3.ALPHA = 90.0
    oe3.DUMMY = 100.0
    oe3.FHIT_C = 1

    oe3.FILE_RIP = filepath

    oe3.FMIRR = 1
    oe3.FWRITE = 1
    oe3.F_DEFAULT = 0
    oe3.F_G_S = 2
    oe3.F_RIPPLE = 1
    oe3.RLEN1 = 0.19127213815886912
    oe3.RLEN2 = 0.19127213815886912
    oe3.RWIDX1 = 0.05
    oe3.RWIDX2 = 0.05
    oe3.SIMAG = 3.2
    oe3.SSOUR = 16.0
    oe3.THETA = 88.75
    oe3.T_IMAGE = 0.0
    oe3.T_INCIDENCE = 88.75
    oe3.T_REFLECTION = 88.75
    oe3.T_SOURCE = 0.05

    oe4.ALPHA = 180.0
    oe4.DUMMY = 100.0
    oe4.FWRITE = 1
    oe4.F_CENTRAL = 1
    oe4.F_GRATING = 1
    oe4.F_RULING = 5
    oe4.F_RUL_ABS = 1
    oe4.PHOT_CENT = 860.0
    oe4.RULING = 1200000.0
    oe4.RUL_A1 = 1990000.0
    oe4.RUL_A2 = 2520000.0
    oe4.RUL_A3 = 28800000.0
    oe4.R_LAMBDA = 5000.0
    oe4.T_IMAGE = 0.0
    oe4.T_REFLECTION = 87.9998043372
    oe4.T_SOURCE = 2.0

    oe5.DUMMY = 100.0
    oe5.FWRITE = 3
    oe5.F_REFRAC = 2
    oe5.F_SCREEN = 1
    oe5.N_SCREEN = 1
    oe5.T_IMAGE = 0.0
    oe5.T_INCIDENCE = 0.0
    oe5.T_REFLECTION = 180.0
    oe5.T_SOURCE = 1.2

    beam.genSource(oe0)

    beam.traceOE(oe1,1)

    beam.traceOE(oe2,2)

    #define a list for slit
    slit_mask_list=[]
    slit_list=np.linspace(-2e-03,2e-03,41)

    for i in range(len(slit_list)):
        #dealing with orientation flip
        slit_select = Shadow.ShadowTools.getshonecol(beam,1)
        slit_position_m = slit_list[i]
        slit_size_m = 100e-6
        slit_mask = np.abs(slit_select-slit_position_m)<slit_size_m/2.0
        slit_mask_list.append(slit_mask)


    beam.traceOE(oe3,3)

    beam.traceOE(oe4,4)

    beam.traceOE(oe5,5)


    z = Shadow.ShadowTools.getshonecol(beam,3)
    mask = Shadow.ShadowTools.getshonecol(beam,10)    

    z_histo_list=[]
    z_center_m_list=[]
    z_size_m_list=[]
    z_intensity_au_list=[]
    for i in range(len(slit_list)):
        #Used for visualization only
        (z_histo, bin_edges) = np.histogram(z[(mask>=0.9)&slit_mask_list[i]], bins=1001, range=(-100e-6,100e-6))
        z_histo_list.append(z_histo)
        #Used for optimization. Calculate three metrics: center, size and intensity
        z_center_m = np.mean(z[(mask>=0.9)&slit_mask_list[i]])
        z_size_m = np.std(z[(mask>=0.9)&slit_mask_list[i]])
        z_intensity = len(z[(mask>=0.9)&slit_mask_list[i]])
        z_center_m_list.append(z_center_m)
        z_size_m_list.append(z_size_m)
        z_intensity_au_list.append(z_intensity)

    return z_histo_list, bin_edges, z_center_m_list, z_size_m_list, z_intensity_au_list
