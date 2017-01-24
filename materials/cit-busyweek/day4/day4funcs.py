import numpy as np
import scipy.linalg as sl
import scipy.special as ss
import libstempo as lt, libstempo.plot as ltp, libstempo.toasim as ltt
import matplotlib.pyplot as plt
import os, sys, glob
import simple_mcmc as smcmc
import triangle

day = 86400.0                   # Seconds per day
year =  31557600.0              # Seconds per year (yr = 365.25 days, so Julian years)
EulerGamma = 0.5772156649015329 # Euler gamma constant
mjdT0 = 54000.0                 # MJD to which all toas are referenced (for precision)

def designqsd(t, order=2):
    """
    Calculate the design matrix for quadratic spindown

    :param t:
        array of toas
    """
    M = np.ones([len(t), order+1])
    for ii in range(1, order+1):
        M[:,ii] = t ** ii
    
    return M.copy()

def fourierdesignmatrix(t, nmodes, Ttot=None):
    """
    Calculate the matrix of Fourier modes A, given a set of timestamps

    These are sine/cosine basis vectors at evenly separated frequency bins

    Mode 0: sin(f_0)
    Mode 1: cos(f_0)
    Mode 2: sin(f_1)
    ... etc

    :param nmodes:
        The number of modes that will be included (= 2*nfreq)
    :param Ttot:
        Total duration experiment (in case not given by t)

    :return:
        (A, freqs), with A the 'fourier design matrix', and f the associa

    """
    N = t.size
    A = np.zeros([N, nmodes])
    T = t.max() - t.min()

    if(nmodes % 2 != 0):
      print "WARNING: Number of modes should be even!"

    if Ttot is None:
        deltaf = 1.0 / T
    else:
        deltaf = 1.0 / Ttot

    freqs1 = np.linspace(deltaf, (nmodes/2)*deltaf, nmodes/2)
    freqs = np.array([freqs1, freqs1]).T.flatten()

    # The cosine modes
    for i in range(0, nmodes, 2):
        omega = 2.0 * np.pi * freqs[i]
        A[:,i] = np.cos(omega * t)

    # The sine modes
    for i in range(1, nmodes, 2):
        omega = 2.0 * np.pi * freqs[i]
        A[:,i] = np.sin(omega * t)

    return (A, freqs)


# ## Functions for power-law spectral analysis ##
# In order to do spectral analysis, we need the PSD, the covariance function (PSD cosine transform), and the covariance matrix.

# In[5]:

def PL_psd(f, amp, alpha, sig_fL):
    """
    PSD for a power-law signal.

    :param f:
        Array of frequencies for which to return the PSD (1/sec)

    :param amp:
        Unitless amplitude of the power-law signal

    :param alpha:
        Spectral index

    :param sig_fL:
        Signal low-frequency cut-off. A physical bound
    
    Note that this function returns the power per
    frequency bin of (1/Tmax) size, so the units are in sec^2 / (1/Tmax)
    """
    Si = 3.0 - 2.0*alpha

    Tmax = 1.0 / np.min(f)

    freqpy = f * year

    mask = (freqpy >= sig_fL)

    psd = np.zeros(len(freqpy))
    psd[mask] = (amp**2 * year**3 / (12*np.pi*np.pi * Tmax)) * freqpy[mask] ** (-Si)
    
    return psd


# Calculate the covariance matrix for a red signal
# (for a GWB with unitless amplitude h_c(1yr^{-1}) = 1)
def PL_covfunc(tau, amp, alpha=-2.0/3.0, fL=1.0/(year*20)):
    """
    Compute the covariance function for a powerlaw
    Result is in units of (sec)^2.

    :param tau:
        the time lag

    :param amp:
        amplitude

    :param alpha:
        the GWB spectral index

    :param fL:
        the low-frequency cut-off
    """
    fL = fL * year
    x = 2 * np.pi * fL * np.abs(tau) / year
    cf = ss.gamma(-2+2*alpha) * np.cos(np.pi*alpha)
    power = cf * x**(2-2*alpha)
    ksum = ss.hyp1f2(alpha-1,0.5,alpha,-0.25*x**2)[0]/(2*alpha-2)
    corr = -(year**2 * fL**(-2+2*alpha)) / (12 * np.pi**2) * (power + ksum)
    
    return amp**2*corr


def PL_covmat(toas, amp, alpha=-2.0/3.0, fL=1.0/(year*20)):
    """
    Use the covariance function PL_covfunc to create the covariance matrix for
    'toas'. Result is in units of (sec)^2

    :param tau:
        the time lag

    :param amp:
        amplitude

    :param alpha:
        the GWB spectral index

    :param fL:
        the low-frequency cut-off
    """
    t1, t2 = np.meshgrid(toas, toas)
    tau = np.abs(t1-t2)

    return PL_covfunc(tau, amp, alpha, fL)


# ## A simple pulsar class ##
# 
# This class does not do much. It neatly stores data for you, and conveniently creates the Fourier basis for spectral analysis

# In[6]:

class Pulsar(object):
    def __init__(self, toas, residuals, toaerrs, desmat=None, nfreqs=20,
            qsdorder=2, Tlow=None):
        """
        Initialise the pulsar from data

        :param toas:
            The barycentric times of arrival [MJD]

        :param residuals:
            The timing residuals [sec]

        :param toaerrs:
            The TOA uncertainties [sec]

        :param desmat:
            The design matrix (default None)

        :param nfreqs:
            The number of frequencies we'll use in the Fourier expansion

        :param qsdorder:
            If desmat=None, make a design matrix from scratch, with this order

        :param Tlow:
            1/lowest frequency used in expansion (default length of dataset)(
        """
        self.toas = (toas - mjdT0) * day       # MJD to seconds
        self.residuals = residuals
        self.toaerrs = toaerrs
        self.Mmat = desmat
        self.nobs = len(toas)
        self.nfreqs = nfreqs

        self.T = (np.max(self.toas) - np.min(self.toas))

        if self.Mmat is None:
            self.Mmat = designqsd(self.toas, order=qsdorder)

        # Create the basis of Fourier components
        if Tlow is None:
            Tlow = (np.max(self.toas) - np.min(self.toas))
        (self.Fmat, self.freqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Ttot=Tlow)


def mark1loglikelihood(psr, Aw, Ar, Si):
    """
    Log-likelihood for our pulsar. This one does not marginalize
    over the timing model, so it cannot be used if the data has been
    'fit'. Use when creating data with 'dofit=False':
    psr = Pulsar(dofit=False)
    
    Calculate covariance matrix in the time-domain with:
    
    ll = -0.5 * res^{T} C^{-1} res - 0.5 * log(det(C))
    
    :param psr:
        pulsar object, containing the data and stuff

    :param Aw:
        White noise amplitude, model parameter

    :param Ar:
        Red noise amplitude, model parameter

    :param Si:
        Spectral index of red noise, model parameter
    """
    
    # The function that builds the non-diagonal covariance matrix is Cred_sec
    #Cov = Aw**2 * np.eye(len(psr.toas)) + \
    #      Ar**2 * Cred_sec(psr.toas, alpha=0.5*(3-Si))
    Cov = Aw**2 * np.eye(len(psr.toas)) +         PL_covmat(psr.toas, Ar, alpha=0.5*(3-Si), fL=1.0/(year*20))
    
    cfC = sl.cho_factor(Cov)
    ldetC = 2 * np.sum(np.log(np.diag(cfC[0])))
    rCr = np.dot(psr.residuals, sl.cho_solve(cfC, psr.residuals))
    
    return -0.5 * rCr - 0.5 * ldetC - 0.5*len(psr.residuals)*np.log(2*np.pi)

def mark2loglikelihood(psr, Aw, Ar, Si):
    """
    Log-likelihood for our pulsar
    
    This likelihood does marginalize over the timing model. Calculate
    covariance matrix in the time-domain with:
    
    ll = 0.5 * res^{t} (C^{-1} - C^{-1} M (M^{T} C^{-1} M)^{-1} M^{T} C^{-1} ) res - \
         0.5 * log(det(C)) - 0.5 * log(det(M^{T} C^{-1} M))
         
    In relation to 'mark1loglikelihood', this likelihood has but a simple addition:
    res' = res - M xi
    where M is a (n x m) matrix, with m < n, and xi is a vector of length m. The xi
    are analytically marginalised over, yielding the above equation (up to constants)
    
    :param psr:
        pulsar object, containing the data and stuff

    :param Aw:
        White noise amplitude, model parameter

    :param Ar:
        Red noise amplitude, model parameter

    :param Si:
        Spectral index of red noise, model parameter
    """
    Mmat = psr.Mmat
    
    Cov = Aw**2 * np.eye(len(psr.toas)) + \
        PL_covmat(psr.toas, Ar, alpha=0.5*(3-Si), fL=1.0/(year*20))
    
    cfC = sl.cho_factor(Cov)
    Cinv = sl.cho_solve(cfC, np.eye(len(psr.toas)))
    ldetC = 2 * np.sum(np.log(np.diag(cfC[0])))

    MCM = np.dot(Mmat.T, np.dot(Cinv, Mmat))
    cfM = sl.cho_factor(MCM)
    ldetM = 2 * np.sum(np.log(np.diag(cfM[0])))
    
    wr = np.dot(Cinv, psr.residuals)
    rCr = np.dot(psr.residuals, wr)
    MCr = np.dot(Mmat.T, wr)
    
    return -0.5 * rCr + 0.5 * np.dot(MCr, sl.cho_solve(cfM, MCr)) - \
            0.5 * ldetC - 0.5 * ldetM -0.5*len(psr.residuals)*np.log(2*np.pi)


def mark3loglikelihood(psr, Nvec, psd):
    """
    Log-likelihood for our pulsar (hierarchical/Woodbury)

    :param psr:
        pulsar object, containing the data and stuff

    :param Nvec:
        Vector of squared white noise amplitude (diagonal of covariance matrix)

    :param psd:
        Power-spectral density per frequency bin (of size 1/T)
    """
    Mmat, Fmat = psr.Mmat, psr.Fmat
    Tmat = np.append(Mmat, Fmat, axis=1)
    n, m, l = len(psr.toas), Mmat.shape[1], Fmat.shape[1]

    # Sigma matrix (inverse)
    TNT = np.dot(Tmat.T / Nvec, Tmat)
    phi_inv = np.zeros(m+l)
    phi_inv[m:] = 1.0/psd
    Sigma_inv = TNT + np.diag(phi_inv)

    # Invert Sigma_inv for use in the Woodbury identity
    #Sigma = sl.inv(Sigma_inv)       # Does not use Cholesky
    cf = sl.cho_factor(Sigma_inv)
    STNr = sl.cho_solve(cf, np.dot(Tmat.T / Nvec, psr.residuals))
    rNTSTNr = np.dot(psr.residuals, np.dot((Tmat.T / Nvec).T, STNr))
    rNr = np.sum(psr.residuals**2/Nvec)

    # The full Woodbury expansion
    rCr = rNr - rNTSTNr

    # Determinants
    logNdet = np.sum(np.log(Nvec))
    logPhidet = np.sum(np.log(psd))
    logSigmadet = 2.0*np.sum(np.log(np.diag(cf[0])))

    # Sylvester Matrix identity
    logCdet = logNdet + logPhidet + logSigmadet

    return -0.5*rCr-0.5*logCdet-0.5*n*np.log(2*np.pi)

def pl_loglikelihood(psr, Aw, Ar, Si):
    """
    Log-likelihood for our pulsar (hierarchical/Woodbury)

    :param psr:
        pulsar object, containing the data and stuff

    :param Aw:
        White noise amplitude [sec]
        
    :param Ar:
        Power-law red noise amplitude []

    :param Si:
        Power-law red noise spectral index []
    """
    Nvec = np.ones(len(psr.toas)) * Aw**2
    psd = PL_psd(psr.freqs, Ar, 0.5*(3-Si), sig_fL=1.0/(20.0*year))
    return mark3loglikelihood(psr, Nvec, psd)

def sp_loglikelihood(psr, Aw, psd):
    """
    Log-likelihood for our pulsar (hierarchical/Woodbury)

    :param psr:
        pulsar object, containing the data and stuff

    :param Aw:
        White noise amplitude [sec]
        
    :param psd:
        Vector PSD amplitudes [sec]
    """    
    if len(psd)*2 != psr.Fmat.shape[1]:
        raise ValueError("PSD vector not of appropriate length!")

    Nvec = np.ones(len(psr.toas)) * Aw**2
    return mark3loglikelihood(psr, Nvec, psd.repeat(2))


