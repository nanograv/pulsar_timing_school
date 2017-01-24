from __future__ import division

import numpy as np
import ephem
import scipy.linalg as sl
import libstempo as t2
import utils

# Pulsar Class
class Pulsar(object):
    
    def __init__(self, parfile, timfile):
        
        # initialize libstempo object
        t2psr = t2.tempopulsar(parfile, timfile, maxobs=30000, )
        
        # get attributes
        self.name = t2psr.name
        self.residuals = np.double(t2psr.residuals())
        self.toas = np.double(t2psr.toas()*86400)
        self.toaerrs = np.double(t2psr.toaerrs*1e-6)
        self.flags = t2psr.flagvals('f')
        Mmat = np.double(t2psr.designmatrix())
        
        # sort
        self.isort, self.iisort = utils.argsortTOAs(self.toas, self.flags, 
                                                       which='jitterext', dt=1.0)
        
        self.toas = self.toas[self.isort]
        self.toaerrs = self.toaerrs[self.isort]
        self.residuals = self.residuals[self.isort]
        self.flags = self.flags[self.isort]
        Mmat = Mmat[self.isort, :]
        
        u, s, v = np.linalg.svd(Mmat, full_matrices=False)
        self.Musvd = u
        self.Mssvd = s
        
        # get sky location
        pars = t2psr.pars()
        if 'RAJ' not in pars and 'DECJ' not in pars:
            elong, elat = t2psr['ELONG'].val, t2psr['ELAT'].val
            ec = ephem.Ecliptic(elong, elat)
                
            # check for B name
            if 'B' in t2psr.name:
                epoch = '1950'
            else:
                epoch = '2000'
            eq = ephem.Equatorial(ec, epoch=epoch)
            self.phi = np.double([eq.ra])
            self.theta = np.pi/2 - np.double([eq.dec])
        
        else:
            self.phi = np.double(t2psr['RAJ'].val)
            self.theta = np.pi/2 - np.double(t2psr['DECJ'].val)



# Model Class
class Model(object):
    
    def __init__(self, psrs):
        self.psr = psrs
        
    def setup(self, nf=30):
        """
        Simple model setup with red noise, GWB, 
        EFAC, EQUAD, and ECORR 
        """
        
        Tmax = np.max([p.toas.max()-p.toas.min() for p in self.psr])
        
        # loop over pulsars
        for ct, p in enumerate(self.psr):

            # get Fourier matrix and frequencies
            p.Fmat, p.Ffreqs = utils.createfourierdesignmatrix(p.toas, nf, 
                                                                  freq=True, 
                                                                  Tspan=Tmax)

            # build Tmatrix
            p.Tmat = np.concatenate((p.Musvd, p.Fmat), axis=1)

            # initialize N
            p.Nvec = p.toaerrs**2

            # get quantization matrix
            avetoas, Umat, Ui = utils.quantize_split(p.toas,
                                                        p.flags,
                                                        dt=1.0,
                                                        calci=True)

            # get only epochs that need jitter/ecorr
            p.Umat, p.avetoas, aveflags = utils.quantreduce(
                                                    Umat, avetoas,
                                                    p.flags)

            # get quantization indices
            p.Uinds = utils.quant2ind(p.Umat)
            p.aveflags = p.flags[p.Uinds[:, 0]]
            
        # get HD correlation matrix
        self.corrmat = utils.computeORFMatrix(self.psr) / 2
        
        # initialize matrices
        tot = np.sum([p.Tmat.shape[1] for p in self.psr])
        self.TNT = np.zeros((tot, tot))
        self.Sigma = np.zeros((tot, tot))
        self.Phiinv = np.zeros((tot, tot))
        
        
    def set_nvec(self, pars):
        
        npsr = len(self.psr)
        
        # get parameters
        efacs = pars[::5]
        equads = 10**pars[1::5]
        ecorrs = 10**pars[2::5]
        
        for ct, p in enumerate(self.psr):
            
            # set Nvec
            p.Nvec = efacs[ct] * p.toaerrs**2 + equads[ct]**2
            
            # set ECORR prior amplitudes
            p.Jvec = np.ones(len(p.avetoas)) * ecorrs[ct]**2
    
    
    def set_phi_matrix(self, pars):
        
        npsr = len(self.psr)
        Areds = 10**pars[3::5]
        greds = pars[4::5]
        Agw = 10**pars[-2]
        ggw = pars[-1]
        
        Tmax = np.max([p.toas.max()-p.toas.min() for p in self.psr])
        
        # begin loop over pulsars 
        sigdiag, sigoffdiag = [], []
        for ct, p in enumerate(self.psr):
            
            freq = p.Ffreqs
            f1yr = 1 / 3.16e7
            rns = Areds[ct] ** 2 / 12 / np.pi ** 2 * f1yr ** (greds[ct] - 3) * \
                    freq ** (-greds[ct]) / Tmax
            gws = Agw ** 2 / 12 / np.pi ** 2 * f1yr ** (ggw - 3) * freq ** (-ggw) / Tmax
            
            sigdiag.append(rns+gws)
            sigoffdiag.append(gws)
            
        # get Phi inverse matrix
        nftot = len(self.psr[0].Ffreqs)
        smallMatrix = np.zeros((nftot, npsr, npsr))
        for ii in range(npsr):
            for jj in range(ii,npsr):

                if ii == jj:
                    smallMatrix[:,ii,jj] = sigdiag[jj] 
                else:
                    smallMatrix[:,ii,jj] = self.corrmat[ii,jj] * sigoffdiag[jj]
                    smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]
        
        # invert them
        self.logdetPhi = 0
        for ii in range(nftot):
            try:
                L = sl.cho_factor(smallMatrix[ii, :, :])
            except np.linalg.LinAlgError:
                return 0
            smallMatrix[ii, :, :] = sl.cho_solve(L, np.eye(npsr))
            self.logdetPhi += np.sum(2 * np.log(np.diag(L[0])))
        
        
        # put back together
        stop = np.cumsum([p.Tmat.shape[1] for p in self.psr])
        start = stop - nftot
        ind = [np.arange(sta, sto) for sta, sto in zip(start, stop)]
        for ii in range(npsr):
            for jj in range(npsr):
                self.Phiinv[ind[ii], ind[jj]] = smallMatrix[:, ii, jj]
                
    
    def get_likelihood(self, pars):
        
        loglike = 0

        # set pulsar white noise parameters

        self.set_nvec(pars)
    
        # set red noise and GW parameters
        self.set_phi_matrix(pars)

        # compute the white noise terms in the log likelihood
        nfref = 0
        for ct, p in enumerate(self.psr):

            nf = p.Tmat.shape[1]

            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(p.Tmat.T, utils.python_block_shermor_0D(
                    p.residuals, p.Nvec, p.Jvec, p.Uinds))
            else:
                d = np.append(d, np.dot(p.Tmat.T, utils.python_block_shermor_0D(
                    p.residuals, p.Nvec, p.Jvec, p.Uinds)))

            # compute T^T N^{-1} T
            self.TNT[nfref:(nfref + nf), nfref:(nfref + nf)] = \
                utils.python_block_shermor_2D(
                    p.Tmat, p.Nvec, p.Jvec, p.Uinds)

            # triple product in likelihood function
            logdet_N, rNr = utils.python_block_shermor_1D(p.residuals,
                                                             p.Nvec, p.Jvec, 
                                                             p.Uinds)

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr) - 0.5 * \
                len(p.toas) * np.log(2 * np.pi)
                
            nfref += nf

        # compute sigma
        self.Sigma = self.TNT + self.Phiinv

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(self.Sigma)
            expval2 = sl.cho_solve(cf, d)
            self.logdet_Sigma = np.sum(2 * np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            return -np.inf

        loglike += -0.5 * \
            (self.logdetPhi + self.logdet_Sigma) + \
            0.5 * (np.dot(d, expval2))
            
        return loglike         
