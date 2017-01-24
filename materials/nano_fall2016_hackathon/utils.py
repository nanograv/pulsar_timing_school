from __future__ import division

import numpy as np


def python_block_shermor_2D(Z, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    :param Z:       The design matrix, array (n x m)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    zNz = np.dot(Z.T*ni, Z)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                zNz -= beta * np.outer(zn.T, zn)

    return zNz

def python_block_shermor_2D2(Z, X, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiX

    :param Z:       The design matrix, array (n x m)
    :param X:       The second design matrix, array (n x l)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: Z.T * N^-1 * X
    """
    ni = 1.0 / Nvec
    zNx = np.dot(Z.T*ni, X)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
                Xblock = X[Uinds[cc,0]:Uinds[cc,1], :]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                zNx -= beta * np.outer(zn.T, xn)

    return zNx

def python_block_shermor_0D(r, Nvec, Jvec, Uinds): 
    """
    Sherman-Morrison block-inversion for Jitter 
    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    
    ni = 1/Nvec
    Nx = r/Nvec
    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                rblock = r[Uinds[cc,0]:Uinds[cc,1]]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                Nx[Uinds[cc,0]:Uinds[cc,1]] -= beta * np.dot(niblock, rblock) * niblock

    return Nx


def python_block_shermor_1D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: r.T * N^-1 * r, log(det(N))
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)
    
    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                rblock = r[Uinds[cc,0]:Uinds[cc,1]]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                xNx -= beta * np.dot(rblock, niblock)**2
                Jldet += np.log(jv) - np.log(beta)

    return Jldet, xNx


def quantize_fast(times, dt=1.0, calci=False):
    """ Adapted from libstempo: produce the quantisation matrix fast """
    isort = np.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def quantize_split(times, flags, dt=1.0, calci=False):
    """
    As quantize_fast, but now split the blocks per backend. Note: for
    efficiency, this function assumes that the TOAs have been sorted by
    argsortTOAs. This is _NOT_ checked.
    """
    isort = np.arange(len(times))
    
    bucket_ref = [times[isort[0]]]
    bucket_flag = [flags[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt and flags[i] == bucket_flag[-1]:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_flag.append(flags[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv



def argsortTOAs(toas, flags, which=None, dt=1.0):
    """
    Return the sort, and the inverse sort permutations of the TOAs, for the
    requested type of sorting

    NOTE: This one is _not_ optimized for efficiency yet (but is done only once)

    :param toas:    The toas that are to be sorted
    :param flags:   The flags that belong to each TOA (indicates sys/backend)
    :param which:   Which type of sorting we will use (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks, default [10 secs]

    :return:    perm, perminv       (sorting permutation, and inverse)
    """
    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        isort = np.argsort(toas, kind='mergesort')
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)
                    
                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Sort the indices of this epoch and backend
                        # We need mergesort here, because it is stable
                        # (A stable sort keeps items with the same key in the
                        # same relative order. )
                        episort = np.argsort(flagmask[colmask], kind='mergesort')
                        isort[colmask] = isort[colmask][episort]
                else:
                    # Only one element, always ok
                    pass

        # Now that we have a correct permutation, also construct the inverse
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    else:
        isort, iisort = np.arange(len(toas)), np.arange(len(toas))

    return isort, iisort

def checkTOAsort(toas, flags, which=None, dt=1.0):
    """
    Check whether the TOAs are indeed sorted as they should be according to the
    definition in argsortTOAs

    :param toas:    The toas that are supposed to be already sorted
    :param flags:   The flags that belong to each TOA (indicates sys/backend)
    :param which:   Which type of sorting we will check (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks, default [10 secs]

    :return:    True/False
    """
    rv = True
    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        if not np.all(isort == np.arange(len(isort))):
            rv = False
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        #isort = np.argsort(toas, kind='mergesort')
        isort = np.arange(len(toas))
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)
                    
                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Keys are not sorted for this epoch/flag
                        rv = False
                else:
                    # Only one element, always ok
                    pass
    else:
        pass

    return rv


def checkquant(U, flags, uflagvals=None):
    """
    Check the quantization matrix for consistency with the flags

    :param U:           quantization matrix
    :param flags:       the flags of the TOAs
    :param uflagvals:   subset of flags that are not ignored

    :return:            True/False, whether or not consistent

    The quantization matrix is checked for three kinds of consistency:
    - Every quantization epoch has more than one observation
    - No quantization epoch has no observations
    - Only one flag is allowed per epoch
    """
    if uflagvals is None:
        uflagvals = list(set(flags))

    rv = True
    collisioncheck = np.zeros((U.shape[1], len(uflagvals)), dtype=np.int)
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)

        Umat = U[flagmask, :]

        simepoch = np.sum(Umat, axis=0)
        if np.all(simepoch <= 1) and not np.all(simepoch == 0):
            rv = False
            #raise ValueError("quantization matrix contains non-jitter-style data")

        collisioncheck[:, ii] = simepoch

        # Check continuity of the columns
        for cc, col in enumerate(Umat.T):
            if np.sum(col > 2):
                # More than one TOA for this flag/epoch
                epinds = np.flatnonzero(col)
                if len(epinds) != epinds[-1] - epinds[0] + 1:
                    rv = False
                    print("WARNING: checkquant found non-continuous blocks")
                    #raise ValueError("quantization matrix epochs not continuous")
        

    epochflags = np.sum(collisioncheck > 0, axis=1)

    if np.any(epochflags > 1):
        rv = False
        print("WARNING: checkquant found multiple backends for an epoch")
        #raise ValueError("Some observing epochs include multiple backends")

    if np.any(epochflags < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (eflags)")
        #raise ValueError("Some observing epochs include no observations... ???")

    obsum = np.sum(U, axis=0)
    if np.any(obsum < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (all)")
        #raise ValueError("Some observing epochs include no observations... ???")

    return rv


def quant2ind(U):
    """
    Convert the quantization matrix to an indices matrix for fast use in the
    jitter likelihoods

    :param U:       quantization matrix
    
    :return:        Index (basic slicing) version of the quantization matrix

    This function assumes that the TOAs have been properly sorted according to
    the proper function argsortTOAs above. Checks on the continuity of U are not
    performed
    """
    inds = np.zeros((U.shape[1], 2), dtype=np.int)
    for cc, col in enumerate(U.T):
        epinds = np.flatnonzero(col)
        inds[cc, 0] = epinds[0]
        inds[cc, 1] = epinds[-1]+1

    return inds

def quantreduce(U, eat, flags, calci=False):
    """
    Reduce the quantization matrix by removing the observing epochs that do not
    require any jitter parameters.

    :param U:       quantization matrix
    :param eat:     Epoch-averaged toas
    :param flags:   the flags of the TOAs
    :param calci:   Calculate pseudo-inverse yes/no

    :return     newU, jflags (flags that need jitter)
    """
    uflagvals = list(set(flags))
    incepoch = np.zeros(U.shape[1], dtype=np.bool)
    jflags = []
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)
        
        Umat = U[flagmask, :]
        ecnt = np.sum(Umat, axis=0)
        incepoch = np.logical_or(incepoch, ecnt>1)

        if np.any(ecnt > 1):
            jflags.append(flagval)

    Un = U[:, incepoch]
    eatn = eat[incepoch]

    if calci:
        Ui = ((1.0/np.sum(Un, axis=0)) * Un).T
        rv = (Un, Ui, eatn, jflags)
    else:
        rv = (Un, eatn, jflags)

    return rv

def createfourierdesignmatrix(t, nmodes, freq=False, Tspan=None,
                              logf=False, fmin=None, fmax=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing

    :return: F: fourier design matrix
    :return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1/T, nmodes/T, nmodes)
    if logf:
        f = np.logspace(np.log10(1/T), np.log10(nmodes/T), nmodes)
        #f = np.logspace(np.log10(1/2/T), np.log10(nmodes/T), nmodes)
    Ffreqs = np.zeros(2*nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = np.cos(2*np.pi*f[ct]*t)
        F[:,ii+1] = np.sin(2*np.pi*f[ct]*t)
        ct += 1
    
    if freq:
        return F, Ffreqs
    else:
        return F

def computeORFMatrix(psr):
    """
    Compute ORF matrix.

    :param psr: List of pulsar object instances

    :return: Matrix that has the ORF values for every pulsar
             pair with 2 on the diagonals to account for the 
             pulsar term.

    """

    # begin loop over all pulsar pairs and calculate ORF
    npsr = len(psr)
    ORF = np.zeros((npsr, npsr))
    phati = np.zeros(3)
    phatj = np.zeros(3)
    for ll in xrange(0, npsr):
        phati[0] = np.cos(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[1] = np.sin(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[2] = np.cos(psr[ll].theta)

        for kk in xrange(0, npsr):
            phatj[0] = np.cos(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[1] = np.sin(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[2] = np.cos(psr[kk].theta)
            
            if ll != kk:
                xip = (1.-np.sum(phati*phatj)) / 2.
                ORF[ll, kk] = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
            else:
                ORF[ll, kk] = 2.0

    return ORF

