from __future__ import division
import numpy as np
import scipy.interpolate as interp

# function to create sine wave
def signal(t, A, f=1e-8, phi=0):
    """
    Return a sinusoid with amplitude A and frequency f and phase phi, 
    sampled at times t.

    :param t: Time samples
    :param A: Amplitude of sine wave
    :param f: Frequency of sine wave
    :param phi: phase of sine wave

    :returns: sine wave at specified t, A, phi and f
    """
    return A*np.sin(2*np.pi*f*t + phi)

# inner product
def innerProduct(data, template, sigma):
    """
    Computes the noise weighted inner product
    of the data and template. If data is d and template is s
    each with N samples, then the inner product is

    (d|s) = \sum_{i=0}^N d_i s_i/\sigma^2

    :param data: data time series
    :param template: signal time series
    :param sigma: standard deviation of noise

    :returns: noise weighted inner product
    """
    return np.dot(data, template)/sigma**2

# simulate a new dataset
def simData(t, A, f, sigma, phi=0):
    """
    Simulated a data set with white noise with standard deviation sigma
    sampled at times t and a sinusoid with amplitude A and frequency f.

    :param t: Time samples
    :param A: Amplitude of sine wave
    :param f: Frequency of sine wave
    :param sigma: Standard deviation of noise
    :param phi: phase of sine wave

    :returns: simulated time series
    """
    sig = signal(t, A, f, phi)
    x = sig + np.random.randn(len(t))*sigma
    return x


def confinterval(samples, sigma=0.68, onesided=False, weights=None, bins=40):
    """
    Given a list of samples, return the desired cofidence intervals.
    Returns the minimum and maximum confidence levels.

    :param samples: Samples that we wish to get confidence intervals
    :param sigma: Percent confidence interval
    :param onesided: Boolean to use onesided or twosided confidence
                     limits.
    :param bins: Number of bins used when constructing hitogram

    :returns: lower bound, upper bound
    """

    # Create the histogram
    hist, xedges = np.histogram(samples, bins=bins, weights=weights)
    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])

    # CDF
    cdf = np.cumsum(hist/hist.sum())

    # interpolate
    x = np.linspace(xedges.min(), xedges.max(), 10000)
    ifunc = interp.interp1d(xedges, cdf, kind='linear')
    y = ifunc(x)
    

    # Find the intervals
    x2min = y[0]
    if onesided:
        bound = 1 - sigma
    else:
        bound = 0.5*(1-sigma)

    for i in range(len(y)):
        if y[i] >= bound:
            x2min = x[i]
            break

    x2max = y[-1]
    if onesided:
        bound = sigma
    else:
        bound = 1 - 0.5 * (1 - sigma)

    for i in reversed(range(len(y))):
        if y[i] <= bound:
            x2max = x[i]
            break

    return x2min, x2max


def confinterval_like(like, x, sigma=0.68):
    """
    Given the values of the likelihood and samples at which
    it is evaluated, return the upper and lower bounds at
    sigma confidence

    :param like: Values of the likelihood function
    :param x: Amplitudes at which likelihood was evaluated
    :param sigma: Percent confidence interval

    :returns: lower bound, upper bound
    """
    
    like /= like.sum()
    
    ind1 = np.flatnonzero(np.cumsum(like) > (1-sigma)/2)[0]
    ind2 = np.flatnonzero(np.cumsum(like) > (1+sigma)/2)[0]
    
    return x[ind1], x[ind2]

