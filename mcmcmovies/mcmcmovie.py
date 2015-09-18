""" Originally based on the matplotlib mixing movie code by Abraham Flaxman.
    Rutger van Haasteren
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import subprocess

def visualize_single_step(chain, i, alpha=0., parlabels=None, description=''):
    """ Show how a random walk in a two dimensional space has
    progressed up to step i"""

    #X = mod.X.trace()      # As done with PyMC
    X = chain
    N, D = X.shape

    if parlabels is None:
        parlabels = ['$X_%d'%j for j in range(D)]

    plt.clf()

    sq_size=.3

    # show 2d trace
    plt.axes([.05, .05, sq_size, sq_size])

    plt.plot(X[:i, 0], X[:i, 1], 'b.-', alpha=.1)

    Y = alpha * X[i, :] + (1 - alpha) *X[i-1, :]
    plt.plot([Y[0], Y[0]], [Y[1], 2.], 'k-', alpha=.5)
    plt.plot([Y[0], 2], [Y[1], Y[1]], 'k-', alpha=.5)
    plt.plot(Y[0], Y[1], 'go')

    """
    if hasattr(mod, 'shape'):
        plt.fill(mod.shape[:,0], mod.shape[:,1], color='b', alpha=.2)
    if hasattr(mod, 'plot_distribution'):
        mod.plot_distribution()
    """

    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.xticks([])
    plt.yticks([])

    # show 1d marginals

    ## X[0] is horizontal position
    plt.axes([.05, .05+sq_size, sq_size, 1.-.1-sq_size])
    plt.plot(X[:(i+1), 0], i+1-np.arange(i+1), 'k-')
    plt.axis([-1.1, 1.1, 0, 1000])
    plt.xticks([])
    plt.yticks([])
    plt.text(-1, .2, parlabels[0])

    ## X[1] is vertical position
    plt.axes([.05+sq_size, .05, 1.-.1-sq_size, sq_size])
    plt.plot(i+1-np.arange(i+1), X[:(i+1), 1], 'k-')
    plt.axis([0, 1000, -1.1, 1.1])
    plt.xticks([])
    plt.yticks([])
    plt.text(10, -1., parlabels[1])

    ## show X[i, j] acorr
    if i > 250:
        for j in range(D):
            plt.axes([1-.1-1.5*sq_size*(1-j*D**-1.), 1.-.1-1.5*sq_size*D**-1, 1.5*sq_size*D**-1., 1.5*sq_size*D**-1.])
            plt.acorr(X[(i/2.):i:10, j], detrend=plt.mlab.detrend_mean)
            plt.xlabel(parlabels[j])
            if j == 0:
                plt.ylabel('autocorr')
            plt.xticks([])
            plt.yticks([])
            plt.axis([-10, 10, -.1, 1])
    ## show X[1] acorr

    ## textual information
    str = ''
    str += 't = %d\n' % i
    str += 'acceptance rate = %.2f\n\n' % (1. - np.mean(np.diff(X[(i/2.):i, 0]) == 0.))

    str += 'mean(X) = %s' % pretty_array(X[(i/2.):i, :].mean(0))
    if False: # hasattr(mod, 'true_mean'):
        str += ' / true mean = %s\n' % pretty_array(mod.true_mean)
    else:
        str += '\n'

    if i > 10:
        iqr = np.sort(X[(i/2.):i,:], axis=0)[[.25*(i/2.), .75*(i/2.)], :].T

        for j in range(D):
            str += 'IQR(X[%d]) = (%.2f, %.2f)' % (j, iqr[j,0], iqr[j,1])
            if False: # hasattr(mod, 'true_iqr'):
                str += ' / true IQR = %s\n' % mod.true_iqr[j]
            else:
                str += '\n'
    plt.figtext(.05 + .01 + sq_size, .05 + .01 + sq_size, str, va='bottom', ha='left')

    plt.figtext(sq_size + .5 * (1. - sq_size), .96, 
               description, va='top', ha='center', size=32)

    plt.figtext(.95, .01, 'credit: Rutger van Haasteren & Justin Ellis', ha='right')

def pretty_array(x):
    return '(%s)' % ', '.join('%.2f' % x_i for x_i in x)


def make_movie(mcmcfilename, avifilename, inds=[0,1], burnin=0, times=None,
        parlabels=None, description=''):
    """
    Make a MCMC movie from an MCMC file

    :param mcmcfilename:
        The filename of the mcmc file. Assumed are that this is a file where all
        columns are values of the mcmc parameters. The parameters given by inds
        are plotted

    :param avifilename:
        Name of the output file

    :param inds:
        Which columns of the mcmc file to plot. Default: [0,1]

    :param burnin:
        How long of a burnin to ignore before creating a movie

    :param times:
        Which sample indices to create a movie frame for. If None, defaults to
        some values

    :param description:
        Short description of the MCMC movie to be shown on screen
    """
    rawchain = np.loadtxt(mcmcfilename)[burnin:,inds]

    meanchain = rawchain - np.mean(rawchain, axis=0)
    chain = 2*meanchain / (np.max(meanchain, axis=0)-np.min(meanchain, axis=0))

    # We need to explicitly set at which samples we create a frame for the movie
    # This is just some scheme I used before. Make sure to provide your own
    if times is None:
        times = list(np.arange(0, 90, .2)) + range(90, 600) + range(600, 4500, 10)
        times += range(4500, 5100) + range(5100, 9000, 10)
        times += range(9000, 9600) + range(9600, len(chain), 10)

    assert np.all(np.diff(times) >= 0.), 'movies where time is not increasing are confusing and probably unintentional'

    try:
        print 'generating %d images' % len(times)
        fit = plt.figure(figsize=(9,7))
        for i, t in enumerate(times):
            if i % 100 == 99:
                print '%d of %d (t=%.2f)' % (i, len(times), t)
            sys.stdout.flush()
            visualize_single_step(chain, int(t), t - int(t),
                    parlabels=parlabels, description=description)
            plt.savefig('mod%06d.png' % i)
    except KeyboardInterrupt:
        pass

    # The encoding can be set with -pvc codec. Now using libavc
    #subprocess.call('mencoder mf://mod*.png -mf w=800:h=600 -ovc x264 -of avi -o %s' % fname, shell=True)
    subprocess.call('mencoder mf://mod*.png -mf w=800:h=600 -ovc lavc -of avi -o %s' % avifilename, shell=True)
    subprocess.call('mplayer -loop 1 %s' % avifilename, shell=True)
    subprocess.call('rm mod*.png', shell=True)
