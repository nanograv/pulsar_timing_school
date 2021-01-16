from __future__ import division
import numpy as np
import sys

class SimpleMCMC(object):
    
    def __init__(self, lnlikefn, lnpriorfn, sigmas):
        """
        Simple MCMC Sampler class that will perform
        sampling using gaussian jump proposals
        
        :param lnlikefun: ln-likelihood function.
        :param lnpriorfun: ln-prior function.
        :param sigmas: jump size (either single value or vector size of parameter array)
        
        """
        
        # collect input values
        self.lnlikefun = lnlikefn
        self.lnpriorfn = lnpriorfn
        self.sigmas = np.atleast_1d(sigmas)
        
    
    def sample(self, p0, N):
        """
        Sample for N steps.
        
        :param p0: Initial parameter array
        :param N: Number of iterations
        
        """
        
        self.ndim = len(p0)
        naccepted = 0
        
        # initialize arrays
        self.chain = np.zeros((N, self.ndim))
        self.lnlike, self.lnprob = np.zeros(N), np.zeros(N)
        
        # get likelihood and prior for initial point
        lnlike0, lnprior0 = self.lnlikefun(p0), self.lnpriorfn(p0)
        
        self.iter = 0
        # start iterations
        for ii in range(N):
            
            # jump
            p1, qxy = self.jump(p0)
            
            # compute likelihoods and priors for new point
            lnprior1 = self.lnpriorfn(p1)
            if lnprior1 == -np.inf:
                lnlike1 = -np.inf
            else:
                lnlike1 = self.lnlikefun(p1)
            
            # hastings step
            diff = (lnlike1 + lnprior1) - (lnlike0 + lnprior0) + qxy
            if diff > np.log(np.random.rand()):

                # accept jump
                p0, lnlike0, lnprior0 = p1, lnlike1, lnprior1
                naccepted += 1
                
            # update arrays
            self.chain[ii,:] = p0
            self.lnlike[ii], self.lnprob[ii] = lnlike0, (lnlike0 + lnprior0)

            
            if self.iter % 100 == 0 and self.iter > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent. Acceptance Rate: %g'%(self.iter / N * 100, naccepted / self.iter * 100))
                sys.stdout.flush()
            
            self.iter += 1
            
    
    def jump(self, p0):
        
        """
        Standard gaussian jump
        
        :param p0: Current parameter vector
        
        :returns: new parameter vector, transition probability
        """
        
        qxy = 0
        q = p0.copy()
        
        # 50/50 jup probability
        jumps = ['gaussian', 'de']
        jprobs = [0.5, 0.5]

        # draw jump type
        jname = np.random.choice(jumps, p=jprobs)
        
        # differential evolution
        if jname == 'de' and self.iter > 1000:

            # draw a random integer from 0 - iter
            mm = np.random.randint(0, self.iter)
            nn = np.random.randint(0, self.iter)

            # make sure mm and nn are not the same iteration
            while mm == nn:
                nn = np.random.randint(0, self.iter)

            # mode jump
            if np.random.rand() > 0.5:
                scale = 1.0

            else:
                scale = 2.4 / np.sqrt(2 * self.ndim)

            q += scale * (self.chain[mm,:] - self.chain[nn,:])

        # standard gaussian default
        else:
            # step sizes
            probs = [0.05, 0.7, 0.25]
            sizes = [0.1, 1.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            q += np.random.randn(len(q)) * self.sigmas * scale


        
        return q, qxy
    
    
class PTMCMC(object):
    
    def __init__(self, lnlikefn, lnpriorfn, ndim, sigmas, ntemp):
        """
        Simple PTMCMC Sampler class that will perform
        sampling using gaussian jump proposals
        
        :param lnlikefun: ln-likelihood function.
        :param lnpriorfun: ln-prior function.
        :param ndim: Dimension of problem
        :param sigmas: jump size (either single value or vector size of parameter array)
        :param ntemp: Number of different temperature chains to run
        
        """
        
        # collect input values
        self.lnlikefn = lnlikefn
        self.lnpriorfn = lnpriorfn
        self.ndim = ndim
        self.sigmas = np.atleast_1d(sigmas)
        self.ntemp = ntemp
        
        # acceptance arrays
        self.naccepted = np.zeros(self.ntemp)
        self.nswap_proposed = np.zeros(self.ntemp)
        self.nswap_accepted = np.zeros(self.ntemp)
        
        # set up temperature ladder
        tstep = 1 + np.sqrt(2 / self.ndim)
        self.ladder = np.zeros(self.ntemp)
        for ii in range(self.ntemp):
            self.ladder[ii] = tstep ** ii   
            
    def PTlnlike(self, x):
        
        lnprior = self.lnpriorfn(x)
        if lnprior == -np.inf:
            lnlike = -np.inf
        else:
            lnlike = self.lnlikefn(x)
        
        return lnlike, lnprior
    
    def sample(self, p0, N):
        """
        Sample for N steps.
        
        :param p0: Initial parameter array
        :param N: Number of iterations
        
        """
        
        # initialize arrays
        self.chain = np.zeros((self.ntemp, N, self.ndim))
        self.lnlike, self.lnprob = np.zeros((self.ntemp, N)), np.zeros((self.ntemp, N))
        
        # get likelihood and prior for initial point
        results = map(self.PTlnlike, p0)
        lnlike0 = np.array([r[0] for r in results])
        lnprior0 = np.array([r[1] for r in results])
        
        self.iter = 0
        # start iterations
        for ii in range(N):
            
            # jump
            jumps = [self.jump(p, temp) for p, temp in zip(p0, self.ladder)]
            p1 = np.array([j[0] for j in jumps])
            qxy = np.array([j[1] for j in jumps])
            
            # compute likelihoods and priors for new point
            results = map(self.PTlnlike, p1)
            lnlike1 = np.array([r[0] for r in results])
            lnprior1 = np.array([r[1] for r in results])
            
            # hastings step
            diff = (lnlike1 - lnlike0) / self.ladder + (lnprior1 - lnprior0) + qxy
            accepts = diff > np.log(np.random.rand(self.ntemp))
            
            # update
            p0[accepts] = p1[accepts]
            lnlike0[accepts] = lnlike1[accepts]
            lnprior0[accepts] = lnprior1[accepts]
            self.naccepted[accepts] += 1
                
            # update arrays
            self.chain[:,ii,:] = p0
            self.lnlike[:, ii], self.lnprob[:, ii] = lnlike0, (lnlike0 / self.ladder + lnprior0)
            
            # temperature swap
            if self.iter % 10 == 0:
                p0, lnlike0, lnprob0 = self.temperature_swap(p0, lnlike0, lnprior0)

            
            if self.iter % 100 == 0 and self.iter > 0:
                sys.stdout.write('\r')
                sys.stdout.write('Finished %g percent. Acceptance Rate: %g'%(self.iter / N * 100, self.naccepted[0] / self.iter * 100))
                sys.stdout.flush()
            
            self.iter += 1

    def jump(self, p0, temp=1):
        
        """
        Standard gaussian jump
        
        :param p0: Current parameter vector
        :param temp: Temperature of chain
        
        :returns: new parameter vector, transition probability
        """
        
        qxy = 0
        q = p0.copy()
        
        # 50/50 jup probability
        jumps = ['gaussian', 'de']
        jprobs = [0.5, 0.5]

        # draw jump type
        jname = np.random.choice(jumps, p=jprobs)

        # temperature index
        tidx = self.ladder == temp
        
        # differential evolution
        if jname == 'de' and self.iter > 1000:

            # draw a random integer from 0 - iter
            mm = np.random.randint(0, self.iter)
            nn = np.random.randint(0, self.iter)

            # make sure mm and nn are not the same iteration
            while mm == nn:
                nn = np.random.randint(0, self.iter)

            # mode jump
            if np.random.rand() > 0.5:
                scale = 1.0

            else:
                scale = 2.4 / np.sqrt(2 * self.ndim)

            q += scale * (self.chain[tidx,mm,:] - self.chain[tidx,nn,:]).flatten()

        # standard gaussian default
        else:
            # step sizes
            probs = [0.05, 0.7, 0.25]
            sizes = [0.1, 1.0, 10.0]
            scale = np.random.choice(sizes, p=probs)
            q += np.random.randn(len(q)) * self.sigmas * scale


        
        return q, qxy
            
    
    def temperature_swap(self, p, lnlike, lnprior):
        """
        Proposes temperature swaps between adjacent chains
        
        :param p: Current parameter vector (ntemp, ndim)
        :param lnlike: Current ln-likelihood values (ntemp)
        :param lnprior: Current ln-prior values (ntemp)
        
        :return: (p, lnlike, lnprior) updated values after swap
        """
               
        # loop from hottest to coldest chain
        for ii in range(self.ntemp-1, 0, -1):
            
            self.nswap_proposed[ii-1] += 1
            
            # attempt swap with next coldest chain
            diff = (1 / self.ladder[ii] - 1 / self.ladder[ii-1]) * (lnlike[ii-1] - lnlike[ii])
            accept = diff > np.log(np.random.rand())
            
            if accept:
                
                # temporary arrays
                ptemp = np.copy(p[ii,:])
                lltemp = np.copy(lnlike[ii])
                lptemp = np.copy(lnprior[ii])
                
                # swap values
                p[ii] = p[ii-1]
                lnlike[ii] = lnlike[ii-1]
                lnprior[ii] = lnprior[ii-1]
                
                p[ii-1] = ptemp
                lnlike[ii-1] = lltemp
                lnprior[ii-1] = lptemp
                
                self.nswap_accepted[ii-1] += 1
                
        return p, lnlike, lnprior
