# EMCEE MCMC

import numpy as np #imports numpy
import emcee #imports emcee
import matplotlib.pyplot as plt
import scipy.stats as ss

class EMCEE_MCMC:
    def __init__(self,
                 prior_mean, 
                 prior_cov,
                 ndim,
                 nwalkers,
                 noise_cov):
        
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.noise_cov = noise_cov
    
        
    def create_log_prob_fn(self, A=None, y=None):
        if A is None: 
            A = self.A
        prior_cov_inv = np.linalg.inv(self.prior_cov)
        noise_cov_inv = np.linalg.inv(self.noise_cov)

        def log_prob_fn(theta):
            theta = np.atleast_1d(theta)


            #log prior
            log_prior = -0.5 * (theta - self.prior_mean).T @ prior_cov_inv @ (theta - self.prior_mean)

            #log likelihood
            log_likelihood = -0.5 * (y - A @ theta).T @ noise_cov_inv @ (y - A @ theta)

            return log_prior + log_likelihood
    
        return log_prob_fn
    
    def run_all(self, A, y, nsteps=None):

        log_prob_fn = self.create_log_prob_fn(A, y)

        ensemble_sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            log_prob_fn
        )

        initial = (
            self.prior_mean
            + 0.1*np.random.randn(
                self.nwalkers,
                self.ndim
            )
        )

        ensemble_sampler.run_mcmc(initial, nsteps, progress=True)

        return ensemble_sampler
    



    def plot_distribution_contour(self,mean,cov, param_names = None):
    
        rv = ss.multivariate_normal(mean=mean, cov=cov)
        
        std = np.sqrt(np.diag(cov))


        x = np.linspace(mean[0] - 3*std[0], mean[0]+3*std[0], 100)
        y = np.linspace(mean[1]- 3*std[1] ,mean[1] + 3*std[1],100)

        
        X,Y = np.meshgrid(x,y)
        pos = np.dstack((X,Y))
        
        post_pdf = np.exp(rv.logpdf(pos))
        post_pdf[post_pdf==0] = np.nan

        # distr = rv.pdf(pos)

        plt.figure(figsize=(6,4))
        plt.contour(X,Y,post_pdf,levels = 8, colors = 'blue')
        if param_names is None:
            param_names = ['Parameter 1', 'Parameter 2']

        plt.xlabel(param_names[0])
        plt.ylabel(param_names[1])
        
        plt.title('Probability density contours')

        plt.tight_layout()
        plt.show()









