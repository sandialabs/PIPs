# EMCEE MCMC

import numpy as np #imports numpy
import emcee #imports emcee
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.lines import Line2D
from BayesianMVLinGaussIP import BayesianMVLinGaussIP

class EMCEE_MCMC_DIFF_DATASETS:
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

        
        calls = 0
        def log_prob_fn(theta):
            nonlocal calls
            calls +=1
            
            # theta = np.atleast_1d(theta)
            theta = np.asarray(theta).reshape(-1)


            #log prior
            log_prior = -0.5 * (theta - self.prior_mean).T @ prior_cov_inv @ (theta - self.prior_mean)

            #log likelihood
            log_likelihood = -0.5 * (y - A @ theta).T @ noise_cov_inv @ (y - A @ theta)

            return log_prior + log_likelihood
        def get_calls():
            return calls
    
        return log_prob_fn, get_calls

    def generate_data(sef, prior_mean, prior_cov, A, noise_cov,rng):
        parameter_sample = rng.multivariate_normal(prior_mean, prior_cov)
        noise_sample = rng.multivariate_normal(np.zeros(2), noise_cov)
        data = A @ parameter_sample + noise_sample

        return data, parameter_sample
    
    def run_all(self, A, nsteps=1, burn_in = 0, base_seed = 123, Nrep = 50):

         
        
        results =[]
        mean_errors_emcee = []
        trace_errors_emcee = []
        for i in range(Nrep):
            data_seed = base_seed + i
            data_rng = np.random.default_rng(data_seed)

            data, true_parameter = self.generate_data(
                A=A, 
                prior_mean = self.prior_mean,
                prior_cov = self.prior_cov,
                noise_cov = self.noise_cov,
                rng = data_rng
            )

            model = BayesianMVLinGaussIP(prior_mean=self.prior_mean, prior_cov = self.prior_cov, dim = 2, noise_cov=self.noise_cov) 
            # TNP: True posterior for comparison to computed one.
            exact_posterior_cov, exact_posterior_mean  = model.posterior_calculations(
                        A=A,
                        y=data
                    )

            log_prob_fn, get_calls = self.create_log_prob_fn(A, data)

            init_seed = base_seed + 1000 + i
            np.random.seed(init_seed)

            ensemble_sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                log_prob_fn
            )

            initial = np.random.multivariate_normal(
                mean=self.prior_mean,
                cov=self.prior_cov,
                size=self.nwalkers,
                )


            ensemble_sampler.run_mcmc(initial, nsteps, progress=False)

            #results.append(ensemble_sampler.get_chain(discard = 0, flat = True))
            chain = ensemble_sampler.get_chain(discard = burn_in, flat = True)
            emcee_mean = np.mean(chain,axis = 0)
            emcee_cov = np.cov(chain.T)
            mean_error_emcee = np.linalg.norm(emcee_mean - exact_posterior_mean)
            emcee_trace = np.trace(emcee_cov)
            trace_actual = np.trace(exact_posterior_cov)
            mean_error_trace = abs(emcee_trace - trace_actual)

            
            results.append({"chain":chain,
                            "data": data,
                            "true_parameter": true_parameter})
            mean_errors_emcee.append(mean_error_emcee)
            trace_errors_emcee.append(mean_error_trace)


        return results, mean_errors_emcee, trace_errors_emcee
    
    
    



    def plot_distribution_contour(self,mean,cov, param_names = None, ax = None, label = None, color = 'blue'):
        if ax is None:
            fig, ax = plt.subplots()

        rv = ss.multivariate_normal(mean=mean, cov=cov)
        
        std = np.sqrt(np.diag(cov))


        x = np.linspace(mean[0] - 3*std[0], mean[0]+3*std[0], 100)
        y = np.linspace(mean[1]- 3*std[1] ,mean[1] + 3*std[1],100)

        
        X,Y = np.meshgrid(x,y)
        pos = np.dstack((X,Y))
        
        post_pdf = np.exp(rv.logpdf(pos))
        post_pdf[post_pdf==0] = np.nan

        # distr = rv.pdf(pos)
        from matplotlib.ticker import MaxNLocator
       
        contour = ax.contour(X,Y,post_pdf,levels = 6, colors = color,lw = 1.0)

        # if param_names is None:
            # param_names = ['Parameter 1', 'Parameter 2']

        # Remove top and right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # ax.set_xlabel(param_names[0])
        # ax.set_ylabel(param_names[1])

        # Remove tick marks and tick labels
        ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )

        # ax.set_title('Probability density contours', fontsize = 18)

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))




        proxy = Line2D([0], [0], color=color, lw=2, label=label) if label is not None else None

        return ax, proxy