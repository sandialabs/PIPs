import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
import numpy as np
import optax
optimizer = optax.adam(learning_rate = 1e-3)
import matplotlib.pyplot as plt
import scipy.stats as ss
from matplotlib.lines import Line2D



class FullRankVI_DIFF_DATASETS:
    def __init__(self, dim,
                 prior_mean, 
                 prior_cov,
                 noise_cov):
        self.dim = dim #stores dimension of parameter vector
        self.prior_mean = prior_mean #prior mean
        self.prior_cov = prior_cov #prior covariance
        self.noise_cov = noise_cov


    def init_params(self): 
            return {
            "mu": self.prior_mean, #initalialization of mu
            "raw_L": self.prior_cov #initialization of the raw_L
        }

    
    

    def make_L(self, raw_L): #choosing L so that it is a valid Cholesky factor
        L = jnp.tril(raw_L) #gets the lower triangular matrix, inlcuding the diagonal
        diag_raw = jnp.diagonal(L) #Gets the diagonal elements of L
        diag_pos = jnp.exp(diag_raw) #ensures diagonal entries are strictly positve
        L = L - jnp.diag(diag_raw) + jnp.diag(diag_pos) #Lower triangular L with positive entries on the diagonal
        return L #creates lower traiangular structure with positive entries on the diagonal
   
    
    
    def reparameterize(self, key, params): #reparameterization trick
        L = self.make_L(params["raw_L"]) # makes L from raw_L
        key,subkey = random.split(key)
        eps = random.normal(subkey, shape=params["mu"].shape) #samples standard normal noise 
        return params["mu"] + L @ eps, key #generates sampling
    
    
    def log_q(self, theta, params): # computes log density of q(theta
        mu = params["mu"]
        L = self.make_L(params["raw_L"])

        diff = theta - mu
        z = jsp.linalg.solve_triangular(L, diff, lower=True) #solves Lz = x, where x = (theta - mu)
       
        constant_term = self.dim * jnp.log(2 * jnp.pi)
        log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
        inv_term= jnp.dot(z,z) 


        
        return -0.5 * (constant_term + log_det + inv_term)
    
    def generate_data(sef, prior_mean, prior_cov, A, noise_cov,rng):
        print(np.shape(prior_cov))
        print(prior_cov)
        parameter_sample = rng.multivariate_normal(prior_mean, prior_cov)
        noise_sample = rng.multivariate_normal(np.zeros(2), noise_cov)
        data = A @ parameter_sample + noise_sample

        return data, parameter_sample
    

    
    def elbo(self, key, params, A, y, prior_mean, prior_cov_inv, noise_cov_inv,n_samples=1): #Computes the ELBO
        elbo_vals = []

        for _ in range(n_samples):

            theta, key = self.reparameterize(key, params) #generating a sample of theta from the variational distribution q(theta).
    
            def f(theta, A):
                # print("f(theta) called")
                return A @ theta

            y_mod = f(theta,A)
            resid = y - y_mod
            ll_dim = y.shape[0]
            theta_dim = theta.shape[0]

            log_det_noise_cov = -jnp.linalg.slogdet(noise_cov_inv)[1]
            log_det_prior_cov = -jnp.linalg.slogdet(prior_cov_inv)[1]

            ll = -0.5 * (
            resid @ noise_cov_inv @ resid
            + ll_dim * jnp.log(2 * jnp.pi)
            + log_det_noise_cov
        )
            diff = theta - prior_mean
            lp = -0.5 * (
            diff @ prior_cov_inv @ diff
            + theta_dim * jnp.log(2 * jnp.pi)
            + log_det_prior_cov
        )

            lq = self.log_q(theta, params)
             
            elbo_vals.append(ll + lp - lq)
        

        return jnp.mean(jnp.stack(elbo_vals)), key #log likelihood + log prior + log q(theta), also returning the key

    
    
    def fit(self, key, params, optimizer, A, y, prior_mean, prior_cov_inv, noise_cov_inv, n_steps = 1000):
        opt_state = optimizer.init(params)
        loss_history = []
        model_eval_history = []

        @jax.jit
        def train_step(params, opt_state, key):
            key, subkey = jax.random.split(key)

            

            def loss_fn(p):
                elbo, _ = self.elbo(
                subkey,
                p,
                A,
                y,
                prior_mean,
                prior_cov_inv,
                noise_cov_inv,
                n_samples=10
            )
                return -elbo

    
    

            loss_value, grads = jax.value_and_grad(loss_fn)(params)
        
            

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state, key, loss_value
        
        model_evals = 0

        for i in range(n_steps):
            params, opt_state, key, loss_value = train_step(params, opt_state, key)
            model_evals += 1
            loss_history.append(float(loss_value))
            model_eval_history.append(model_evals)


            
            if i % 500 == 0:
                print(f'step {i}, loss: {loss_value}')

            

        return params, opt_state, loss_history, model_eval_history
    

    def plot_distribution_contour(self,mean,cov, param_names = None,ax = None, label = None, color = 'blue'):
        if ax is None:
            fig, ax = plt.subplots()
    
        rv = ss.multivariate_normal(mean=mean, cov=cov)
        
        std = jnp.sqrt(jnp.diag(cov))


        x = jnp.linspace(mean[0] - 3*std[0], mean[0]+3*std[0], 100)
        y = jnp.linspace(mean[1]- 3*std[1] ,mean[1] + 3*std[1],100)

        
        X,Y = jnp.meshgrid(x,y)
        pos = jnp.dstack((X,Y))
        
        post_pdf = jnp.exp(rv.logpdf(pos))
        post_pdf = post_pdf.at[post_pdf==0].set(jnp.nan)

        # distr = rv.pdf(pos)

        contour = ax.contour(X,Y,post_pdf,levels = 8, colors = color)

        if param_names is None:
            param_names = ['Parameter 1', 'Parameter 2']

        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])

        ax.set_title('Probability density contours')

        proxy = Line2D([0], [0], color=color, lw=2, label=label) if label is not None else None

        return ax, proxy