#Dual Problem Bayes Class
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from BayesianMVLinGaussIP import BayesianMVLinGaussIP
import scipy.stats as ss

# Enable 64-bit precision for better numerical stability in optimization (optional)
jax.config.update("jax_enable_x64", True)

class Dual_Problem_Bayes_Class_DIFF_DATASETS:
    def __init__(self, 
                 prior_mean,
                 prior_cov,
                 ndim,
                 noise_cov):
         
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.ndim = ndim
        self.noise_cov = noise_cov

    def solve_saa_jax(self, g_samples, sigma, psi, t_init=0.0): #Similar to MCMC
        """
        saa: sample average approximation

        Solves min_t { t + E[psi( (g(theta) - t) / sigma )] } using JAX.

        Args:
            g_samples (jnp.array): Pre-computed samples of the log-likelihood g(theta).
            sigma (float): Scaling parameter.
            psi (callable): JAX-compatible function psi(x).
            t_init (float): Initial guess for t.

        Returns:
            dict: Result dictionary containing optimal t and minimum value.
        """

         # Convert inputs to JAX arrays
        g_samples = jnp.array(g_samples)

        # --- 1. Define the Objective Function in JAX ---
        def objective_fn(t):
            # TNP: (24) objective function in the paper
            # Ensure t is a scalar for broadcasting
            # Calculate argument: (g(theta) - t) / sigma
            m = (g_samples - t) / sigma

            # Apply psi and take the sample average (Expectation)
            expectation = jnp.mean(psi(m))
            scaled_expectation = sigma * expectation

            psi_m = psi(m)
            out = t + scaled_expectation

            #print("psi(m) shape:", jnp.shape(psi_m))
            #print("expectation shape:", jnp.shape(expectation))
            #print("out shape:", jnp.shape(out))


            return t + scaled_expectation

        # --- 2. Create Value-and-Gradient Function ---
        # jax.value_and_grad returns a tuple: (loss_value, gradient)
        # jax.jit compiles the function for high performance
        val_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))

        # --- 3. Wrapper for Scipy ---
        # Scipy expects the function to accept a numpy array (even if scalar)
        # and return numpy types, not JAX device arrays.
        def scipy_wrapper(t_arr):
            val, grad = val_and_grad_fn(t_arr[0])
            return float(val), np.array([float(grad)])

        # --- 4. Optimization ---
        # BFGS is efficient for smooth, convex scalar optimization using gradients
        # TNP: solving (24)
        res = minimize(
            fun=scipy_wrapper,
            x0=np.array([t_init]),
                 method='BFGS',
            jac=True  # We provide the gradient
        )

        return {
            "optimal_t": res.x[0],
            "min_value": res.fun,
            "success": res.success,
            "message": res.message
        }
    
    def generate_data(sef, prior_mean, prior_cov, A, noise_cov,rng):
        parameter_sample = rng.multivariate_normal(prior_mean, prior_cov)
        noise_sample = rng.multivariate_normal(np.zeros(2), noise_cov)
        data = A @ parameter_sample + noise_sample
        print("parameter_sample shape:", parameter_sample.shape)
        print("noise_sample shape:", noise_sample.shape)
        print("data shape:", data.shape)

        return data, parameter_sample
    

 
    def sample_from_prior(self, N, Nrep=50, base_seed=123):
        samples = []
        for i in range(Nrep):
            rng = np.random.default_rng(base_seed + 1000 + i)
            samples.append(
            rng.multivariate_normal(
                    self.prior_mean,
                    self.prior_cov,
                    size=N
                )
            )
        return np.stack(samples, axis=0)

    def get_g_theta(self, data, A, noise_cov, theta_samples):
        return self.vectorized_log_likelihood(data, theta_samples, A, noise_cov)
    
    def scalar_log_likelihood(self, data, theta_vals, A, noise_cov):
        """
        Calculates the log likelihood for an array of theta values.

        data: (n,) array of observations
        theta_vals: (m,) array of proposed means
        sigma: standard deviation (scalar)
        """
        # n = len(data)
        n = data.shape[0]
        noise_cov_inv = np.linalg.inv(noise_cov)
        _, logdet = np.linalg.slogdet(noise_cov)
        # Use broadcasting: (n, 1) - (1, m) results in (n, m) matrix
        y_pred = theta_vals @ A.T       # (N,n)

        residuals = data - y_pred
    

        quad = residuals @ noise_cov_inv @ residuals        # (N,n)

  

        return -0.5 * (
            quad
            + logdet
            + n * np.log(2*np.pi))



    def vectorized_log_likelihood(self, data, theta_vals, A, noise_cov):
        #print("data.shape:", data.shape)
        #print("theta_vals.shape:", theta_vals.shape)
        #print("A.shape:", A.shape)
        #print("noise_cov.shape:", noise_cov.shape)
        n = data.shape[0]
        noise_cov_inv = np.linalg.inv(noise_cov)
        _, logdet = np.linalg.slogdet(noise_cov)

        # theta_batch: (m, p)
        # A: (n, p)
        y_pred = theta_vals @ A.T      # (m, n)
        #print("y_pred.shape:", y_pred.shape)

        residuals = data[None,:] - y_pred       # (m, n)
        #print("residuals.shape:", residuals.shape)

        first_r = residuals[0]
        #print("first_r.shape:", first_r.shape)

        quad = np.sum((residuals @ noise_cov_inv) * residuals, axis=1)

        return -0.5 * (quad + logdet + n * np.log(2*np.pi))
    


    def psi_kl(self,m):
        return jnp.exp(m) - 1
    psi = psi_kl

    def get_result(self, g_theta_samples2, sigma_val,psi):
        result = self.solve_saa_jax(g_theta_samples2, sigma_val, psi)

        print(f"Optimal t: {result['optimal_t']:.6f}")
        print(f"Min Value: {result['min_value']:.6f}")
        print(f"Status:    {result['message']}")
        print("-" * 50)

    def exact_posterior_parameters(self, A, data):
        model = BayesianMVLinGaussIP(prior_mean=self.prior_mean, prior_cov = self.prior_cov, dim = 2, noise_cov=self.noise_cov) 
        # TNP: True posterior for comparison to computed one.
        exact_posterior_cov, exact_posterior_mean  = model.posterior_calculations(
            A=A,
            y=data
        )

        return exact_posterior_cov, exact_posterior_mean




    def vectorized_grad_psi(self, psi):
        grad_psi = jax.grad(psi) # TNP: evaluatable function to evaluate the gradient of psi with respect to its argument
        vectorized_grad_psi = jax.vmap(grad_psi)
        return vectorized_grad_psi


    def semianalytic_scalar(self, theta, data, A, noise_cov, result, sigma_val, psi):
        x = (self.scalar_log_likelihood(data, theta, A, noise_cov) - result['optimal_t'])/sigma_val
        return jax.grad(psi)(x)

    def semianalytic_vectorized(self,theta, data, A, noise_cov, result, sigma_val, psi):
        x = (self.vectorized_log_likelihood(data, theta, A, noise_cov) - result['optimal_t']) / sigma_val
        return self.vectorized_grad_psi(psi)(x)

    def prior_density(self, theta):
        return ss.multivariate_normal.pdf(theta, mean=self.prior_mean, cov=self.prior_cov)

   

    def plot_contour_meshgrid(self, theta1, theta2, posterior_dist, title_label):
        T1, T2 = np.meshgrid(theta1, theta2, indexing="ij")
        Z = posterior_dist.pdf(np.stack([T1.ravel(), T2.ravel()], axis=-1))
        Z = Z.reshape(T1.shape)

        plt.figure(figsize=(10, 8))
        contour = plt.contour(T1, T2, Z, levels=8, cmap='viridis')
        plt.colorbar(contour, label=title_label)
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_tricontour_prior(self, prior_samples, g_theta_samples2):
        plt.figure()
        plt.tricontour(prior_samples[:, 0], prior_samples[:, 1], g_theta_samples2)
        plt.title("tricontour plot using prior samples")
        plt.show(block = False)

    def log_prior(self, theta):
        return ss.multivariate_normal.logpdf(theta, mean=self.prior_mean, cov=self.prior_cov)

    def log_posterior(self, theta, data, A, noise_cov, result, sigma_val, psi):
        g_val = self.vectorized_log_likelihood(
            data, theta[None, :], A, noise_cov
        )[0]
        x = (g_val - result["optimal_t"]) / sigma_val
        psi_val = psi(x)

        return self.log_prior(theta) + jnp.log(psi((g_val - result["optimal_t"]) / sigma_val))
    
    def posterior (self, theta, data, A, noise_cov, result, sigma_val, psi):
        return self.prior_density(theta) *self.semianalytic_scalar(theta = theta, data = data, A=A, noise_cov = noise_cov, result = result, sigma_val = sigma_val, psi=psi)