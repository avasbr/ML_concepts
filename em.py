import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def em_gmm(X,K=3,eps=1e-2,max_iter=100):
	
	"""Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMM)

	Parameters
	----------
	X : ndarray,required
		N x d array of feature vectors
	K : int, optional
		number of clusters
	eps : float, optional
		  termination tolerance
	max_iter : int, optional
			   number of maximum iterations

	Returns
	-------
	gamma : ndarray
			N x K array of probabilities, where each entry (i,j) denotes the
			probability that point x_i (in row i of X) belonging to cluster j
	exp_loglik : list
			   	 expected complete-data log-likelihood at each iteration
	"""

	# Initialize all the parameters we're interested in estimating. For the 
	# mixture of gaussians, it's the mixture weights, means, and covariance
	# matrices.
	shp = np.shape(X)
	if len(shp) == 1:
		d = 1
	else:
		d = shp[1]
	N = shp[0]

	iter_num = 0
	
	w_hat = 1./K*np.ones(K) # start with uniform mixture weights
	idx = np.random.permutation(N)[:K]
	mu_hat = X[idx] # start means at random data points
	scov = compute_weighted_samp_cov(X,N)
	cov_hat = K*[scov] # start cov matrices with sample covariance
	exp_loglik = []

	# compute the expectation of the complete-data log-likelihood
	gamma,curr_loglik = compute_complete_data_log_likelihood(X,N,d,K,w_hat,mu_hat,cov_hat)
	exp_loglik.append(curr_loglik)
	err = np.inf

	update_plot(X,iter_num,K,mu_hat,cov_hat,err)
	
	while(err > eps and iter_num < max_iter):
		iter_num += 1
		for k in range(K):
			gamma_sum = np.sum(gamma[:,k])
			
			# update parameters which maximize the expected log-likelihood
			w_hat[k] = 1./N*gamma_sum
			mu_hat[k] = np.sum(gamma[:,k][:,np.newaxis]*X,axis=0)/gamma_sum
			cov_hat[k] = compute_weighted_samp_cov(X,N,mu_hat[k],gamma[:,k])

		# compute new expected log-likelihood
		gamma,curr_loglik = compute_complete_data_log_likelihood(X,N,d,K,w_hat,mu_hat,cov_hat)
		
		# compute the new error to check for termination
		err = abs(exp_loglik[-1] - curr_loglik)
		exp_loglik.append(curr_loglik)

		update_plot(X,iter_num,K,mu_hat,cov_hat,err)

	return w_hat,mu_hat,gamma,exp_loglik

def update_plot(X,iter_num, K,mu_hat,cov_hat,err):
	plt.cla()
	plt.scatter(X[:,0],X[:,1],color='blue')
	for k in range(K):
		plt.scatter(mu_hat[k][0],mu_hat[k][1],color='red')
		draw_ellipse(mu_hat[k],cov_hat[k]) # draw the 
	plt.title("Iteration: %s, Expected Log-Likelihood Error: %s"%(iter_num,err))
	plt.draw()

def draw_ellipse(mu,cov,num_std=2,ax=None):

    if ax is None:
        ax = plt.gca()
	
    # eigenvalue decomposition
    eig_val, eig_vec = np.linalg.eigh(cov)
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[idx]

    theta = np.degrees(np.arctan2(*eig_vec[:,0][::-1]))
    width, height = 2*num_std* np.sqrt(eig_val)
    ellip = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False,color='red')

    ax.add_artist(ellip)
    return ellip


def compute_complete_data_log_likelihood(X,N,d,K,w_hat,mu_hat,cov_hat):
	# gamma_i,k = probability of cluster assignment
	gamma = np.empty([N,K])
	for k in range(K):
		gamma[:,k] = w_hat[k]*compute_gaussian(X,d,mu_hat[k],cov_hat[k])
	
	gamma = gamma/np.sum(gamma,axis=1)[:,np.newaxis]
	exp_loglik = 0

	# expected log-likelihood computation
	for k in range(K):
		exp_loglik += np.sum(gamma[:,k][:,np.newaxis]*(np.log(w_hat[k]) + compute_log_gaussian(X,d,mu_hat[k],cov_hat[k])))

	return gamma,exp_loglik

def compute_gaussian(X,d,mu,cov):
	Z = ((2*np.pi)**d*np.linalg.det(cov))**-0.5
	invcov = np.linalg.inv(cov)
	return np.array([Z*np.exp(-0.5*np.dot((x-mu),invcov).dot(x-mu)) for x in X])	

def compute_log_gaussian(X,d,mu,cov):
	invcov = np.linalg.inv(cov)
	return np.array([-0.5*np.dot((x-mu),invcov).dot(x-mu) - d/2*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov)) for x in X])

def compute_weighted_samp_cov(X,N,mu=None,wts=None):
	if mu is None:
		mu = np.mean(X,axis=0) 
	if wts is None:
		wts = np.ones(N)
	
	Z = np.sum(wts)

	return 1./Z*sum([w*np.outer(x-mu,x-mu) for w,x in zip(wts,X)]) # weighted sample covariance matrix

def generate_data(N=1000,d=2,num_clusts=3,mean_range=(-5,5)):
		
	# generate the data randomly
	x = np.random.random() 
	Q = np.array([[np.sin(x),np.cos(x)],[-1.0*np.cos(x),np.sin(x)]]) # create a random unitary matrix
	mu = np.random.random_integers(mean_range[0],mean_range[1],size=(num_clusts,d)) # generate random mean values
	cov = [] # covariance matrix
	for i in range(num_clusts):
		a = np.diag([np.random.random(),np.random.random()]) # create a random diagonal matrix (eig_vals > 0)
		cov.append(np.dot(Q,a).dot(Q.T)) # generate a random PSD matrix (Q*D*Q.T)
	w = np.random.rand(num_clusts) # mixture weights
	w = w/np.sum(w) # normalize to sum to 1

	X = np.empty([N,d])

	start_idx = 0
	for k,nk in enumerate(np.random.multinomial(N,w)):
		X[start_idx:start_idx+nk,:] = np.random.multivariate_normal(mu[k],cov[k],int(nk))
		start_idx += nk

	return X

def main():
	plt.ion() # turn on interactive plotting
	
	# generate toy data
	N = 1000
	X = generate_data(N)
	
	K = 3 # number of assumed clusters
	w_hat,mu_hat,gamma,exp_loglik = em_gmm(X,K) # apply the EM algorithm
	plt.cla()
	
	# hard-assignment to cluster
	colors = ['red','green','blue']
	for k in range(K):
		idx = [i for i in range(N) if np.argmax(gamma[i])==k]
		plt.scatter(X[idx,0],X[idx,1],color=colors[k])
	
	plt.title("Final Clusters")
	plt.show()

if __name__=='__main__':
	main()