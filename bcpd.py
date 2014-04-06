# Simple application of BCPD assuming a gaussian distribution with unknown mean and known variance

import numpy as np
import random as random
import matplotlib.pyplot as plt

def bcpd(n_pts=1000,p_cp=0.004,mu=4,var_mu=3,var=0.05,lmbda=200):
	
	"""Bayesian change-point detection (BCPD) using a gaussian likelihood with known variance
	and unknown mean. The data is generated synthetically using the provided parameters

	Parameters
	----------
	n_pts :	int, optional
			number of points to generate
	p_cp : float, optional
		   probability of generating a change-point
	mu : float, optional
	     mean prior
	var_mu : float, optional
	         variance prior
	var : float, optional
	      observation noise
	lmbda : float, optional 
	        hazard function parameter
	"""

	# generate synthetic data
	data = generate_data(n_pts,p_cp,mu,var,var_mu)

	# prir parameters for unknown mean parameter
	mu0 = 2.0
	var0 = 2.0

	mu_t = [mu0]
	var_t = [var0]

	# hazard function p(r_t|r_(t-1)) - in this case, assume constant
	h = 1.0/lmbda # coin-flip

	r_t = [np.array([1.0])] # this is really a phantom point

	maxs = [0] # holds the map estimate of the run-lengths
	
	# needed for plotting
	x = []
	y = []

	for t,x_t in enumerate(data):

		x.append(t)
		y.append(x_t)

		# predictive probability p(x_t|u_t,var_t+var)
		pred_prob = evaluate_gaussian(x_t,mu_t,[v+var for v in var_t])
		
		# growth probability p(r_t = r_t-1 | r_t-1)
		r_g = pred_prob*r_t[t]*(1-h)

		# change-point probability p(r_t = 0 | r_t-1) - sum aross all values of r_t-1
		r_c = sum(pred_prob*r_t[t]*h)

		# p(r_t|x_1:t)
		r_t.append(np.insert(r_g,0,r_c)/(sum(r_g)+r_c)) # normalized to get conditional probability

		# update model parameters
		var_new = [(1/var + 1/v)**-1 for v in var_t]
		mu_new = [v_new*(mu/v + x_t/var) for mu,v,v_new in zip(mu_t,var_t,var_new)]
		mu_t = [mu0] + mu_new
		var_t = [var0] + var_new

		maxs.append(np.argmax(r_t[t+1])) # if the max is at idx 0, we have a change point!

	# generate plots

	plt.figure(1)
	plt.subplot(211)
	plt.title("Synthetically generated stream of data")
	plt.xlabel("Time")
	plt.ylabel("Data point (x_t)")
	plt.plot(x,y)
	
	plt.subplot(212)
	plt.plot(x,maxs[1:])
	plt.title("MAP estimate of run-length")
	plt.xlabel("Time")
	plt.ylabel("Run-length (r_t)")
	plt.show()
	
def evaluate_gaussian(x,mu,var):
	"""1D gaussian evaluated at x for an array of mu/sigma values"""
	return np.array([1.0/(np.sqrt(2*np.pi*v))*np.exp(-0.5*((x-m)**2/v)) for m,v in zip(mu,var)])
	
def generate_data(n_pts,p_cp,mu,var,var_mu):
	""" Generator function that generats n_pts number of points, arbitrarily
	adding in change-points along the way """

	for pt in range(n_pts):
		if random.random() < p_cp:
			mu = np.random.normal(mu,np.sqrt(var_mu)) # regime change
		yield np.random.normal(mu,np.sqrt(var)) # new point

def main():
	bcpd()

if __name__=='__main__':
	main()