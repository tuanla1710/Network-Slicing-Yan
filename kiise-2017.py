"""Import"""
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import random as rand

np.random.seed(10)
D = 5  # Number of D2D
K = 1  # Number of WiFi APs
N_K = 10  # Number of WiFi users
T = 100  # Number of iterations
t_d2d_k = 0.6
b_k = 1
noise = 10**(-11)

# Matrix of the D2D fraction of time allocation matrix following T
BETA = np.zeros(shape=(D, T))
BETA[:, [0]] = 0.01*np.ones(shape=(D, 1))
Lambda = np.zeros(shape=(1, T))
Lambda[0][0] = 0.5
Gain = np.array([10**(-4), 15**(-4), 20**(-4), 3**(-4), 4**(-4)])
Power = 0.1  # Wats


def return_r_d_k(t_d2d_k, beta_d_k, h_d_k, p_d_k, b_k, delta_0):
	# solving problem: y = t_d2d_k*beta_d_k*B_k*log2(1+h_d_k*P_d_k/(t_d2d_k*beta_d_k*B_k*delta+0))
	r_d_k = t_d2d_k*beta_d_k*b_k*np.log2(1+h_d_k*p_d_k/(t_d2d_k*b_k*delta_0))
	return r_d_k


def update_beta_d_k(beta_d_k, t_d2d_k, b_k, h_d_k, p_d_k, delta_0, lambda_k):
	# print("(beta_d_k, t_d2d_k, b_k, h_d_k, p_d_k, delta_0, lambda_k) = ", beta_d_k, t_d2d_k, b_k, h_d_k, p_d_k, delta_0, lambda_k)

	gamma_d_k = h_d_k*p_d_k/delta_0

	print("t_d2d_k*b_k*np.log2(1 + (gamma_d_k/(t_d2d_k*b_k))/beta_d_k) - (t_d2d_k*b_k*gamma_d_k*beta_d_k)/(t_d2d_k*b_k*beta_d_k + gamma_d_k)", t_d2d_k*b_k*np.log2(1 + (gamma_d_k/(t_d2d_k*b_k))/beta_d_k) - (t_d2d_k*b_k*gamma_d_k*beta_d_k)/(t_d2d_k*b_k*beta_d_k + gamma_d_k))
	print('np.log2(1 + (gamma_d_k/(t_d2d_k*b_k))/beta_d_k)', np.log2(1 + (gamma_d_k/(t_d2d_k*b_k))/beta_d_k))
	print('(t_d2d_k*b_k*gamma_d_k*beta_d_k)/(t_d2d_k*b_k*beta_d_k + gamma_d_k)', (t_d2d_k*b_k*gamma_d_k*beta_d_k)/(t_d2d_k*b_k*beta_d_k + gamma_d_k))

	f = t_d2d_k*b_k*np.log2(1 + (gamma_d_k/(t_d2d_k*b_k))/beta_d_k) - (t_d2d_k*b_k*gamma_d_k*beta_d_k)/(t_d2d_k*b_k*beta_d_k + gamma_d_k) - lambda_k
	# print("f =", f)

	f_p_1 = (t_d2d_k*b_k*gamma_d_k/(t_d2d_k*beta_d_k*b_k + gamma_d_k))
	f_p_2 = (-t_d2d_k*b_k-gamma_d_k/(t_d2d_k*beta_d_k*b_k + gamma_d_k))
	f_p = f_p_1 * f_p_2
	# print("f_p =", f_p)

	beta_k_d = beta_d_k - f/f_p
	return beta_k_d


def update_lambda_k(lambda_k, sum_beta_d_k, delta_k):
	f = lambda_k + delta_k*(sum_beta_d_k - 1)
	lambda_k = max(0, f)
	return lambda_k

# Testing
# print(max(0, 1))
# print(M)
print(Gain)
print(BETA[:, [0]])
print(Lambda[[0], [0]])


# main algorithm
for t in range(1, T):
	# At each iteration t do
	# Step 1: Update each beta_d_k
	for d in range(D):
		print('i,d=', t, d)
		beta_d_k = BETA[d, [t-1]]
		beta_d_k = beta_d_k[0]
		t_d2d_k = t_d2d_k
		b_k = b_k
		# print(np.size(Gain))
		h_d_k = Gain[d]
		p_d_k = Power
		delta_0 = noise
		lambda_k = Lambda[0][t-1]
		beta_d_k_t = update_beta_d_k(beta_d_k, t_d2d_k, b_k, h_d_k, p_d_k, delta_0, lambda_k)
		# Update to the matrix
		BETA[d, [t]] = beta_d_k_t

	# Step 2: Update each lambda
		lambda_k = Lambda[0][t-1]
		sum_beta_d_k = np.sum(BETA[:, [t-1]])
		# print('sum_beta_d_k=',sum_beta_d_k)
		# print('lambda_k = ', lambda_k)
		delta_k = 0.02
		print('sum_beta_d_k = ', sum_beta_d_k)
		lambda_k_t = update_lambda_k(lambda_k, sum_beta_d_k, delta_k)
		# Update Lagrangian Multipliers
		Lambda[0][t] = lambda_k_t
	print('t,Lambda[0][t] = ', t, Lambda[0][t])

# Plot results:
print('Lambda =', Lambda)
print(np.size(T), np.size(Lambda))
plt.plot(range(0, T), Lambda[0])
plt.xlabel('Time t')
plt.ylabel('Lambda value')
plt.show()

