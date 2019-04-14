"""Drop-in replacement for quadprog package."""

import matplotlib.pyplot as plt
import numpy as np


def solve_qp(hessian, weights, coeffs, constants, num_eq, rho=0.1, max_iter=10000, tol_x=1e-8, verbose=False):
	"""
	Solves a quadratic program
	min    0.5 * x.T.dot(hessian).dot(x) - weights.dot(x)
	s.t.   coeffs.T.dot(x) = constants, for first num_eq entries
	then   coeffs.T.dot(x) >= constants

	Solves using a dual gradient descent on the augmented Lagrangian with a
	primal-dual gradient ascent-descent, with adagrad for descent

	:param hessian: n by n matrix
	:type hessian: ndarray of shape (n, n)
	:param weights: length-n vector
	:type weights: ndarray of shape (n,)
	:param coeffs: constraint coefficients
	:type coeffs: ndarray of shape (n, num_constraints)
	:param constants: contraint constants
	:type constants: ndarray of shape (num_constraints,)
	:param num_eq: number of equality constraints
	:type num_eq: int
	:param rho: augmented Lagrangian scalar (step size)
	:type rho: float
	:param max_iter: maximum iterations of ascent-descent
	:type max_iter: int
	:param tol_x: convergence tolerance for change in primal and dual variables. Exits if
	change is less than tol_x
	:type tol_x: float
	:param verbose: flag to print diagnostic information and plot objective
	:type verbose: boolean
	:return: tuple containing (0) the solution, (1) the objective value, and (2) the Lagrange variables
	:rtype: tuple of length 3
	"""
	num_constraints = constants.size

	x = np.zeros(weights.size)
	gamma = np.zeros(num_constraints)

	grad_sum = 1

	if verbose:
		objectives = []

	for t in range(max_iter):
		# save old parameters
		old_x = x.copy()
		old_gamma = gamma.copy()

		objective = 0.5 * x.dot(hessian).dot(x) - weights.dot(x)
		violations = coeffs.T.dot(x) - constants
		clipped_violations = violations.copy()
		clipped_violations[num_eq:] = np.clip(violations[num_eq:], a_min=None, a_max=0)

		linear_lagrange = gamma.dot(violations)
		augmented_term = 0.5 * rho * clipped_violations.dot(clipped_violations)

		if verbose:
			augmented_lagrangian = objective + linear_lagrange + augmented_term

		violations = clipped_violations != 0

		# update x

		grad_x = hessian.dot(x) - weights + gamma.dot(coeffs.T) \
			+ rho * coeffs[:, violations].dot(coeffs[:, violations].T).dot(x) \
			- rho * constants[violations].dot(coeffs[:, violations].T)

		rate = 1

		grad_sum += grad_x ** 2

		x -= rate * grad_x / np.sqrt(grad_sum)

		# update gamma
		violations = coeffs.T.dot(x) - constants
		gamma += rho * violations
		gamma[num_eq:] = np.clip(gamma[num_eq:], a_min=None, a_max=0)

		if verbose:
			objectives.append(augmented_lagrangian)
			print("t = %d, f(x) = %e, ||x|| = %e, ||df/dx|| = %e, violation = %e" %
				  (t, augmented_lagrangian, np.linalg.norm(x), np.linalg.norm(grad_x), np.linalg.norm(clipped_violations)))

		if np.linalg.norm(gamma - old_gamma) + np.linalg.norm(x - old_x) < tol_x:
			break

	if verbose:
		plt.plot(objectives)
		plt.show()

	return x, objective, gamma
