import unittest
import numpy as np
import quadprog as sys_quadprog
from my_quadprog import solve_qp
import matplotlib.pyplot as plt
import time
# from scipy.optimize import check_grad


class QuadProgTestCase(unittest.TestCase):
	@staticmethod
	def create_problem(n, n_eq, n_ineq):
		"""
		Generates a random feasible quadratic program.

		:param n: number of variables
		:type n: int
		:param n_eq: number of equality constraints
		:type n_eq: int
		:param n_ineq: number of inequality constraints
		:type n_ineq: int
		:return:
		:rtype:
		"""
		mat = np.random.randn(n, n)
		hessian = mat.T.dot(mat)

		weights = np.random.randn(n)

		x = np.random.randn(n)  # guaranteed feasible point

		coeffs = np.random.randn(n, n_eq + n_ineq)
		constants = coeffs.T.dot(x)

		return hessian, weights, coeffs, constants, n_eq, x

	def test_problem(self):
		np.random.seed(0)
		hessian, weights, coeffs, constants, n_eq, x = self.create_problem(100, 10, 10)

		w, v = np.linalg.eig(hessian)

		print("Norm of Hessian: %e" % np.linalg.norm(hessian))

		assert min(w) > 0, "Hessian wasn't PSD"

		assert np.allclose(coeffs.T.dot(x), constants), "Constraints were not set up right."

	def test_compare_against_quadprog(self):
		np.random.seed(0)
		hessian, weights, coeffs, constants, n_eq, x = self.create_problem(100, 20, 20)

		sol = sys_quadprog.solve_qp(hessian, weights, coeffs, constants, n_eq)

		my_sol = solve_qp(hessian, weights, coeffs, constants, n_eq)

		print("Returned objective values (quadprog, my_qp): (%e, %e)" % (sol[1], my_sol[1]))
		print("Norm of solution difference: %f" % np.linalg.norm(sol[0] - my_sol[0]))

		assert np.allclose(sol[0], my_sol[0], rtol=1e-3, atol=1e-3), "Solutions were too different"

	def test_constraint_satisfaction(self):
		np.random.seed(0)
		n_eq = 10
		n_ineq = 10
		hessian, weights, coeffs, constants, n_eq, x = self.create_problem(100, n_eq, n_ineq)

		sol = sys_quadprog.solve_qp(hessian, weights, coeffs, constants, n_eq)

		my_sol = solve_qp(hessian, weights, coeffs, constants, n_eq)

		violation = coeffs.T.dot(sol[0]) - constants

		print("Quadprog equality violations: ", violation[:n_eq])
		print("Quadprog inquality violations: ", violation[n_eq:])

		assert np.linalg.norm(violation[:n_eq]) <= 1e-4, "Equality constraints were not satisfied by quadprog"
		assert np.min(violation[n_eq:]) > -1e-4, "Inequality constraints were not satisfied by quadprog"

		violation = coeffs.T.dot(my_sol[0]) - constants

		print("My quadprog equality violations: ", violation[:n_eq])
		print("Norm: %e" % np.linalg.norm(violation[:n_eq]))
		print("My quadprog inquality violations: ", violation[n_eq:])

		assert np.linalg.norm(violation[:n_eq]) <= 1e-4, "Equality constraints were not satisfied by my_quadprog"
		assert np.min(violation[n_eq:]) > -1e-4, "Inequality constraints were not satisfied by my_quadprog"

	def test_speed(self):
		np.random.seed(0)
		sizes = [8, 16, 32, 64, 128, 256, 512]
		num_trials = 5

		qp_time = []
		my_qp_time = []

		for n in sizes:
			trial_qp_time = 0
			trial_my_time = 0
			for _ in range(num_trials):
				hessian, weights, coeffs, constants, n_eq, x = self.create_problem(n, int(n/2), 0*int(n/2))

				start_time = time.time()
				sol = sys_quadprog.solve_qp(hessian, weights, coeffs, constants, n_eq)
				trial_qp_time += time.time() - start_time

				start_time = time.time()
				my_sol = solve_qp(hessian, weights, coeffs, constants, n_eq)
				trial_my_time += time.time() - start_time

				print("Norm of solution difference: %f" % np.linalg.norm(sol[0] - my_sol[0]))
				assert np.allclose(sol[0], my_sol[0], 1e-3, 1e-3), "Solutions were not close"

			qp_time += [trial_qp_time / num_trials]
			my_qp_time += [trial_my_time / num_trials]

			print("Finished size %d QPs" % n)

		plt.plot(sizes, qp_time, label="Quadprog")
		plt.plot(sizes, my_qp_time, label="My Quadprog")
		plt.xlabel('Num variables')
		plt.ylabel('Running time')
		plt.legend()
		plt.show()


if __name__ == '__main__':
	unittest.main()
