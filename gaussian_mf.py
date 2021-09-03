"""
Implementation of gaussian membership function

Author: Kai Zhang (www.kaizhang.us)
https://github.com/taokz
"""

import numpy as np 

def gaussmf(elements, mean, sigma):
	"""
	input:
		elements: (array) elements of the set
		mean: (real) mean of the set
		sigma (real) standart deviation of the set, or 
			  (array) covariance matrix
	"""

	if isinstance(sigma, np.ndarray):
		sigma = np.linalg.inv(sigma)
		values = np.einsum('ij,ij->i', np.dot((elements-mean), sigma), (elements-mean))
		values = np.exp(-values)

	if isinstance(sigma, (float, int)):
		values = np.exp(-np.square(elements - mean) / sigma**2)

	return values
