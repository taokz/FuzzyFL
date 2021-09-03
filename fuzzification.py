"""
Implementation of data fuzzification

Author: Kai Zhang (www.kaizhang.us)
https://github.com/taokz
"""

import numpy as np 
from fuzzyset import FuzzySet 
from gaussian_mf import gaussmf 
import math

class FuzzyData(object):

	_data = None
	_fuzzydata = None
	_epistemic_values = None
	_target = None

	def __init__(self, data = None, target = None): 

		if data is not None:
			self._data = data
			self._target = target

	def quantile_fuzzification(self):

		# reference: Guevara et al., Cross product kernels for fuzzy set similarity, 2017 FUZZY-IEEE
		grouped = self._data.groupby([self._target])

		self._epistemic_values = grouped.transform(lambda x:
			np.exp(-np.square(x - x.quantile(0.5))
				/
				(np.abs(x.quantile(0.75) - x.quantile(0.25)) / (
					2 * np.sqrt(2 * np.log(2))) + 0.001) ** 2
				))

		# fill up the NA (which is caused by the equal quantiles)
		# self._epistemic_values = self._epistemic_values.fillna(0)

		# join data and epistemistic values
		num_rows = self._epistemic_values.shape[0]
		num_cols = self._epistemic_values.shape[1]

		self._fuzzydata=np.asarray([[FuzzySet(elements = self._data.iloc[j, i],
			md=self._epistemic_values.iloc[j, i])
						for i in range(num_cols)]
					for j in range(num_rows)])

		# return self._fuzzydata


	def get_fuzzydata(self):
		return self._fuzzydata

	def get_data(self):
		return self._data

	def get_epistemic_values(self):
		return self._epistemic_values

	def get_target(self):
		return self._data[self._target]

	def show_class(self):
		# print all contents of the class
		print("(data)             \n", _data, "\n")
		print("(fuzzydata)        \n", _fuzzydata, "\n")
		print("(epistemic_values) \n", _epistemic_values, "\n")
		print("(target)           \n", _target, "\n")
