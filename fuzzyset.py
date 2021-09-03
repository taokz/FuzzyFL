"""
Implementation of convert from crisp data to fuzzy data

Author: Kai Zhang (www.kaizhang.us)
https://github.com/taokz
"""

class FuzzySet(object):

	_elements = None
	_elements_type = None
	_mf = None
	_md = None
	_params = None

	def __init__(self, elements = None, md = None, mf = None, params = None):
		"""
		initialize a fuzzy set
		
		input:
			elements: (array) elements of the set
			md: (array) membership degrees
			mf: (callable) membership function
			params: (list) function custom parameters


		ouput:
			(object) "FuzzySet"
		"""

		# 1st type: empty fuzzy set
		if elements is None and md is None and mf is None and params is None:
			self._elements = None
			self._elements_type = None
			self._md = md = None
			self._mf = None
			self._params = params = None

		# 2nd type: given elements and membership degrees without membership function
		# if elements is not None:
		# 	self._elements = elements
		# 	if isinstance(self._elements, (float, int)):
		# 		self._elements_type = type(elements)
		# 	else:
		# 		self._elements_type = type(elements[0])

		if elements is not None and mf is None:
			self._elements = elements
			self._md = md

		# 3rd type: given elements and membership function, then the membership degress are estimated
		if elements is not None and md is None:
			self._params = params
			self._mf = mf
			self._md = self.mf(self._elements, *self._params)

	def set_md(self, md):
		# set membership degree
		self._md = md

	def get_set(self):
		# return the set
		return self._elements

	def get_function(self):
		# return the membership function
		return self._mf

	def get_pair(self):
		# return the pair (elements, md) elements and membership degree
		if  isinstance(self._elements,  (float, int)) and isinstance(self._md,  (float, int)):
			return list(zip(list([self._elements]),list([self._md])))
		else:
			return list(zip(self._elements,self._md))

	def get_degree(self):
		# return the membership degree
		return self._md

	def show_class(self):
		# print all contents of the class
		print("(elements)          \n", self._elements, "\n")
		print("(elements_type)     \n", self._elements_type, "\n")
		print("(mf)                \n", self._mf, "\n")
		print("(md)                \n", self._md, "\n")
		print("(params)            \n", self._params, "\n")
