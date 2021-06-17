from CTL.funcs.funcs import randomString

class StringSet:

	def __init__(self, n = 10):
		self.stringSet = set([])
		self.n = n

	def newString(self, inputStr = None):
		if (inputStr is None):
			res = randomString(self.n)
			while (res in self.stringSet):
				res = randomString(self.n)
		else:
			assert not (inputStr in self.stringSet), "Error: name '{}' is already used.".format(inputStr)
			res = inputStr

		self.stringSet.add(res)
		return res

	def add(self, s):
		self.stringSet.add(s)
		self.n += 1

	def remove(self, s):
		self.stringSet.remove(s)
		self.n -= 1

	def contains(self, s):
		return (s in self.stringSet)

MLStringSet = StringSet()
getMLBondName = MLStringSet.newString

TensorStringSet = StringSet()
getTensorName = TensorStringSet.newString

BondStringSet = StringSet()
getBondName = BondStringSet.newString

# LegStringSet = StringSet()
# getLegName = LegStringSet.newString

# used for makeLink