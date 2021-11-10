from CTL.funcs.funcs import randomString
import CTL.funcs.funcs as funcs

class StringSet:
    """
    A set of unique strings.

    Parameters
    ----------
    n : int, default 10
        The length of strings in this set.
    
    Attributes
    ----------
    stringSet : set of str
        The current set of strings.
    n : int
        The length of new generated strings.
    """

    def __init__(self, n = 10):
        self.stringSet = set([])
        self.n = n

    def newString(self, inputStr = None):
        """
        Parameters
        ----------
        inputStr : str, optional
            The string that is preferred to be added. If None, then generate a new string.

        Returns
        -------
        str
            A new string generated, or the input string if it is not None.

        Raises
        ------
        AssertionError
            Raised when inputStr is not None, but the string has already appeared in this set.
        """
        if (inputStr is None):
            res = randomString(self.n)
            while (res in self.stringSet):
                res = randomString(self.n)
        else:
            assert not (inputStr in self.stringSet), funcs.errorMessage(err = "Error: name '{}' is already used.".format(inputStr), location = "StringSet.newString")
            res = inputStr

        self.stringSet.add(res)
        return res

    def add(self, s):
        """
        Add a new string to current set.

        Parameters
        ----------
        s : str
            The new string to be added.
        
        Raises
        ------
        AssertionError
            Raised if s is already in this set.
        """
        assert not (s in self.stringSet), funcs.errorMessage(err = "Error: name '{}' is already used.".format(s), location = "StringSet.add")
        self.stringSet.add(s)
        self.n += 1

    def remove(self, s):
        """
        Remove a string to current set.

        Parameters
        ----------
        s : str
            The string to be removed.
        
        Raises
        ------
        AssertionError
            Raised if s is not in this set.
        """
        assert (s in self.stringSet), funcs.errorMessage(err = "Error: name '{}' is not used.".format(s), location = "StringSet.remove")
        self.stringSet.remove(s)
        self.n -= 1

    def contains(self, s):
        """
        Check if the set contains the given string.

        Parameters
        ----------
        s : str

        Returns
        -------
        bool
            Whether stringSet contains s.
        """
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