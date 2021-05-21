class TensorDict:

    def __init__(self, tensorDict = None):
        # print(tensorDict)
        if (tensorDict is None):
            self.tensors = dict()
        else:
            self.tensors = tensorDict
        # print(self.tensors.keys())

    @property 
    def tensorCount(self):
        return len(self.tensors)

    def setTensor(self, name, tensor):
        self.tensors[name] = tensor
    def getTensor(self, name):
        return self.tensors[name]
    def deleteTensor(self, name):
        assert (name in self.tensors), "Error: tensor {} does not exist.".format(name)
        del self.tensors[name]