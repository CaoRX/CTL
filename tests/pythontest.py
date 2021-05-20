import numpy as np 
class Base:
    def __init__(self):
        self.x = 1

class Child(Base):

    def __init__(self):
        super().__init__()
        self.y = self.x 
        print(self.y)
        print(self.x)

if __name__ == '__main__':
    c = Child()

    x = np.zeros((3, 4, 5))
    y = np.moveaxis(x, 0, -1)
    
    y[1][1][1] = 1
    print(x.shape)
    print(x)