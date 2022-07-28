import math
import numpy as np

def print_2d_array(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            print(array[i][j], end=" ")
        print()


class Interval():
    # constructor
    def __init__(self,*kwargs):
        if (len(kwargs) == 0):
            self.lower = float(0)
            self.upper = float(0)
        elif (len(kwargs) == 1):
            self.lower = float(kwargs[0])
            self.upper = float(kwargs[0])
        elif (len(kwargs) == 2):
            self.lower = float(kwargs[0])
            self.upper = float(kwargs[1])
        else:
            raise ValueError("Invalid constructor parameters: expected 0-2 parameters")
    
    # actually false implementation
    def __add__(self, other: 'Interval'):
        if  not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            math.nextafter(self.lower + other.lower, -math.inf),
            math.nextafter(self.upper + other.upper, math.inf))

    def __iadd__(self, other: 'Interval'):
        return self + other
    
    def __sub__(self, other: 'Interval'):
        if  not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            math.nextafter(self.lower - other.upper, -math.inf),
            math.nextafter(self.upper - other.lower, math.inf))
    
    def __isub__(self, other: 'Interval'):
        return self - other
    
    def __mul__(self, other: 'Interval'):
        if  not isinstance(other, Interval):
            other = Interval(other)
        l = math.nextafter(
            min(self.lower * other.lower, self.upper * other.upper, self.lower * other.upper, self.upper * other.lower)
            , -math.inf)
        u = math.nextafter(
            max(self.lower * other.lower, self.upper * other.upper, self.lower * other.upper, self.upper * other.lower)
            , math.inf)
        return Interval(l, u)

    def __imul__(self, other: 'Interval'):
        return self * other

    def __truediv__(self, other: int):
        if  not isinstance(other, Interval):
            other = Interval(other)
        l = math.nextafter(
            min(self.lower / other.lower, self.upper / other.upper, self.lower / other.upper, self.upper / other.lower)
            , -math.inf)
        u = math.nextafter(
            max(self.lower / other.lower, self.upper / other.upper, self.lower / other.upper, self.upper / other.lower)
            , math.inf)
        return Interval(l, u)
    
    def __itruediv__(self, other: 'Interval'):
        return self / other

    def __str__(self):
        return f"[{self.lower:.4}, {self.upper:.4}]"

    def __eq__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return self.lower == other.lower and self.upper == other.upper
    

# inline test for Interval class, print is necessary to check soundness level
if __name__ == '__main__':
    # add test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a+b: {a+b}")
    a += b
    print(f"a+=b:{a}")

    # minus test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a-b: {a-b}")
    a -= b
    print(f"a-=b:{a}")

    # mul test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a*b: {a*b}")
    a *= b
    print(f"a*=b:{a}")

    # div test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a/b: {a/b}")
    a /= b
    print(f"a/=b:{a}")

    # test vectorization
    a = np.array([[Interval(1,1), Interval(1,1)], [Interval(1,1), Interval(1,1)]], dtype = Interval)
    b = np.array([[Interval(1,1), Interval(1,1)], [Interval(1,1), Interval(1,1)]], dtype = Interval)
    c = a @ b
    print("array a:")
    print_2d_array(a)
    print("array b:")
    print_2d_array(b)
    print("array c = a @ b:")
    print_2d_array(c)

    # test soundness
    a = np.array([[Interval(1,1), Interval(1,1)], [Interval(1,1), Interval(1,1)]]) # no need to specify dtype, python will veiw it as object
    for i in range (45): # propagate error 2^45 times (estimated)
        a = a @ a / 2
    print("array a:")
    print_2d_array(a)

    # other properties
    print("cannot use APIs like ones, zeros as it will change the type")
    a = np.ones((3,4))
    print(a.dtype)
    a = np.ones((3,4)).astype(Interval)
    print(a.dtype) # object class, which means np does not recognize it as Interval