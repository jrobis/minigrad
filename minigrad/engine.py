class Value:
    
    ## TODO Add data types (half, fp16, int8, etc.)
    ## TODO Write test cases
    
    def __init__(self, data):
        self.data = data
        self.grad = 0

    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        return out
    
    def __mul__ (self, other): ## self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        return out
    
    def __pow__ (self, other): ## self ** other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data)
        return out
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __sub__(self, other): # self - other
        return self.data + (-other)
    
    def __truediv__(self, other): # self / other
        return self * other ** -1 
    
    def __rtruediv__(self, other): # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

