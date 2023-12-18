class Value:
    
    ## TODO Add data types (half, fp16, int8, etc.)
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__ (self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    def __pow__ (self, other): # self ** other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
        return out
    
    def relu (self): # self.relu()
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        return out
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other ** -1 
    
    def __rtruediv__(self, other): # other / self
        return other * self ** -1
    
    def __rpow__(self, other): # other ** self
        return Value(other) ** self

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

