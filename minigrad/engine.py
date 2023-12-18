class Value:
    
    ## TODO Add data types (half, fp16, int8, etc.)
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # Graphviz variables
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other): # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__ (self, other): # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__ (self, other): # self ** other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')

        def _backward():
            self.grad += (other.data * self.data ** (other.data-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu (self): # self.relu()
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data>0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def get_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    get_topo(child)
                topo.append(node)
        get_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()

        
    
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

