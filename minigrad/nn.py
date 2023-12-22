import random
from typing import List
from minigrad.engine import Value

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, nin: int, nonlin: str='relu'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        out = sum(wi*xi for wi, xi in zip(self.w,x)) + self.b
        return getattr(out, self.nonlin)() if hasattr(out, self.nonlin) else out

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.nonlin if hasattr(self.b, self.nonlin) else 'Linear'} Neuron({len(self.w)})"
    

class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int], **kwargs):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i+1], nonlin='linear' if i==len(nouts)-1 else 'relu') for i in range(len(nouts))]

        # self.layers = [Layer(sz1, sz2, **kwargs) for sz1, sz2 in zip(size[:-1], size[1:])]
        # self.layers[-1].nonlin = 'linear'
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"