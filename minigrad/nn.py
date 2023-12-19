import random
from minigrad.engine import Value

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for i in self.parameters():
            i.grad = 0


class Neuron(Module):
    def __init__(self, nin, nonlin='relu'):
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