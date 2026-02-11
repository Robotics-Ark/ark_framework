import numpy as np

class Value:
    def __init__(self, val, parents=(), backward=None, name=None):
        self.val = np.asarray(val, dtype=float)
        self.grad = np.zeros_like(self.val)
        self._prev = parents
        self._backward = backward or (lambda: None)
        self.name = name
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.val)
        self.grad = self.grad + grad
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        for v in reversed(topo):
            v._backward()
    def __add__(self, other):
        out = Value(self.val + other.val, parents=(self, other))
        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad
        out._backward = _backward
        return out
    def __sub__(self, other):
        out = Value(self.val - other.val, parents=(self, other))
        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad - out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        out = Value(self.val * other.val, parents=(self, other))
        def _backward():
            self.grad = self.grad + other.val * out.grad
            other.grad = other.grad + self.val * out.grad
        out._backward = _backward
        return out
    def __neg__(self):
        out = Value(-self.val, parents=(self,))
        def _backward():
            self.grad = self.grad - out.grad
        out._backward = _backward
        return out
def clear_grads(params):
    for p in params:
        p.grad = np.zeros_like(p.val)
