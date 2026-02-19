import torch
from ark_msgs import Value


class Variable:

    def __init__(self, name, value, mode, variables_registry, lock, clock, create_queryable_fn):
        self.name = name
        self.mode = mode
        self._variables_registry = variables_registry
        self._lock = lock
        self._clock = clock
        self._grads = {}  # input vars: {output_name: grad_value}

        if mode == "input":
            self._tensor = torch.tensor(value, requires_grad=True)
            self._history = {}
            self._replay_tensor = None
        else:
            self._tensor = None
            self._computation_ts = clock.now()
            self._replay_fn = None
            for inp_name, inp_var in variables_registry.items():
                if inp_var.mode == "input":
                    grad_channel = f"grad/{inp_name}/{name}"

                    def _make_handler(iv, ov_name, reg, lk):
                        def handler(_req):
                            out_var = reg.get(ov_name)
                            if _req.timestamp != 0 and out_var._replay_fn:
                                val, grad = out_var._replay_fn(_req.timestamp, iv.name, ov_name)
                                return Value(val=val, grad=grad, timestamp=_req.timestamp)
                            with lk:
                                val = float(out_var._tensor.detach()) if out_var and out_var._tensor is not None else 0.0
                                grad = iv._grads.get(ov_name, 0.0)
                                ts = out_var._computation_ts if out_var else 0
                            return Value(val=val, grad=grad, timestamp=ts)
                        return handler

                    create_queryable_fn(grad_channel, _make_handler(inp_var, name, variables_registry, self._lock))

    def snapshot(self, ts):
        """Record current tensor value at clock timestamp ts."""
        self._history[ts] = float(self._tensor.detach())

    def at(self, ts):
        """Return a fresh requires_grad tensor from history at ts."""
        val = self._history[ts]
        self._replay_tensor = torch.tensor(val, requires_grad=True)
        return self._replay_tensor

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        if self.mode == "output":
            self._tensor = value
            self._compute_and_store_grads()
        else:
            self._tensor.data = value.data if isinstance(value, torch.Tensor) else torch.tensor(value)

    def _is_last_output(self):
        output_names = [k for k, v in self._variables_registry.items() if v.mode == "output"]
        return output_names and output_names[-1] == self.name

    def _compute_and_store_grads(self):
        if self._tensor is None or not self._tensor.requires_grad:
            return
        with self._lock:
            for var in self._variables_registry.values():
                if var.mode == "input" and var._tensor.grad is not None:
                    var._tensor.grad.zero_()
            self._tensor.backward(retain_graph=not self._is_last_output())
            for var in self._variables_registry.values():
                if var.mode == "input":
                    grad = float(var._tensor.grad) if var._tensor.grad is not None else 0.0
                    var._grads[self.name] = grad
            self._computation_ts = self._clock.now()
