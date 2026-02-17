# Gradient Experiment

Demonstrates differentiable simulation using ark framework with distributed parameter publishing. A `ParamPublisherNode` publishes parameter values (`v`, `m`, `c`), a `LineVariableNode` subscribes to those parameters, computes position on a line (`y = m*x + c`, `x = v*t`) with autograd gradients, and an `AutodiffPlotterNode` subscribes to position and queries gradients in real time.

## Architecture

```
ParamPublisherNode          LineVariableNode              AutodiffPlotterNode
 publishes:                  subscribes to:                subscribes to:
   param/v (Value)   ──►      param/v, param/m, param/c     position
   param/m (Value)           computes:                     queries:
   param/c (Value)             x = v*t, y = m*x + c          grad/v/x, grad/m/y
                             publishes:                    plots:
                               position (Translation)        trajectory + gradients
                             serves queryables:               vs sim time
                               grad/{v,m,c}/{x,y}
```

## Key Concepts

- **`create_variable(name, value, mode="input", fields=...)`** on `BaseNode`:
  - Creates a `torch.tensor` with `requires_grad=True`
  - Auto-subscribes on `param/{name}` to receive values from other nodes
  - Auto-creates gradient queryables at `grad/{name}/{field}` for each field
- **`update_variable(name, grad_dict)`**: Caches gradients after `backward()`, served by queryables

## Prerequisites

- Install ark framework and dependencies (`zenoh`, `torch`, `matplotlib`, `ark_msgs`)
- Run all commands from the `test/` directory

## Running the Experiment

Open four separate terminals. All commands are run from the `test/` directory.

### Shell 1 — Sim Clock

Drives simulated time for all sim-enabled nodes.

```bash
cd test
python simstep.py
```

### Shell 2 — Parameter Publisher

Publishes fixed parameter values: `v=1.0`, `m=0.5`, `c=0.0`.

```bash
cd test
python param_publisher.py
```

### Shell 3 — Diff Variable Publisher

Subscribes to parameters, computes position and gradients, publishes position, serves gradient queryables.

```bash
cd test
python diff_variable_pub.py
```

### Shell 4 — Autodiff Plotter

Subscribes to position and queries gradients, then plots both against simulation time.

```bash
cd test
python ad_plotter_sub.py
```

## What to Expect

- **Shell 1** prints real elapsed time and sim time advancing each tick.
- **Shell 2** publishes parameter values at 10Hz (no output by default).
- **Shell 3** prints computed gradients (`dx/dv`, `dy/dm`) each step.
- **Shell 4** opens a matplotlib window with two plots:
  - **Left**: Position trajectory (x vs y), autoscaling.
  - **Right**: Gradients vs simulation time (dx/dv in green, dy/dm in magenta), autoscaling.
