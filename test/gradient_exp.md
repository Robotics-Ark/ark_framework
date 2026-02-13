# Gradient Experiment

Demonstrates differentiable simulation using ark framework. A `LinePublisherNode` publishes position on a line (`y = m*x + c`, `x = v*t`) along with autograd gradients (dx/dv, dy/dm, dy/dc), and an `AutodiffPlotterNode` subscribes to position and queries gradients in real time.

## Prerequisites

- Install ark framework and dependencies (`zenoh`, `torch`, `matplotlib`, `ark_msgs`)
- Run all commands from the `test/` directory

## Running the Experiment

Open three separate terminals. All commands are run from the `test/` directory.

### Shell 1 — Sim Clock

Drives simulated time for all sim-enabled nodes.

```bash
cd test
python simstep.py
```

### Shell 2 — Diff Publisher

Publishes position (`Translation`) and serves gradient queryables (`grad/v/x`, `grad/m/y`, etc.).

```bash
cd test
python diff_publisher.py
```

### Shell 3 — Autodiff Plotter

Subscribes to position and queries gradients, then plots both in real time.

```bash
cd test
python ad_plotter_sub.py
```

## What to Expect

- **Shell 1** prints the simulated time advancing each tick.
- **Shell 2** prints computed gradients (`Grad v_x`, `Grad m_y`) each step.
- **Shell 3** opens a matplotlib window with two plots:
  - **Left**: Position trajectory (x vs y).
  - **Right**: Gradients over time (dx/dv in green, dy/dm in magenta).
