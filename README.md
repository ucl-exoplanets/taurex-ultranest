# TauREx-Ultranest

version 0.5.0-alpha

A plugin for [TauREx](https://github.com/ucl-exoplanets/TauREx3_public) 3.1 that provides the [Ultranest](https://johannesbuchner.github.io/UltraNest/index.html) sampler by Johannes Buchner for the retireval


## Installation

Installing is simply done by running
```bash
pip install taurex_ultranest
```

### Installing from source


You can install from source by doing:
```bash
git clone https://github.com/ucl-exoplanets/taurex-ultranest.git
cd taurex_ultranest
pip install .
```

## Running in TauREx

Once installed you can select the sampler through the **optimize** keyword under
Optimizer.

```
[Optimizer]
optimizer = ultranest
num_live_points=500
dlogz=0.5
dkl=0.5
```

### Input arguments:

The input arguments generally match the arguments from Ultranest.


|Argument| Description| Type| Default | Required |
---------|------------|-----|---------|----------|
num_live_points |  minimum number of live points throughout the run | int | 100 | |
dlogz | Target evidence uncertainty.  | float | 0.5 | |
dkl | Target posterior uncertainty. | float | 0.5 | |
frac_remain | Integrate until this fraction of the integral is left in the remainder. | float | 0.01 | |
cluster_num_live_points | require at least this many live points per detected cluster | Type | 40 | |
max_num_improvement_loops |  limits the number of improvement loops. | int | Default | |
stepsampler | Choose which stepsampler to use. See StepSamplers | str | default | |
nsteps | number of accepted steps until the sample is considered independent. | int | 10 | |
step_scale | initial proposal size | float | 1.0 | |
adaptive_nsteps | Select a strategy to adapt the number of steps.  | (False, 'proposal-distance', 'move-distance') | Default | |
region_filter | if True, use region to check if a proposed point can be inside before calling likelihood. | bool | False | |
resume | See [docs](https://johannesbuchner.github.io/UltraNest/ultranest.html) | ('resume', 'resume-similar', 'overwrite' or 'subfolder')  | subfolder | |
log_dir | Directory to store sampling checkpoint files | str | . | Y |

## Step samplers

You can select a specific sampler by passing in the correct string to *stepsampler*.
Documentation for each sampler is found [here](https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.stepsampler)

|Keyword | Sampler|
---------|--------|
cube-mh| CubeMHSampler|
region-mh| RegionMHSampler|
cube-slice| CubeSliceSampler|
region-slice| RegionSliceSampler|
region-sequentical-slice| RegionSequentialSliceSampler|
ball-slice| BallSliceSampler|
region-ball-slice| RegionBallSliceSampler|