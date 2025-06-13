# Test Runtime GPyTorch

Simple script to test the runtime of GPyTorch on CPU and GPU on my PC.

- It uses a `MultitaskGPModel` trained.
- It trains for `4000` epochs.
- The input dim is `12`.
- The number of tasks is `6`.
- The number of predicted points is `10`.

- It repeat the experiment `5` times and average the time.


## How to run
```bash
git clone https://github.com/lscarton/test_runtime_gpytorch.git
cd test_runtime_gpytorch
```
Then for simplicity it uses `uv`.
```bash
uv run main.py
```
