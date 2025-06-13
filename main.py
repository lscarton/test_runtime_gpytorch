import torch
import gpytorch
import time
from loguru import logger

# Set the seed for reproducibility
torch.manual_seed(0)


EPOCHS = 4_000
RERUN_N_TIMES = 5
N_PREDICTION = 10
N_TRAIN_DATAPOINTS = 2_000

# Define the toy function
def toy_function(x):
    y = torch.zeros(x.shape[0], 6)
    y[:, 0] = torch.sin(x[:, 0]) + 0.1 * torch.randn(x.shape[0])
    y[:, 1] = torch.cos(x[:, 1]) + 0.1 * torch.randn(x.shape[0])
    y[:, 2] = x[:, 2] ** 2 + x[:, 3] + 0.1 * torch.randn(x.shape[0])
    y[:, 3] = x[:, 4] ** 3 + x[:, 5] ** 2 + 0.1 * torch.randn(x.shape[0])
    y[:, 4] = torch.sin(x[:, 6] + x[:, 7]) + torch.cos(x[:, 8]) + 0.1 * torch.randn(x.shape[0])
    y[:, 5] = x[:, 9] ** 2 + x[:, 10] ** 3 + x[:, 11] + 0.1 * torch.randn(x.shape[0])
    return y

# Define the MultitaskGPModel
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(MultitaskGPModel, self).__init__(train_x, train_y, gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=6))
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=6
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=6, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# Train the model on CPU
def train_on_cpu(train_x, train_y, epochs):
    start_time = time.time()
    model = MultitaskGPModel(train_x, train_y)
    likelihood = model.likelihood
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.debug(f"CPU Epoch {i+1}, Loss: {loss.item():.4f}")
    end_time = time.time()
    return end_time - start_time

# Train the model on GPU
def train_on_gpu(train_x, train_y, epochs):
    if torch.cuda.is_available():
        start_time = time.time()
        model = MultitaskGPModel(train_x.cuda(), train_y.cuda())
        likelihood = model.likelihood
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood.cuda(), model.cuda())
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(train_x.cuda())
            loss = -mll(output, train_y.cuda())
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.debug(f"GPU Epoch {i+1}, Loss: {loss.item():.4f}")
        end_time = time.time()
        return end_time - start_time
    else:
        logger.warn("GPU not available, skipping GPU training")
        return None

# Make predictions on CPU
def make_predictions_on_cpu(train_x, train_y):
    start_time = time.time()
    model = MultitaskGPModel(train_x, train_y)
    likelihood = model.likelihood
    model.eval()
    likelihood.eval()
    test_x = torch.randn(N_PREDICTION, 12)
    with torch.no_grad():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
    logger.info("CPU predicted")
    end_time = time.time()
    return end_time - start_time

# Make predictions on GPU
def make_predictions_on_gpu(train_x, train_y):
    if torch.cuda.is_available():
        start_time = time.time()
        model = MultitaskGPModel(train_x.cuda(), train_y.cuda()).cuda()
        likelihood = model.likelihood.cuda()
        model.eval()
        likelihood.eval()
        test_x = torch.randn(N_PREDICTION, 12).cuda()
        with torch.no_grad():
            predictions = likelihood(model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
        logger.info("GPU predicted")
        end_time = time.time()
        return end_time - start_time
    else:
        logger.warn("GPU not available, skipping GPU predictions")
        return None

def main():
    # Generate the dataset
    n_train = N_TRAIN_DATAPOINTS
    train_x = torch.randn(n_train, 12)
    train_y = toy_function(train_x)

    # Set the number of iterations
    n_iterations = RERUN_N_TIMES

    # Initialize the total times
    total_cpu_train_time = 0
    total_gpu_train_time = 0
    total_cpu_prediction_time = 0
    total_gpu_prediction_time = 0

    # Call each function N times
    for i in range(n_iterations):
        logger.info(f"Iteration {i+1}")
        gpu_train_time = train_on_gpu(train_x, train_y, EPOCHS)
        if gpu_train_time is not None:
            total_gpu_train_time += gpu_train_time
        cpu_train_time = train_on_cpu(train_x, train_y, EPOCHS)
        total_cpu_train_time += cpu_train_time
        gpu_prediction_time = make_predictions_on_gpu(train_x, train_y)
        if gpu_prediction_time is not None:
            total_gpu_prediction_time += gpu_prediction_time
        cpu_prediction_time = make_predictions_on_cpu(train_x, train_y)
        total_cpu_prediction_time += cpu_prediction_time

    # Calculate the average times
    average_cpu_train_time = total_cpu_train_time / n_iterations
    if torch.cuda.is_available():
        average_gpu_train_time = total_gpu_train_time / n_iterations
    else:
        average_gpu_train_time = None
    average_cpu_prediction_time = total_cpu_prediction_time / n_iterations
    if torch.cuda.is_available():
        average_gpu_prediction_time = total_gpu_prediction_time / n_iterations
    else:
        average_gpu_prediction_time = None

    # Print the average times
    logger.success(f"Average CPU training time: {average_cpu_train_time:.2f} seconds for {EPOCHS} epochs,  {N_TRAIN_DATAPOINTS} datapoints")
    if average_gpu_train_time is not None:
        logger.success(f"Average GPU training time: {average_gpu_train_time:.2f} seconds for {EPOCHS} epochs,  {N_TRAIN_DATAPOINTS} datapoints")
    logger.success(f"Average CPU prediction time: {average_cpu_prediction_time:.2f} seconds")
    if average_gpu_prediction_time is not None:
        logger.success(f"Average GPU prediction time: {average_gpu_prediction_time:.2f} seconds")

if __name__ == "__main__":
    main()