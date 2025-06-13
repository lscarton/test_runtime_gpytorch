import torch
import gpytorch
import time
from loguru import logger

# Set the seed for reproducibility
torch.manual_seed(0)


EPOCHS = 50
RERUN_N_TIMES = 5
N_PREDICTION = 10
N_TRAIN_DATAPOINTS = 2000



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

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_task, ard_num, rank):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_task
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num),
            num_tasks=n_task,
            rank=rank,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x += 1e-9 * torch.eye(covar_x.shape[0], device=covar_x.device)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
# Train the model on CPU


# Train the model on GPU
def train_on_(train_x, train_y, epochs, device, n_task, ard_num, rank):
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=n_task,
    ).to(device)
    model = MultitaskGPModel(train_x, train_y, likelihood, n_task, ard_num, rank).to(device)
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    start_time = time.time()
    for i in range(EPOCHS):
        optimizer.zero_grad()
        output = model(train_x.to(device))
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, EPOCHS, loss.item()))
        optimizer.step()
        end_time = time.time()
    return model, likelihood, end_time - start_time
    


# Make predictions on CPU
def make_predictions_on_(model, likelihood, test_x, device):
    start_time = time.time()

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        predictions = likelihood(model(test_x))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
    logger.info(f"{device} predicted")
    end_time = time.time()
    return end_time - start_time


def main():
    # Generate the dataset
    n_train = N_TRAIN_DATAPOINTS
    train_x = torch.randn(n_train, 12)
    train_y = toy_function(train_x)
    test_x = torch.randn(N_PREDICTION, 12)
    ard_num = train_x.shape[1]
    rank = 6
    logger.info(f"Number of independent input is: {ard_num}")

    n_task = train_y.shape[1]
    logger.info(f"number of tasks: {n_task}")
    if torch.cuda.is_available():
        train_x_cuda = train_x.to('cuda')
        train_y_cuda = train_y.to('cuda')
        test_x_cuda = test_x.to('cuda')
    else:
        logger.warn("CUDA not available, using CPU only")


    # Set the number of iterations
    n_iterations = RERUN_N_TIMES

    # Initialize the total times
    total_cpu_train_time = 0
    total_gpu_train_time = 0
    total_cpu_prediction_time = 0
    total_gpu_prediction_time = 0

    # Call each function N times
    if torch.cuda.is_available():
        for i in range(n_iterations):
            logger.info(f"GPU Iteration {i+1}")
            model, likelihood, gpu_train_time = train_on_(train_x_cuda, train_y_cuda, EPOCHS, 'cuda', n_task, ard_num, rank)
            if gpu_train_time is not None:
                total_gpu_train_time += gpu_train_time
            gpu_prediction_time = make_predictions_on_(model, likelihood, test_x_cuda,'cuda')
            if gpu_prediction_time is not None:
                total_gpu_prediction_time += gpu_prediction_time

    for i in range(n_iterations):
        logger.info(f"CPU Iteration {i+1}")
        model, likelihood, cpu_train_time = train_on_(train_x, train_y, EPOCHS, 'cpu', n_task, ard_num, rank)
        total_cpu_train_time += cpu_train_time
        cpu_prediction_time = make_predictions_on_(model, likelihood, test_x, 'cpu')
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