import argparse
import os

from datetime import datetime
import torch
import time
from torch.utils.tensorboard import SummaryWriter


from datasets.get_dataset import get_dataset
from models.get_model import get_model, get_loss_function
from util.training import plot_training_curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--n_basis", type=int, default=11)
    parser.add_argument("--n_example_points", type=int, default=100)
    parser.add_argument("--n_points", type=int, default=1000)
    parser.add_argument("--algorithm", type=str, default="NODE_FE")
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--dataset_name", type=str, default="VanDerPol")
    parser.add_argument("--n_params", type=int, default=int(1e6))
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

# find device
if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device not in ["cpu", "cuda"]:
    args.device = f"cuda:{args.device}"

if args.device == "cpu":
    print("WARNING: Running on CPU. This will be slow.")

# arguments
load_dir = args.load_dir
log_dir = args.log_dir
n_basis = args.n_basis
n_example_points = args.n_example_points
n_points = args.n_points
algorithm = args.algorithm
epochs = args.epochs
dataset_name = args.dataset_name
# set random seed for reproducibility
torch.manual_seed(args.seed)
n_params = args.n_params
n_layers = args.n_layers
seed = args.seed
device = args.device
optimizer = args.optimizer
lr = args.lr
batch_size = args.batch_size

# init dataset
train_dataset = get_dataset(dataset_name, n_example_points=n_example_points, n_points=n_points)
test_dataset = get_dataset(dataset_name, n_example_points=100, n_points=100)

# init model
model = get_model(
    algorithm=algorithm,
    n_layers=n_layers,
    n_params=n_params,
    n_basis=n_basis,
    device=device
)

# set up optimizer
if optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
else:
    raise ValueError(f"Unknown optimizer: {optimizer}")

# train
loss_function = get_loss_function(algorithm=algorithm)
losses = model.fit(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    optimizer=optimizer,
    loss_function=loss_function,
    epochs=epochs,
    batch_size=batch_size,
    device=device
)


# save model and training logs

if load_dir is None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path = os.path.join(log_dir, f"{dataset}_{algorithm}_{timestamp}.pt")
    
    log_dir = f"{log_dir}/{dataset_name}_{algorithm}"
    save_path = f"{log_dir}/model.pth"

    writer = SummaryWriter(log_dir=log_dir)
    for epoch, (train_loss, test_loss) in enumerate(zip(losses['train'], losses['test'])):
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
    writer.close()

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# plot training curves
plot_training_curves(
    losses=losses
)