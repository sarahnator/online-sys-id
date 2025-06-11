import torch
from itertools import islice
from torch.utils.data import IterableDataset

def train_step(model, optimizer, batch, loss_function, device: str = "cpu"):
    """Performs a single training step, clips grad norm, and returns the loss value."""
    model.train()
    optimizer.zero_grad()
    loss = loss_function(model, batch, device=device)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def test_eval(model, dataloader, loss_function, device: str = "cpu"):
    """Evaluates the model on a dataset and returns the average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        if isinstance(dataloader.dataset, IterableDataset):
            # If the dataset is iterable, limit evaluation to a finite number of batches
            # to avoid infinite loops during testing.
            for batch in islice(dataloader, 10):
                loss = loss_function(model, batch, device=device)
                total_loss += loss.item()
                count += 1
        else:
            for batch in dataloader:
                loss = loss_function(model, batch, device=device)
                total_loss += loss.item()
                count += 1
    return total_loss / count if count > 0 else float("nan")

def plot_training_curves(losses: dict):
    """Plots training and test losses."""
    import matplotlib.pyplot as plt
    n_plots = len(losses)
    n_rows = 1 if n_plots <= 3 else 2

    fig, axs = plt.subplots(n_rows, n_plots, figsize=(10, 5 * n_rows))
    if n_rows == 1:
        axs = axs.reshape(1, n_plots)

    for i, loss_type in enumerate(losses.keys()):
        axs[0, i].plot(losses[loss_type], label=f"{loss_type.capitalize()} Loss")
        axs[0, i].set_title(f"{loss_type.capitalize()} Loss")
        axs[0, i].set_xlabel("Epochs")
        axs[0, i].set_ylabel("Loss") if i == 0 else None
        axs[0, i].set_yscale('log')
        
    plt.tight_layout()
    plt.show()