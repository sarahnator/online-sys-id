import torch


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
    """Evaluates the model on the dataset and returns the average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = loss_function(model, batch, device=device)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float("nan")