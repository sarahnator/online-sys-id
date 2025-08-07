import torch


def train_step(model, optimizer, loss_fn, batch, device):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model, batch, device)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def test_eval(model, loss_fn, batch, device):
    model.eval()
    with torch.no_grad():
        if len(batch[0].shape) == 2:
            batch = [b.unsqueeze(0) for b in batch[:-1]]
            batch.append(batch[-1])  # Keep the info dict as is
        loss = loss_fn(model, batch, device)
    return loss.item()

def rls_test_eval(model, loss_fn, batch, coeffs, device):
    model.eval()
    with torch.no_grad():
        if len(batch[0].shape) == 2:
            batch = [b.unsqueeze(0) for b in batch]
        loss = loss_fn(model, batch, coeffs, device)
    return loss.item()


def inertial_to_body(
        bIMat,    # (K, 3) matrix of body frame origin vectors in the inertial frame
        xIMat,    # (K, 3) matrix of vectors in the inertial frame
        device
):
    """ Transforms inertial frame vectors into the body frame. """

    # Extract the rotation angles. Ensure separate memory by cloning.
    yaws = bIMat[:,2].clone()  

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0], device=device)
    ones = torch.ones(yaws.shape[0], device=device)

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([cos_yaw, sin_yaw, zeros], dim=1),
        torch.stack([-sin_yaw, cos_yaw, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # Shape: (K, 3, 3)

    # Perform batch matrix-vector multiplication
    xBMat = torch.bmm(R, xIMat.unsqueeze(-1)).squeeze(-1) #+ bIMat # Shape: (N, 3)
    return xBMat


def inertial_to_body_XL(
        bIMat,    # (N, K, 3) matrix of body frame origin vectors in the inertial frame
        xIMat,    # (N, K, 3) matrix of vectors in the inertial frame
        device
):
    """ Transforms inertial frame vectors into the body frame. """

    # Extract the rotation angles. Ensure separate memory by cloning.
    yaws = bIMat[:,:,2].clone()  

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros_like(yaws, device=device)
    ones = torch.ones_like(yaws, device=device)

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([cos_yaw, sin_yaw, zeros], dim=2),
        torch.stack([-sin_yaw, cos_yaw, zeros], dim=2),
        torch.stack([zeros, zeros, ones], dim=2)
    ], dim=2)  # Shape: (N, K, 3, 3)

    # Perform batch matrix-vector multiplication
    xBMat = torch.matmul(R, xIMat.unsqueeze(-1)).squeeze(-1)
    return xBMat