import torch
from models.maml import copy_params
from typing import Callable, Iterable, Tuple, Dict

def online_adapt_maml(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_stream: Iterable[Tuple[Tuple, torch.Tensor]],
    lr: float,
    use_full_history: bool = False,
) -> Tuple[Dict[str,torch.Tensor], list]:
    """
    Generic online adaptation for any model.forward signature.
    data_stream is an iterable of tuples, where each tuple contains:
    - inputs: a tuple of inputs to the model (e.g., (x,)),
    - targets: the target tensor (e.g., y).
    """
    batch_size = 1  # We assume one sample per time-step

    # 1) Extract and clone all parameters for adaptation
    params = copy_params(model, batch_size)

    losses = []
    param_updates = []

    for t in range(len(data_stream)):
        # 2) Run your model with the current parameters
        if use_full_history:
            # feed in x[0:t+1], y[0:t+1]
            data = data_stream[:t+1]
        else:
            # just the single step at t
            data = data_stream[t:t+1]

        # now unzip the list of tuples into two lists
        inputs_seq, targets_seq = zip(*data)
        # inputs_seq is a tuple of input‐tensors, length = len(data)
        # targets_seq is the same for targets

        # 1) stack the targets
        targets = torch.stack(targets_seq, dim=1)

        # 2) figure out how many "positional args" each inputs tuple has
        # if only one step, just grab that tuple directly
        if len(inputs_seq) == 1:
            # inputs_seq[0] is either a Tensor or a tuple of Tensors
            args = (inputs_seq[0],) if isinstance(inputs_seq[0], torch.Tensor) \
                else inputs_seq[0]
            targets = targets_seq[0]
        else:
            # full‐history case: do your normal stack-along-time
            N = len(inputs_seq[0])
            stacked_args = []
            for i in range(N):
                slot = [inp[i] for inp in inputs_seq]
                # now slot is a list of Tensors of length >1
                stacked_args.append(torch.cat(slot, dim=1))
            args = tuple(stacked_args)
            targets = torch.cat(targets_seq, dim=1)
        # compute the loss and grads for the current step
        y_pred = model.forward(args, {'params': params})
        loss = loss_fn(y_pred, targets)
        losses.append(loss.item())

        grads = torch.autograd.grad(loss, [p for tuple in params for p in tuple], retain_graph=True)

        # 4) Do a step of SGD in param-space
            
        for i in range(len(params)):
            W, b = params[i]
            W_grad, b_grad = grads[2*i], grads[2*i + 1]
            if torch.isnan(W_grad).any():
                print("NaN detected")
            if torch.isnan(b_grad).any():
                print("NaN detected")
            W = W - lr * W_grad
            b = b - lr * b_grad
            params[i] = (W, b)
        param_updates.append(params)

    return params, losses, param_updates
