import torch
from models.maml import copy_params, copy_model_params
from typing import Callable, Iterable, List, Optional, Tuple, Dict
from architecture.neural_ode import NeuralODE  

def online_adapt_maml(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_stream: Iterable[Tuple[Tuple, torch.Tensor]],
    lr: float,
    use_full_history: bool = False,
    _params: Optional[List[Tuple[torch.tensor, torch.tensor]]] = None
) -> Tuple[Dict[str,torch.Tensor], list]:
    """
    Generic online adaptation for any model.forward signature.
    data_stream is an iterable of tuples, where each tuple contains:
    - inputs: a tuple of inputs to the model (e.g., (x,)),
    - targets: the target tensor (e.g., y).
    The data_stream is the data used for online adaptation at a single timestep, which can be a single observation or a window of observations.
    
    To continuously adapt the model, you can call this function in a loop, feeding it new data at each step and updating the model parameters.
    """
    batch_size = 1  # We assume a batch size of 1 for online adaptation (there is one observation/task at a timestep, this can be either a single datapoint or a window of datapoints).

    # 1) Extract and clone all parameters for adaptation
    if _params is None:
        # if no params are provided, we copy the model parameters
        params = copy_model_params(model, batch_size)
    else:
        # params = copy_params(_params, batch_size)
        params = _params # use the provided parameters for adaptation

    losses = []
    param_updates = [params.copy()]

    for t in range(len(data_stream)):
        # 2) Run your model with the current parameters
        if use_full_history:
            # feed in x[0:t+1], y[0:t+1]
            data = data_stream[:t+1]
        else:
            # just the single step at t
            data = data_stream[t:t+1]

        # unzip the list of tuples into two lists
        inputs_seq, targets_seq = zip(*data)
        # inputs_seq is a tuple of input‐tensors, length = len(data)
        # targets_seq is the same for targets

        # stack along the time dimension for each of the inputs and targets
        if type(inputs_seq[0]) is not tuple:
            args = torch.cat([inp for inp in inputs_seq], dim=1)
            # args is now a Tensor of shape [batch, points, in_features]
        else:
            # in the neural_ode case, each entry of inputs_seq is a tuple of Tensors, with shape ([batch, points, in_features], [1,1])
            # we need to stack the inputs along the time dimension
            stacked_args = []
            for i in range(len(inputs_seq[0])):
                slot = [inp[i] for inp in inputs_seq]
                # now slot is a list of Tensors of length >1
                stacked_args.append(torch.cat(slot, dim=1))
            args = tuple(stacked_args) # [batch, points, in_features] = [1, N, in_features]
        targets = torch.cat([target for target in targets_seq], dim=1)
        # targets is now a Tensor of shape [batch, points, out_features]

        # compute the loss and grads for the current step
        if type(model) is NeuralODE:
            y_pred = model.forward(args, {'params': params})
        else:
            y_pred = model.forward(args, params)
        loss = loss_fn(y_pred, targets)
        losses.append(loss.item())

        grads = torch.autograd.grad(loss, [p for tuple in params for p in tuple], retain_graph=True)

        # 4) Do a step of SGD in param-space from the parameters of the previous step
        for i in range(len(params)):
            W, b = param_updates[-1][i]
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
