from typing import Callable, Optional, Tuple, Union
import torch
from models.BaseModel import BaseModel
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import Optimizer
import tqdm

class MAML(BaseModel):
    def __init__(self, model, meta_learning_rate: float = 1e-3, internal_learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.meta_learning_rate = meta_learning_rate
        self.internal_learning_rate = internal_learning_rate
        
        self.loss_function = None  # Placeholder for the loss function, to be set during training
        self.optimizer = None  # Placeholder for the optimizer, to be set during training


    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MAML model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            w (torch.Tensor): Adapted parameters (weights) obtained from inner loop updates.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.model(x, w)
    
    def copy_params(self, num_copies):
        # copy-paste of Tyler's code
        # first generate models based on the current parameters
        # this generates a new model for each example
        params = []
        layers = self.model.layers
        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                W_copy = layer.weight.unsqueeze(0).expand(num_copies, -1, -1).clone()
                b_copy = layer.bias.unsqueeze(0).expand(num_copies, -1,).clone()
                params.append((W_copy, b_copy))
            elif isinstance(layer, torch.nn.Conv2d):
                W_copy = layer.weight.unsqueeze(0).expand(num_copies, -1, -1, -1, -1).clone()
                b_copy = layer.bias.unsqueeze(0).expand(num_copies, -1,).clone()
                params.append((W_copy, b_copy))

        return params

    def meta_update_step(self,  x: torch.Tensor, y: torch.Tensor, adapted_weights: torch.Tensor, clip_grad_norm_: bool) -> torch.Tensor:
        """
        Perform the meta update step for the MAML model.
        This function is typically called after the inner loop updates.
        It updates the model parameters based on the gradients computed from the inner loop.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_points, input_dim).
            y (torch.Tensor): Target tensor of shape (batch_size, n_points, output_dim).
            adapted_weights (torch.Tensor): Weights after inner loop updates.
        Returns:
            torch.Tensor: Meta loss value.
        """
        predictions = self.forward(x, adapted_weights)
        meta_loss = self.loss_function(predictions, y)
        # Backward pass to compute gradients with respect to the meta loss
        meta_loss.backward()

        if clip_grad_norm_:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update model parameters using the meta learning rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return meta_loss


    def inner_update_step(self, x: torch.Tensor, y: torch.Tensor):
        """ Note: Tyler warns this might be very slow.
        Perform inner loop updates for the MAML model. n_points corresponds to the number of time steps in the time series and the number of inner loop updates.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_points, input_dim).
            y (torch.Tensor): Target tensor of shape (batch_size, n_points, output_dim).
        Returns:
            torch.Tensor: Updated weights after the inner loop updates.
        """
        n_points = x.size(1)
        batch_size = x.size(0)
        # params = self.model.parameters().clone() # copy the parameters to avoid modifying the original model
        # might need to iterate through and manually clone
        # params = [p.clone() for p in self.model.parameters()]

        # extend params to (batch_size, parameter_size) so that we have a parameter update for each task (batch)
        # params = params.unsqueeze(0).expand(batch_size, -1, -1)
        params = self.copy_params(batch_size)  # copy the parameters for each task in the batch
        # TODO: test accuracy pre-update. Should be random accuracy.

        losses = torch.zeros(n_points, device=x.device)  # to store the losses for each update
        for t in range(n_points): # num updates is equal to the number of points in the time series
            # Continually update the model using all of the data up to time t? # TODO also implement a version that only uses the last time step
            predictions = self.forward(x[:, :t+1, :], params)
            loss = self.loss_function(predictions, y[:, :t+1, :])
            losses[t] = loss.item()
            # print(f"Inner update {t+1}/{n_points}, loss: {loss.item():.4f}")

            # back prop to compute gradients, retain graph so we can compute the gradient of the loss w.r.t. the parameters
            # without losing the computation graph. 
            # Essentially, this allows us to use the same forward pass results for multiple backward passes.
            grads = torch.autograd.grad(loss, [p for tuple in params for p in tuple], retain_graph=True)
            # update the parameters using the gradients (do stochastic gradient descent)
            # Tyler's code: update the parameters by hand, since torch  is not meant for this.
            for i in range(len(params)):
                W, b = params[i]
                W_grad, b_grad = grads[2*i], grads[2*i + 1]
                if torch.isnan(W_grad).any():
                    print("NaN detected")
                if torch.isnan(b_grad).any():
                    print("NaN detected")
                W = W - self.internal_learning_rate * W_grad
                b = b - self.internal_learning_rate * b_grad
                params[i] = (W, b)

        # TODO: test accuracy post-update. Should be better than pre-update.

        # mean the losses over all updates
        mean_loss = torch.mean(losses.cpu())
        return params, mean_loss

    def _train(self,
            train_dataset: IterableDataset,
            test_dataset: Optional[IterableDataset],
            optimizer: Optimizer,
            loss_function: Optional[Callable]=None,
            epochs: int = 1,
            batch_size: int = 32,
            device: str = "cpu") -> None:
        """
        A basic training loop.
        """
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        train_dataloader_iter = iter(train_dataloader)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        test_dataloader_iter = iter(test_dataloader)

        # set the loss function if not provided
        self.loss_function = torch.nn.MSELoss()

        # set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_learning_rate)

        self.train()
        device = torch.device(device)
        losses = {
            'meta_train': [],
            'inner_train': [],
            'test': []
        }

        with tqdm.trange(epochs) as tqdm_bar:
            for epoch in tqdm_bar:
                train_batch = next(train_dataloader_iter)
                y0, dt, y1, y0_example, dt_example, y1_example, info = train_batch

                # split train_batch into inner_train and meta_train
                inner_train_batch = (y0.to(device), dt.to(device), y1.to(device))
                meta_train_batch = (y0_example.to(device), dt_example.to(device), y1_example.to(device))

                # perform inner loop updates
                adapted_weights, inner_train_loss = self.inner_update_step(
                    x=inner_train_batch[0],
                    y=inner_train_batch[2]
                )
                
                # perform meta update step
                meta_train_loss = self.meta_update_step(
                    x=meta_train_batch[0],
                    y=meta_train_batch[2],
                    adapted_weights=adapted_weights,
                    clip_grad_norm_=True
                )
                losses['inner_train'].append(inner_train_loss.item())
                losses['meta_train'].append(meta_train_loss.item())

                # # test the model - write data stream predict function
                # self.eval()
                # with torch.no_grad():
                #     for test_batch in test_dataloader:
                #         test_batch = test_batch.to(device)
                #         predictions = self.forward(test_batch)
                #         test_loss = loss_function(predictions, test_batch)
                #         losses['test'].append(test_loss.item())

                tqdm_bar.set_postfix_str(f"inner_train_loss: {inner_train_loss:.2e}, meta_train_loss: {meta_train_loss:.2e}")

        return losses
    
    def datastream_predict(self, *args, **kwargs):
        self.eval()

        with torch.no_grad():
            return self.forward(*args, **kwargs)