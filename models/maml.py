from typing import Callable, Optional, Tuple, Union
import torch
from models.BaseModel import BaseModel
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice
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
        return self.model.forward_with_params(x, w)

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
            # elif isinstance(layer, torch.nn.Conv2d):
            #     W_copy = layer.weight.unsqueeze(0).expand(num_copies, -1, -1, -1, -1).clone()
            #     b_copy = layer.bias.unsqueeze(0).expand(num_copies, -1,).clone()
            #     params.append((W_copy, b_copy))
            else:
                raise NotImplementedError("Layer type not supported in copy_params")

        return params

    # def make_batched_state_dict(self, num_copies: int):
    #     """
    #     Returns a dict mapping every param/buffer name to a tensor of shape
    #     [num_copies, *original_shape], so that we can use vmap + functional_call to evaluate the NN with custom parameters in batch.
    #     """
    #     sd = self.state_dict()
    #     batched = {}
    #     for k, v in sd.items():
    #         # unsqueeze a new batch‐axis, expand & clone
    #         batched[k] = v.unsqueeze(0).expand(num_copies, *v.shape).clone()
    #     return batched
    
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
    
    def inner_update_step_from_params(self, x: torch.Tensor, y: torch.Tensor, params: torch.Tensor):
        """ Note: Tyler warns this might be very slow.
        Perform inner loop updates for the MAML model. n_points corresponds to the number of time steps in the time series and the number of inner loop updates.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_points, input_dim).
            y (torch.Tensor): Target tensor of shape (batch_size, n_points, output_dim).
            params (torch.Tensor): Initial parameters of the model of shape (batch_size, n_layers, 2), where each tuple contains (weight, bias).
            The weight has shape (batch_size, out_features, in_features) and the bias has shape (batch_size, out_features).
        Returns:
            torch.Tensor: Updated weights after the inner loop updates.
        """
        n_points = x.size(1)

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

                # test the model - data stream prediction
                total_test_loss = 0.0
                count = 0
                if isinstance(test_dataloader.dataset, IterableDataset):
                    # If the dataset is iterable, limit evaluation to a finite number of batches
                    # to avoid infinite loops during testing.
                    for batch in islice(test_dataloader, 10):
                        y0_test, dt_test, y1_test, y0_test_example, dt_test_example, y1_test_example, test_info = batch
                        test_batch = (y0_test_example.to(device), y1_test_example.to(device), y0_test.to(device), y1_test.to(device))
                        _, test_loss, _ = self.datastream_predict(
                            xs=test_batch[0],
                            ys=test_batch[1],
                            query_xs=test_batch[2],
                            query_ys=test_batch[3]
                        )
                        total_test_loss += test_loss.item()
                        count += 1
                else:
                    for batch in test_dataloader:
                        y0_test, dt_test, y1_test, y0_test_example, dt_test_example, y1_test_example, test_info = batch
                        test_batch = (y0_test_example.to(device), y1_test_example.to(device), y0_test.to(device), y1_test.to(device))
                        _, test_loss, _ = self.datastream_predict(
                            xs=test_batch[0],
                            ys=test_batch[1],
                            query_xs=test_batch[2],
                            query_ys=test_batch[3]
                        )
                        total_test_loss += test_loss.item()
                        count += 1

                avg_test_loss = total_test_loss / count if count > 0 else float("nan")
                losses['test'].append(avg_test_loss)

                tqdm_bar.set_postfix_str(f"inner_train_loss: {inner_train_loss:.2e}, meta_train_loss: {meta_train_loss:.2e}, test_loss: {test_loss:.2e}")

        return losses
    
    def datastream_predict(self, xs, ys, query_xs, query_ys) -> torch.Tensor:
        """
        Predict using streaming updates to the model parameters based on the provided xs and ys.
        This method is used for making predictions on new data after the model has been trained.
        Args:
            xs: Input features for the model.
            ys: Target values for the model.
            query_xs: Query input features for the model.
            query_ys: Query target values for the model.
        Returns:
            torch.Tensor: The output of the model's forward method.
        """

        # Fine tune based on the provided xs and ys, in a streaming fashion
        adapted_weights, _ = self.inner_update_step(xs, ys)

        # Evaluate the model on the query data using the adapted weights from all of the data
        with torch.no_grad():
            predictions =  self.forward(query_xs, adapted_weights)
            loss = self.loss_function(predictions, query_ys)

        return predictions, loss, adapted_weights
    

class MAML2(MAML):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.
    This class extends the BaseModel and implements the MAML algorithm for meta-learning.
    It allows for inner loop updates and meta updates, suitable for few-shot learning tasks.
    """
    def __init__(self, model, meta_learning_rate: float = 1e-3, internal_learning_rate: float = 1e-3):
        super().__init__(model, meta_learning_rate, internal_learning_rate)

    def inner_update_step(self, x: torch.Tensor, y: torch.Tensor):
        """ Note: Tyler warns this might be very slow.
        Perform inner loop updates for the MAML model. n_points corresponds to the number of time steps in the time series and the number of inner loop updates.
        In comparison to the above implementation of MAML, this version uses a single point (n_points=1) for each inner update.
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
            predictions = self.forward(x[:, t:t+1, :], params)
            loss = self.loss_function(predictions, y[:, t:t+1, :])
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
    
    def inner_update_step_from_params(self, x: torch.Tensor, y: torch.Tensor, params: torch.Tensor):
        """ Note: Tyler warns this might be very slow.
        Perform inner loop updates for the MAML model. n_points corresponds to the number of time steps in the time series and the number of inner loop updates.
        In comparison to the above implementation of MAML, this version uses a single point for each inner update.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_points, input_dim).
            y (torch.Tensor): Target tensor of shape (batch_size, n_points, output_dim).
            params (torch.Tensor): Initial parameters of the model of shape (batch_size, n_layers, 2), where each tuple contains (weight, bias).
            The weight has shape (batch_size, out_features, in_features) and the bias has shape (batch_size, out_features).
        Returns:
            torch.Tensor: Updated weights after the inner loop updates.
        """
        n_points = x.size(1)

        losses = torch.zeros(n_points, device=x.device)  # to store the losses for each update
        for t in range(n_points): # num updates is equal to the number of points in the time series
            # Continually update the model using all of the data up to time t? # TODO also implement a version that only uses the last time step
            predictions = self.forward(x[:, t:t+1, :], params)
            loss = self.loss_function(predictions, y[:, t:t+1, :])
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
    
class MAML2_NODE(MAML2):
    """
    Model-Agnostic Meta-Learning (MAML) implementation for Neural ODEs.
    This class extends the MAML2 class and is specifically designed for Neural ODE models.
    It allows for inner loop updates and meta updates, suitable for few-shot learning tasks with Neural ODEs.
    """
    def __init__(self, model, meta_learning_rate: float = 1e-3, internal_learning_rate: float = 1e-3):
        super().__init__(model, meta_learning_rate, internal_learning_rate)


    def forward(self, inputs, model_kwargs=None):
        # return self.model.forward_with_params(inputs, model_kwargs=model_kwargs)
        return self.model.forward(inputs, ode_kwargs=model_kwargs)

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
                    dt=inner_train_batch[1],
                    y=inner_train_batch[2]
                )
                
                # perform meta update step
                meta_train_loss = self.meta_update_step(
                    x=meta_train_batch[0], 
                    dt=meta_train_batch[1],
                    y=meta_train_batch[2],
                    adapted_weights=adapted_weights,
                    clip_grad_norm_=True
                )
                losses['inner_train'].append(inner_train_loss.item())
                losses['meta_train'].append(meta_train_loss.item())

                # test the model - data stream prediction
                total_test_loss = 0.0
                count = 0
                if isinstance(test_dataloader.dataset, IterableDataset):
                    # If the dataset is iterable, limit evaluation to a finite number of batches
                    # to avoid infinite loops during testing.
                    for batch in islice(test_dataloader, 10):
                        y0_test, dt_test, y1_test, y0_test_example, dt_test_example, y1_test_example, test_info = batch
                        test_batch = (y0_test_example.to(device), dt_test_example.to(device), y1_test_example.to(device), y0_test.to(device), dt_test.to(device), y1_test.to(device))
                        _, test_loss, _ = self.datastream_predict(
                            xs=test_batch[0],
                            dt=test_batch[1],
                            ys=test_batch[2],
                            query_xs=test_batch[3],
                            query_dt=test_batch[4],
                            query_ys=test_batch[5]
                        )
                        total_test_loss += test_loss.item()
                        count += 1
                else:
                    for batch in test_dataloader:
                        y0_test, dt_test, y1_test, y0_test_example, dt_test_example, y1_test_example, test_info = batch
                        test_batch = (y0_test_example.to(device), dt_test_example.to(device), y1_test_example.to(device), y0_test.to(device), dt_test.to(device), y1_test.to(device))
                        _, test_loss, _ = self.datastream_predict(
                            xs=test_batch[0],
                            dt=test_batch[1],
                            ys=test_batch[2],
                            query_xs=test_batch[3],
                            query_dt=test_batch[4],
                            query_ys=test_batch[5]
                        )
                        total_test_loss += test_loss.item()
                        count += 1

                avg_test_loss = total_test_loss / count if count > 0 else float("nan")
                losses['test'].append(avg_test_loss)

                tqdm_bar.set_postfix_str(f"inner_train_loss: {inner_train_loss:.2e}, meta_train_loss: {meta_train_loss:.2e}, test_loss: {test_loss:.2e}")
                # tqdm_bar.set_postfix_str(f"inner_train_loss: {inner_train_loss:.2e}, meta_train_loss: {meta_train_loss:.2e}")

        return losses
    
    def inner_update_step(self, x: torch.Tensor, dt: torch.Tensor, y: torch.Tensor):
        """ Note: Tyler warns this might be very slow.
        Perform inner loop updates for the MAML model. n_points corresponds to the number of time steps in the time series and the number of inner loop updates.
        In comparison to the above implementation of MAML, this version uses a single point for each inner update.
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
            predictions = self.forward((x[:, t:t+1, :], dt[:, t:t+1]), model_kwargs={'params': params})
            loss = self.loss_function(predictions, y[:, t:t+1, :])
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
    

    def meta_update_step(self,  x: torch.Tensor, dt: torch.Tensor, y: torch.Tensor, adapted_weights: torch.Tensor, clip_grad_norm_: bool) -> torch.Tensor:
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
    
        predictions = self.forward((x, dt), model_kwargs={'params': adapted_weights})
        meta_loss = self.loss_function(predictions, y)
        # Backward pass to compute gradients with respect to the meta loss
        meta_loss.backward()

        if clip_grad_norm_:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update model parameters using the meta learning rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return meta_loss

    def datastream_predict(self, xs, dt, ys, query_xs, query_dt, query_ys) -> torch.Tensor:
        """
        Predict using streaming updates to the model parameters based on the provided xs and ys.
        This method is used for making predictions on new data after the model has been trained.
        Args:
            xs: Input features for the model. Shape (batch_size, n_points, input_dim).
            dt: Time deltas for the model. Shape (batch_size, n_points, 1).
            ys: Target values for the model.
            query_xs: Query input features for the model.
            query_ys: Query target values for the model.
        Returns:
            torch.Tensor: The output of the model's forward method.
        """

        # Fine tune based on the provided xs and ys, in a streaming fashion
        # return a the adapted weights based on the adapted weights after updating with each point individually
        adapted_weights, _ = self.inner_update_step(
                    x=xs,
                    dt=dt,
                    y=ys
                )
        # return a prediction based on the last adapted weights
        with torch.no_grad():
            predictions =  self.forward((query_xs, query_dt), model_kwargs={'params': adapted_weights})
            loss = self.loss_function(predictions, query_ys)
        
        return predictions, loss, adapted_weights
    