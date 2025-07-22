from util.losses import neural_ode_loss

def maml_inner_update_and_forward(model, data):
    """
    Perform a MAML inner update step and forward pass.

    Returns a tuple (loss: torch.Tensor, output: Any)
    """
    # Unpack data
    y0, dt, y1 = data

    # Inner update
    adapted_weights, _ = model.inner_update_step(x=y0, dt=dt, y=y1)

    # Forward pass
    output = model.forward(inputs=(y0, dt), model_kwargs={'params': adapted_weights})
    loss = neural_ode_loss(output, y1)

    return loss, output

def fe_rls_update_and_forward(model, data):
    """
    Perform a FE-RLS update step and forward pass.

    Returns a tuple (loss: torch.Tensor, output: Any)
    """
    # Unpack data
    y0, dt, y1 = data

    # Inner update
    adapted_weights, _ = model.inner_update_step(x=y0, dt=dt, y=y1)

    # Forward pass
    output = model.forward(inputs=(y0, dt), model_kwargs={'params': adapted_weights})
    loss = neural_ode_loss(output, y1)

    return loss, output