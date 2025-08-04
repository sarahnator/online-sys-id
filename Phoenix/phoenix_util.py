
from models.get_model import *

def load_model(model_type, device, n_basis=8, path=None):
    match model_type:
        case "neural_ode":
            model = NeuralODE(
            ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
            integrator=rk4_step,
        ).to(device)   
            model.loss_fn = torch.nn.MSELoss()
            model.load_state_dict(torch.load(path, map_location=device))
        case "function_encoder":
            model = FunctionEncoder(basis_functions=BasisFunctions(
                *[
                    NeuralODE(
                        ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
                        integrator=rk4_step,
                    )
                    for _ in range(n_basis)
                ]
            )).to(device)
            model.loss_fn = torch.nn.MSELoss()
            model.load_state_dict(torch.load(path, map_location=device))
        case "maml2_node":
            model = MAML2_NODE(
                NeuralODE(ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
                integrator=rk4_step),
                meta_learning_rate=1e-3,    
                internal_learning_rate=1e-3
            ).to(device)
            model.loss_function = torch.nn.MSELoss()
            model.load_state_dict(torch.load(path, map_location=device))
        case "maml_node":
            model = MAML_NODE(
                NeuralODE(ode_func=ODEFunc(model=MLP(layer_sizes=[9, 128, 128, 6], activation=torch.nn.ReLU(), bias=True)),
                integrator=rk4_step),
                meta_learning_rate=1e-3,
                internal_learning_rate=1e-3,
                window_size=100,  
            ).to(device)
            model.loss_function = torch.nn.MSELoss()
            model.load_state_dict(torch.load(path, map_location=device))
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    return model