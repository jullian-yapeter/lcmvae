import torch.nn as nn


class LINEAR_NETWORK_PARAMS:
    output_dim = 768
    activation = nn.LeakyReLU()
    linear_layer_params = [
        {"in_dim": 1536, "out_dim": 1536},
        {"in_dim": 1536, "out_dim": 768},
        {"in_dim": 768, "out_dim": output_dim}
    ]
