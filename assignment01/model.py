import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        
        # input dim is 3 (x,y,z). output dim of the last layer is 1 (SDF value).
        # We append 1 to dims for the final output layer.
        dims = dims + [1]
        self.num_layers = len(dims)
        self.latent_in = latent_in
        self.use_tanh = use_tanh

        # The first layer takes the 3D coordinate as input
        prev_dim = 3
        
        for i in range(len(dims)):
            out_dim = dims[i]
            
            # If this is a skip connection layer, we adjust the input dimension
            # because we will concatenate the original 3D input to this layer's input
            if i in self.latent_in:
                layer_in_dim = prev_dim + 3
            else:
                layer_in_dim = prev_dim

            curr_layer = nn.Linear(layer_in_dim, out_dim)

            # Apply Weight Normalization if requested
            if weight_norm:
                curr_layer = nn.utils.weight_norm(curr_layer)

            setattr(self, "lin" + str(i), curr_layer)
            prev_dim = out_dim

        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_prob)
        self.th = nn.Tanh()
        
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        x = input

        for i in range(self.num_layers):
            lin = getattr(self, "lin" + str(i))

            # Concatenate the original input for skip connections
            if i in self.latent_in:
                x = torch.cat([x, input], 1)

            x = lin(x)

            # Apply activation and dropout for all but the last layer
            if i < self.num_layers - 1:
                x = self.relu(x)
                x = self.dropout(x)

        # Apply Tanh to the final output if specified
        if self.use_tanh:
            x = self.th(x)
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x