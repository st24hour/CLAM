import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from utils.utils import initialize_weights


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 384, dropout = False):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, 1)

    def forward(self, x):
        a = self.attention_a(x)     # N x D     TODO: check it
        b = self.attention_b(x)     # N x D     TODO: check it
        A = a.mul(b)                # N x D     TODO: check it
        A = self.attention_c(A)     # N x 1     TODO: check it
        return A

class WSI_Encoder(nn.Module):
    def __init__(self, 
                 attn, 
                 feature_dim, 
                 output_dim, 
                 dropout):
        super(WSI_Encoder, self).__init__()

        fc = [nn.Linear(feature_dim[0], feature_dim[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if attn == 'gated':
            attention_net = Attn_Net_Gated(L = feature_dim[1], D = feature_dim[2], dropout = dropout)
        else:
            raise NotImplementedError('Only "gated" is allowed')
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        initialize_weights(self)
    
    def forward(self, h):
        A = self.attention_net(h)  # Nx1 (N: slide당 patch 개수)
        A = torch.transpose(A, 1, 0)  # 1xN
        A = F.softmax(A, dim=1)  # softmax over N
        # print(A.size())     # (1,N)
                
        M = torch.mm(A, h)                  # 얘가 feature 같은데?
        print(M.size())     # [1, feature_dim[0]]
        logits = self.classifiers(M)

        return logits


class Genomics_Encoder_FC(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers = 2, 
                 activation_func = nn.ReLU, 
                 norm_layer = None):
        super(Genomics_Encoder_FC, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))  # Fully connected layer
            if norm_layer is not None:  # Add normalization layer if specified
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_func())  # Activation function
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)  # Create a sequential model

    def forward(self, x):
        return self.model(x)


class CLIP(nn.Module):
    def __init__(self,
                 attn = 'gated',
                 size_arg = 'custom2_big',
                 dropout = False,
                 input_dim_genom = 2847,
                 hidden_dim_genom = 1024,
                 output_dim = 1024,
                 num_layers_genom = 2, 
                 activation_func_genom = nn.ReLU, 
                 norm_layer_genom = None):
        super().__init__()

        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384], "custom1":[512, 512, 256], "custom2":[768, 512, 256], 
                          "HIPT_4k_feat": [192, 512, 256], "HIPT_256_feat": [384, 512, 256], "custom2_big":[768, 512, 384]}       
        feature_dim = self.size_dict[size_arg]

        self.encode_image = WSI_Encoder(attn, feature_dim, output_dim, dropout)
        self.encode_text = Genomics_Encoder_FC(input_dim_genom, hidden_dim_genom, output_dim, num_layers_genom, activation_func_genom, norm_layer_genom)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init param ??
    
    def forward(self, image, genomics):
        image_features = self.encode_image(image)
        genomics_features = self.encode_text(genomics)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ genomics_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

