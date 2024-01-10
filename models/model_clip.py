import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import initialize_weights

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial


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
        a = self.attention_a(x)     # N x D
        b = self.attention_b(x)     # N x D
        A = a.mul(b)                # N x D
        A = self.attention_c(A)     # N x 1
        return A

class SI_Attn(nn.Module):
    def __init__(self, dim = 512, heads = 8, dim_head = 64, dropout = 0.):
        super(SI_Attn, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.norm = nn.LayerNorm(inner_dim * 3)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.norm(self.to_qkv(x)).chunk(3, dim = -1)
        # print(qkv[0].size())        # (num_patch, D)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CLAM_Encoder(nn.Module):
    def __init__(self, 
                 attn, 
                 feature_dim, 
                 output_dim, 
                 dropout):
        super(CLAM_Encoder, self).__init__()

        fc = [nn.Linear(feature_dim[0], feature_dim[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if attn == 'gated':
            attention_net = Attn_Net_Gated(L = feature_dim[1], D = feature_dim[2], dropout = dropout)
        else:
            raise NotImplementedError('Only "gated" is allowed')
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        if output_dim is not None:
            self.classifiers = nn.Linear(feature_dim[0], output_dim)

        initialize_weights(self)
    
    def _forward_single(self, h):
        # if len(size)==3:
        #     h = h.view(-1, size[-1])      # (N, 786)  786: feature dim

        A = self.attention_net(h)  # Nx1 (N: slide당 patch 개수)
        A = torch.transpose(A, 1, 0)  # 1xN
        A = F.softmax(A, dim=1)  # softmax over N,   (1,N)
                
        M = torch.mm(A, h)                  # 얘가 feature 같은데? [1, feature_dim[0]]
        return M

    def _forward_batch(self, h):
        # if len(size)==4:
        #     h = h.view(size[0], -1, size[-1])     # (b, N, 768)

        A = self.attention_net(h)  # (b,N,1) (b: batch size, N: slide당 patch 개수)
        A = torch.transpose(A, 2, 1)  # (b,1,N)
        A = F.softmax(A, dim=2)  # softmax over N,  (b,1,N)
                
        # M = torch.bmm(A, h)     # (b,1,768)
        M = torch.matmul(A, h).squeeze(1)   # (b,768)
        return M

    def forward(self, h):        
        size = h.size()
        if len(size)==3:
            h = h.view(-1, size[-1])
            M = self._forward_single(h)
        elif len(size)==4:
            h = h.view(size[0], -1, size[-1])
            M = self._forward_batch(h)

        if hasattr(self, 'classifiers'):
            logits = self.classifiers(M)
            return logits
        else:
            return M
        

class Genomics_Encoder_FC(nn.Module):
    '''
    Caution: The actual number of layers is num_layers + 1, because of the output layer.    
    '''
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers = 2, 
                 activation_func = nn.ReLU(inplace=True), 
                 norm_layer = None):
        super(Genomics_Encoder_FC, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))  # Fully connected layer
            if norm_layer is not None:  # Add normalization layer if specified
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_func)  # Activation function
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)  # Create a sequential model

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        return self.model(x)

class Genomics_Encoder_FC_Skip_early_v1(nn.Module):
    '''
    early fusion
    (2847) + (2847->hidden->2847) -> output_dim
    
    Caution: The actual number of layers is num_layers + 1, because of the output layer.
    '''
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim = 786, 
                 num_layers = 2, 
                 activation_func = nn.ReLU(inplace=True), 
                 norm_layer = None):
        super(Genomics_Encoder_FC_Skip_early_v1, self).__init__()

        layers = []
        prev_dim = input_dim        # 2847

        # Hidden layers
        for i in range(num_layers):
            if i != num_layers-1:
                layers.append(nn.Linear(prev_dim, hidden_dim))  # Fully connected layer
                if norm_layer is not None:  # Add normalization layer if specified
                    layers.append(norm_layer(hidden_dim))
                prev_dim = hidden_dim
            else:       # last layer
                layers.append(nn.Linear(prev_dim, input_dim))
                if norm_layer is not None:  # Add normalization layer if specified
                    layers.append(norm_layer(input_dim))
            layers.append(activation_func)  # Activation function    

        self.blocks = nn.Sequential(*layers)  # Create a sequential model
        
        # Output layer
        self.final_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # if use_batch == False
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        
        out = self.blocks(x)
        out = out + x
        out = self.final_fc(out)
        return out


# class Genomics_Encoder_FC_Skip_late_v1(nn.Module):
#     def __init__(self, 
#                  input_dim, 
#                  hidden_dim, 
#                  num_blocks, 
#                  activation_func = nn.ReLU, 
#                  norm=None):
#         super(Genomics_Encoder_FC_Skip_late_v1, self).__init__()
#         self.num_blocks = num_blocks
#         self.activation_func = activation_func
#         self.norm = norm

#         # Define the initial FC layer
#         self.initial_fc = nn.Linear(input_dim, hidden_dim)

#         # Define FC blocks with residual connections and optional normalization
#         self.fc_blocks = nn.ModuleList()
#         for _ in range(num_blocks):
#             layers = [nn.Linear(hidden_dim, hidden_dim), self.activation_func()]
#             if self.norm is not None:
#                 layers.append(self.norm(hidden_dim))
#             self.fc_blocks.append(nn.Sequential(*layers))

#         # Define the final FC layer to map back to input_dim
#         self.final_fc = nn.Linear(hidden_dim, input_dim)

#         # Define the output FC layer to get 786 dimensions
#         self.output_fc = nn.Linear(input_dim, 786)

#     def forward(self, x):
#         x = self.initial_fc(x)
#         x = self.activation_func(x)
#         if self.norm is not None:
#             x = self.norm(hidden_dim)(x)

#         residual = x

#         for i, fc_block in enumerate(self.fc_blocks):
#             x = fc_block(x)
#             if i % 2 == 1:  # Add the residual connection every two blocks
#                 x += residual
#                 residual = x

#         x = self.final_fc(x)
#         x = self.output_fc(x)

#         return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


#######################################################################################################
#######################################################################################################
#######################################################################################################
class CLIP(nn.Module):
    def __init__(self,
                 image_encoder,
                 text_encoder):
        super().__init__()

        # self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384], "custom1":[512, 512, 256], "custom2":[768, 512, 256], 
        #                   "HIPT_4k_feat": [192, 512, 256], "HIPT_256_feat": [384, 512, 256], "custom2_big":[768, 512, 384]}       
        # feature_dim = self.size_dict[size_arg]
        # output_dim_genomics = output_dim if output_dim is not None else feature_dim[0]

        # self.encode_image = CLAM_Encoder(attn, feature_dim, output_dim, dropout)
        # self.encode_text = Genomics_Encoder_FC(input_dim_genom, hidden_dim_genom, output_dim_genomics, num_layers_genom, activation_func_genom, norm_layer_genom)
        self.encode_image = image_encoder
        self.encode_text = text_encoder

        # init param ??
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # print(image_features.size())        # (1,1024)
        # print(text_features.size())         # (1,1024)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # CLAM은 batch size 1로 고정해야되므로 feature 뽑아놓고 모아놨다가 CLIP_similaritys()에서 구함
        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text
        return image_features, text_features

class CLIP_similaritys(nn.Module):
    def __init__(self):
        super().__init__()    

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()        

        return logits_per_image, logits_per_text


def make_clip(
        image_encoder_name = 'CLAM_Encoder',
        text_encoder_name = 'Genomics_Encoder_FC',
        size_arg='custom2_big',
        output_dim=None,
        input_dim_genom = 2847,
        hidden_dim_genom = 4096,
        num_layers_genom = 2, 
        activation_func_genom = 'ReLU', 
        norm_genom = None):

    size_dict = {
        "small": [1024, 512, 256], 
        "big": [1024, 512, 384], 
        "custom1":[512, 512, 256], 
        "custom2":[768, 512, 256], 
        "HIPT_4k_feat": [192, 512, 256], 
        "HIPT_256_feat": [384, 512, 256], 
        "custom2_big":[768, 512, 384]}       

    feature_dim = size_dict[size_arg]
    output_dim_genomics = output_dim if output_dim is not None else feature_dim[0]

    if activation_func_genom == 'ReLU':
        activation_func = nn.ReLU(inplace=True)
    elif hasattr(nn, activation_func_genom):
        call_activation_func = getattr(nn, activation_func_genom)
        activation_func = call_activation_func()
    else:
        raise ValueError(f"Activation function '{activation_func}' not found in nn module.")

    if norm_genom:
        if hasattr(nn, norm_genom):
            norm_layer = getattr(nn, norm_genom)
        else:
            raise ValueError(f"Normalization function '{norm_genom}' not found in nn module.")
    else:
        norm_layer = None

    config = {
        'CLAM_Encoder': {'attn':'gated', 'feature_dim':feature_dim, 'output_dim':output_dim, 'dropout':False},
        'SI_attn': {'dim': 512, 'heads': 8, 'dim_head': 64, 'dropout': 0.},

        'Genomics_Encoder_FC': {'input_dim':input_dim_genom, 'hidden_dim': hidden_dim_genom, 'output_dim': output_dim_genomics, \
                                'num_layers': num_layers_genom, 'activation_func': activation_func, 'norm_layer': norm_genom},
        'Genomics_Encoder_FC_Skip_early_v1': {'input_dim': input_dim_genom, 'hidden_dim': hidden_dim_genom, 'output_dim': output_dim_genomics, \
                                              'num_layers': num_layers_genom, 'activation_func': activation_func, 'norm_layer': norm_layer},
        'Identity': {}
    }

    call_image_encoder = globals()[image_encoder_name]
    call_text_encoder = globals()[text_encoder_name]     

    image_encoder = call_image_encoder(**config[image_encoder_name])
    text_encoder = call_text_encoder(**config[text_encoder_name])

    model = CLIP(image_encoder, text_encoder)

    return model