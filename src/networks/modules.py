import torch
import torch.nn as nn
import numpy as np
import ml_collections
import copy
import math

def PositionalEncoding3D(channel, shape):

        channels = int(np.ceil(channel/6)*2)
        if channels % 2:
            channels += 1

        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

        x, y, z = shape
        pos_x = torch.arange(x).type(inv_freq.type())
        pos_y = torch.arange(y).type(inv_freq.type())
        pos_z = torch.arange(z).type(inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((1,x,y,z,channels*3)).type(inv_freq.type())
        emb[0,:,:,:,:channels] = emb_x
        emb[0,:,:,:,channels:2*channels] = emb_y
        emb[0,:,:,:,2*channels:] = emb_z

        return emb.permute(0,4,1,2,3).reshape(1, channel, -1).permute(0, 2, 1)

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config["num_heads"]
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.out = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = nn.Dropout(config["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config["attention_dropout_rate"])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config['hidden_size'], config["mlp_dim"])
        self.fc2 = nn.Linear(config["mlp_dim"], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config['hidden_size']
        self.attention_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, config, channel, shape, device, vis=False):
        super(Transformer, self).__init__()
        self.vis = vis
        self.pos_emb = PositionalEncoding3D(config['hidden_size'], shape).to(device)
        self.layer = nn.ModuleList()
        self.proj_layer1 = nn.Linear(channel, config['hidden_size'])
        self.proj_layer2 = nn.Linear(config['hidden_size'], channel)
        self.encoder_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.attn_weights = []
        for _ in range(config["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        b, ch, h, w, d = hidden_states.shape
        hidden_states = hidden_states.reshape(b, ch, -1)
        hidden_states = self.proj_layer1(hidden_states.permute(0, 2, 1))
        hidden_states += self.pos_emb.repeat(b,1,1)
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                self.attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        encoded = self.proj_layer2(encoded).permute(0, 2, 1)
        encoded = encoded.reshape(b, ch, h, w, d)
        return encoded