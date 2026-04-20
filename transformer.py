import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
import copy
from utils.get_config import config, try_cast
from utils.model_helpers import Pad

class Transformer(nn.Module):
    def __init__(self, n_features, n_concepts, output_dim, spatial_patch_size, temporal_patch_size, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.n_features = n_features
        self.n_concepts = n_concepts
        self.output_dim = output_dim
        self.spatial_patch_size = spatial_patch_size
        self.d_model = d_model
        #patch embedding
        self.patch_embedding = ConvPatchEmbedding(d_model, spatial_patch_size, temporal_patch_size, dropout, n_features)
        self.spatial_encoding = SpatialEncodingSinusoidal2D(d_model)
        self.temporal_encoding = TemporalEncodingSinusoidal(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.upsample = nn.Sequential(nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=spatial_patch_size,
            stride=spatial_patch_size
        ),
        nn.BatchNorm2d(d_model),
        nn.GELU())

        self.concept_net = nn.Conv2d(d_model, n_concepts, kernel_size=1)    #decoder queries handle the output_dim explicilty

        self.output_net = nn.Sequential(
            nn.Conv2d(n_concepts, 1, kernel_size=1),
        )

    # TODO: add back in when switching to autoregressive
    # def generate_mask(self, context_window, output_offset):
    #     # change to be land mask
    #     # and temporal mask
    #     context_window_mask = (context_window != 0).unsqueeze(1).unsqueeze(2)
    #     output_offset_mask = (output_offset != 0).unsqueeze(1).unsqueeze(3)
    #     seq_length = output_offset.size(1)
    #     nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    #     output_offset_mask = output_offset_mask & nopeak_mask
    #     return context_window_mask, output_offset_mask

    def forward(self, x):
        #context_window_mask, output_offset_mask = self.generate_mask(context_window, output_offset)
        # context_window_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(context_window)))
        # output_offset_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(output_offset)))
        t_pred_targets = try_cast(config['DATASET']['offset'])

        B, V, T, Y, X = x.shape

        padder = Pad(Y, X, self.spatial_patch_size)

        x, Y, X = padder.pad(x)

        x = x.permute(0, 2, 1, 3, 4)   # (B, T, V, Y, X)
        enc_output = self.patch_embedding(x)
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output) #, context_window_mask)

        nr = Y // config.getint('MODEL.HYPERPARAMETERS', 'spatial_patch_size')
        nc = X // config.getint('MODEL.HYPERPARAMETERS', 'spatial_patch_size')

    # temporal encoding for target timesteps
        t_targets   = torch.tensor(t_pred_targets, device=x.device).unsqueeze(0)   # (1, n_targets)
        t_enc, _    = self.temporal_encoding(t_targets)                             # (1, n_targets, d_model)
        t_enc       = t_enc.expand(B, -1, -1)                                       # (B, n_targets, d_model)

        # spatial encoding for output grid
        sp_enc      = self.spatial_encoding(nr, nc)                 # (1, N, d_model)

        # combine: each target timestep gets a full spatial grid of queries
        dec_queries = t_enc.unsqueeze(2) + sp_enc.unsqueeze(1)                      # (B, n_targets, N, d_model)
        dec_queries = dec_queries.reshape(B, self.output_dim * nr * nc, -1)                     # (B, n_targets*N, d_model)

        for dec_layer in self.decoder_layers:
            dec_queries = dec_layer(dec_queries, enc_output)                        # (B, n_targets*N, d_model)

        # --- Output ---
        dec_queries = dec_queries.reshape(B * self.output_dim, self.d_model, nr, nc)          # (B*n_targets, d_model, nr, nc)
    
        # upsample to full resolution
        print(f"dec_queries min={dec_queries.min():.4f} max={dec_queries.max():.4f}")
        dec_queries = self.upsample(dec_queries) #F.interpolate(dec_queries, size=(Y, X), mode='bilinear', align_corners=True) #mode='nearest') #
                                                                                    # (B*n_targets, d_model, Y, X)
        print(f"after upsample min={dec_queries.min():.4f} max={dec_queries.max():.4f}")
        concepts    = self.concept_net(dec_queries)                                          # (B*n_targets, n_concepts*output_dim, Y, X)
        print(f"concepts min={concepts.min():.4f} max={concepts.max():.4f}")
        output      = self.output_net(concepts)                                     # (B*n_targets, output_dim, Y, X)
        print(f"output min={output.min():.4f} max={output.max():.4f}")

        #print(concepts.shape)
        concepts = concepts.reshape(B, self.output_dim, self.n_concepts, Y, X)
        #print(output.shape)
        output = output.reshape(B, self.output_dim, 1, Y, X)

        concepts = concepts.permute(0, 2, 1, 3, 4)  # (B, n_concepts, lead, Y, X)
        output = output.permute(0, 2, 1, 3, 4)      # (B, output_dim, lead, Y, X)
        #print(concepts.shape)
        print(output.shape)

        output, concepts = padder.crop(output, concepts)

        return output, concepts

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0 #d_model must be divisible by num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.pwff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )        

    def forward(self, x):
        return self.pwff(x)

#done with convolutions which is computationally better but less interpretable?   
class ConvPatchEmbedding(nn.Module):
  def __init__(self, d_model, spatial_patch_size, temporal_patch_size, dropout, n_features):
      super().__init__()
      self.spatial_patch_size = spatial_patch_size
      self.temporal_patch_size = temporal_patch_size
      self.n_features = n_features
      self.patcher = nn.Sequential(
          # We use conv for doing the patching
          nn.Conv3d(
              in_channels= len(try_cast(config['DATASET']['features'])),
              out_channels=d_model,
              # if kernel_size = stride -> no overlap
              kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
              stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size)
          ))
          # Linear projection of Flattened Patches. We keep the batch and the channels (b,c,h,w)
          #nn.Flatten(2))
      self.spatial_encoding = SpatialEncodingSinusoidal2D(d_model)
      self.temporal_encoding = TemporalEncodingSinusoidal(d_model)
      self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
      B, T, V, Y, X = x.shape
      # Create the patches
      x = x.permute(0, 2, 1, 3, 4)                        # (B, V, T, Y, X) -> Conv3d sees (B, C, D, H, W)
      x = self.patcher(x)
      B, E, Tp, nr, nc = x.shape
      x = x.permute(0, 2, 3, 4, 1)
      x = x.reshape(B, Tp, nr * nc, E)
      # Patch + Position Embedding
      sp_enc          = self.spatial_encoding(nr, nc)   # (1, N, d_model)
      x               = x + sp_enc.unsqueeze(1)

      te_enc, cls     = self.temporal_encoding(torch.arange(Tp, device=x.device).unsqueeze(0))               # (B, T, D), (B, 1, D)
      x               = x + te_enc.unsqueeze(2)
      x = x.reshape(B, Tp * nr * nc, E)                      # (B, T*N, embed_dim)
      cls = cls.expand(B, -1, -1)
      x = torch.cat([cls, x], dim=1)                # (B, 1+T*N, embed_dim)
      x = self.dropout(x)
      return x

class SpatialEncodingSinusoidal2D(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=64):
        super().__init__()
        half   = d_model // 2
        d_term = torch.exp(torch.arange(0, half, 2).float()
                           * -(math.log(10000.0) / half))

        # row encoding
        row_pe          = torch.zeros(max_h, half)
        rows            = torch.arange(max_h).unsqueeze(1).float()
        row_pe[:, 0::2] = torch.sin(rows * d_term)
        row_pe[:, 1::2] = torch.cos(rows * d_term)

        # col encoding
        col_pe          = torch.zeros(max_w, half)
        cols            = torch.arange(max_w).unsqueeze(1).float()
        col_pe[:, 0::2] = torch.sin(cols * d_term)
        col_pe[:, 1::2] = torch.cos(cols * d_term)

        self.register_buffer('row_pe', row_pe)
        self.register_buffer('col_pe', col_pe)

    def forward(self, n_rows, n_cols):
        row_enc = self.row_pe[:n_rows].unsqueeze(1).expand(-1, n_cols, -1)  # (n_rows, n_cols, half)
        col_enc = self.col_pe[:n_cols].unsqueeze(0).expand(n_rows, -1, -1)  # (n_rows, n_cols, half)
        pe      = torch.cat([row_enc, col_enc], dim=-1)                     # (n_rows, n_cols, d_model)
        return pe.view(n_rows * n_cols, -1).unsqueeze(0)                    # (1, N, d_model)

class TemporalEncodingSinusoidal(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        pe = torch.zeros(*t.shape[:2], self.d_model, device=t.device)
        print(t.device)
        print(self.div_term.device)
        pe[..., 0::2] = torch.sin(t * self.div_term)
        pe[..., 1::2] = torch.cos(t * self.div_term)
        
        cls = self.cls_token.expand(t.shape[0], -1, -1)   # (B, 1, d_model)
        return pe, cls
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #, mask):
        attn_output = self.self_attn(x, x, x) #, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output): #, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x) #, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output) #, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x