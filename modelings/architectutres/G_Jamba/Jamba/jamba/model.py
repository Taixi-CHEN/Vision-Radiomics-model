import torch
from torch import Tensor, nn
from zeta import MambaBlock
from zeta.nn import FeedForward
from zeta import MultiQueryAttention
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
from .moe import MoE
from zeta.nn import OutputHead
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer  

class TransformerMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        num_experts: int,
        num_experts_per_token: int,
        *args,
        **kwargs,
    ):
        """
        Initializes a TransformerMoEBlock.

        Args:
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            num_experts (int): The total number of experts.
            num_experts_per_token (int): The number of experts per token.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        self.attn = MultiQueryAttention(dim, heads)
        self.moe = MoE(
            dim,
            num_experts=num_experts,
            hidden_dim=dim * 4,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerMoEBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the TransformerMoEBlock.
        """
        skip = x
        x = SimpleRMSNorm(self.dim)(x)
        x, _, _ = self.attn(x)

        x = SimpleRMSNorm(self.dim)(x)
        moe_out, _ = self.moe(x)
        x = moe_out + skip
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *args,
        **kwargs,
    ):
        """
        Initializes a TransformerBlock.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
            num_experts (int): Number of experts.
            num_experts_per_token (int): Number of experts per token.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.attn = MultiQueryAttention(dim, heads)
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swish=True,
            post_act_ln=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the TransformerBlock.
        """
        skip = x
        x = SimpleRMSNorm(self.dim)(x)
        x, _, _ = self.attn(x)
        x += skip

        skip_two = x

        x = SimpleRMSNorm(self.dim)(x)
        x = self.ffn(x) + skip_two
        return x


class MambaMoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initialize the MambaMoELayer.

        Args:
            dim (int): Dimension of the input tensor.
            d_state (int): Dimension of the state tensor.
            d_conv (int): Dimension of the convolutional tensor.
            num_experts (int, optional): Number of experts. Defaults to 8.
            num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # Mamba
        self.mamba = MambaBlock(
            dim,
            depth=1,
            d_state=d_state,
            d_conv=d_conv,
        )

        # MoE
        self.moe = MoE(
            dim,
            num_experts=num_experts,
            hidden_dim=dim * 4,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        skip = x

        x = SimpleRMSNorm(self.dim)(x)
        x = self.mamba(x) + x

        x = SimpleRMSNorm(self.dim)(x)
        moe_out, _ = self.moe(x)
        x = moe_out + skip
        return x

class CrossAttention(nn.Module):
    """Cross-attention layer for query, key, value"""
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.out_linear(attn_output)
        
        return output, attn_weights, None


class AMambaLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            d_state: int,
            d_conv: int,
            num_experts: int = 8,
            num_experts_per_token: int = 2,
            heads: int = 8,
            *args,
            **kwargs,
    ):
            """
            Initialize the MambaMoELayer.

            Args:
                dim (int): Dimension of the input tensor.
                d_state (int): Dimension of the state tensor.
                d_conv (int): Dimension of the convolutional tensor.
                num_experts (int, optional): Number of experts. Defaults to 8.
                num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
            """
            super().__init__()
            self.dim = dim
            self.d_state = d_state
            self.d_conv = d_conv
            self.num_experts = num_experts
            self.num_experts_per_tok = num_experts_per_token

            # Mamba
            self.mamba = MambaBlock(
                dim,
                depth=1,
                d_state=d_state,
                d_conv=d_conv,
            )

            # MoE
            self.moe = MoE(
                dim,
                num_experts=num_experts,
                hidden_dim=dim * 4,
            )

            # self.attn = MultiQueryAttention(dim, heads)
            self.attn = CrossAttention(dim, heads) #coore

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        skip = x

        x1 = SimpleRMSNorm(self.dim)(x)

        mamba_output = self.mamba(x1)

        x = mamba_output
        return x

class AMambaMoELayer(nn.Module):
    def __init__(
            self,
            dim: int,
            d_state: int,
            d_conv: int,
            num_experts: int = 8,
            num_experts_per_token: int = 2,
            heads: int = 8,
            *args,
            **kwargs,
    ):
            """
            Initialize the MambaMoELayer.

            Args:
                dim (int): Dimension of the input tensor.
                d_state (int): Dimension of the state tensor.
                d_conv (int): Dimension of the convolutional tensor.
                num_experts (int, optional): Number of experts. Defaults to 8.
                num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
            """
            super().__init__()
            self.dim = dim
            self.d_state = d_state
            self.d_conv = d_conv
            self.num_experts = num_experts
            self.num_experts_per_tok = num_experts_per_token

            # Mamba
            self.mamba = MambaBlock(
                dim,
                depth=1,
                d_state=d_state,
                d_conv=d_conv,
            )

            # MoE
            self.moe = MoE(
                dim,
                num_experts=num_experts,
                hidden_dim=dim * 4,
            )

            self.attn = MultiQueryAttention(dim, heads)
            # self.attn = CrossAttention(dim, heads)

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class JambaBlock(nn.Module):
    """
    JambaBlock is a module that combines MambaBlock, MambaMoELayer, and TransformerBlock
    to process input tensors.

    Args:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int, optional): The number of experts in MambaMoELayer. Defaults to 8.
        num_experts_per_token (int, optional): The number of experts per token in MambaMoELayer. Defaults to 2.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int): The number of experts in MambaMoELayer.
        num_experts_per_tok (int): The number of experts per token in MambaMoELayer.
        mamba_layer (MambaBlock): The MambaBlock layer.
        mamba_moe_layer (MambaMoELayer): The MambaMoELayer layer.
        transformer (TransformerBlock): The TransformerBlock layer.

    """

    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # self.mamba_layer = MambaBlock(
        #     dim,
        #     depth=1,
        #     d_state=d_state,
        #     d_conv=d_conv,
        # )

        self.mamba_layer = AMambaLayer(
            dim,
            d_state,
            d_conv,
            num_experts,
            num_experts_per_token,
            heads,
        )

        # # Mamba MoE layer
        self.mamba_moe_layer = MambaMoELayer(
            dim,
            d_state,
            d_conv,
            num_experts,
            num_experts_per_token,
            heads,
        )

        # # # # Transformer
        self.transformer = TransformerBlock(
            dim,
            heads,
        )

        self.transformer_moe = TransformerMoEBlock(
            dim,
            heads,
            num_experts,
            num_experts_per_token,
        )

        self.gat = GAT(nfeat=dim, 
                nhid=8, 
                nclass=3, 
                dropout=0.01, 
                nheads=8, 
                alpha=0.2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mamba_layer(x) 
        x = self.mamba_layer(x) 
        x = self.mamba_layer(x) 
        x = self.mamba_layer(x) 
      
        return x


class Jamba(nn.Module):
    """
    Jamba model implementation.

    Args:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to 8.
        num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
        pre_emb_norm (bool, optional): Whether to normalize the embeddings. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

    Attributes:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int): Number of experts.
        num_experts_per_tok (int): Number of experts per token.
        pre_emb_norm (bool): Whether to normalize the embeddings.
        return_embeddings (bool): Whether to return the embeddings.
        layers (nn.ModuleList): List of JambaBlock layers.
        embed (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm or nn.Identity): Normalization layer.

    """

    def __init__(
        self,
        dim: int,
        d_model: int,
        depth: int,
        num_tokens: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        pre_emb_norm: bool = False,
        return_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.depth = depth
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token
        self.pre_emb_norm = pre_emb_norm
        self.return_embeddings = return_embeddings

        # Layers
        self.layers = nn.ModuleList(
            [
                JambaBlock(
                    d_model,
                    d_state,
                    d_conv,
                    heads,
                    num_experts,
                    num_experts_per_token,
                )
                for _ in range(depth)
            ]
        )

        # Pre Emb
        # self.embed = nn.Embedding(num_tokens, dim)
        # self.embed = nn.Linear(num_tokens, dim)
        self.input_projection = nn.Linear(dim, d_model)

        # Embedding Norm
        self.norm = (
            nn.LayerNorm(d_model) if pre_emb_norm else nn.Identity()
        )

        # self.output_head = OutputHead(self.dim, -1, 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model,self.d_model),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model,self.d_model),
            nn.ReLU(),
            # # nn.Dropout(0.1),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model,self.d_model // 2),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Linear(self.d_model // 2, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Jamba model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.

        """
        # Embed the input tensor to transform
        # From tokens -> tensors
        # print(x.shape)

        # x = x.unsqueeze(-1)
        x  = self.input_projection(x)
        # print(x.shape)
        # x = self.embed(x)
        # # print(x)
        # x = x.squeeze(2)
        # print(x.shape)
        # Normalize the embeddings
        x = self.norm(x)
        # x = x.unsqueeze(1)
        # print(x.shape)
        # Apply the layers
        for layer in self.layers:
            x = layer(x)
        # print(x)
        # print(f"x: {x.shape}")
        # return x
        # if self.return_embeddings:
        #     return x
        # else:
            # return the logits
        return self.classifier(x)
