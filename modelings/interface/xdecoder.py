#!/usr/bin/env python3
"""
图像-radiomics相似性预训练模块
通过计算图像特征和radiomics特征的相似性分数作为loss进行预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImageRadiomicsSimilarityModule(nn.Module):
    """
    图像-radiomics相似性计算模块
    用于预训练阶段，计算图像特征和radiomics特征的相似性
    """
    
    def __init__(self, image_feature_dim, radiomics_feature_dim, similarity_method='cosine'):
        super().__init__()
        self.image_feature_dim = image_feature_dim
        self.radiomics_feature_dim = radiomics_feature_dim
        self.similarity_method = similarity_method
        
        # 将radiomics特征投影到图像特征维度，图像特征保持不变
        self.radiomics_proj = nn.Sequential(
            nn.Linear(radiomics_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, image_feature_dim)  # 投影到图像特征维度
        )
        
        # 温度参数，用于控制相似性分布的锐度
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.radiomics_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, image_features, radiomics_features):
        """
        计算图像特征和radiomics特征的相似性
        
        Args:
            image_features: [B, C, H, W] - 图像特征
            radiomics_features: [B, num_rois, radiomics_dim] - radiomics特征
            
        Returns:
            similarity_scores: [B, B] - 相似性分数矩阵
            image_embeddings: [B, C, H, W] - 图像特征（保持不变）
            radiomics_embeddings: [B, C, H, W] - 投影后的radiomics嵌入
        """
        B, C, H, W = image_features.shape
        
        # 1. 图像特征保持不变
        image_embeddings = image_features  # [B, C, H, W]
        
        # 2. 处理radiomics特征：平均池化 + 投影到图像特征维度 + 重塑为空间格式
        radiomics_global = radiomics_features.mean(dim=1)  # [B, radiomics_dim]
        radiomics_projected = self.radiomics_proj(radiomics_global)  # [B, C] - 投影到图像特征维度
        
        # 3. 将投影后的radiomics特征重塑为空间格式 [B, C, H, W]
        radiomics_embeddings = radiomics_projected.view(B, C, 1, 1)  # [B, C, 1, 1]
        radiomics_embeddings = radiomics_embeddings.expand(B, C, H, W)  # [B, C, H, W]
        
        # 4. 计算相似性分数（使用全局平均池化后的特征）
        image_global = F.adaptive_avg_pool2d(image_embeddings, (1, 1)).view(B, -1)  # [B, C]
        radiomics_global = F.adaptive_avg_pool2d(radiomics_embeddings, (1, 1)).view(B, -1)  # [B, C]
        similarity_scores = self._compute_similarity(image_global, radiomics_global)
        
        return similarity_scores, image_embeddings, radiomics_embeddings
    
    def _compute_similarity(self, image_emb, radiomics_emb):
        """
        计算相似性分数
        
        Args:
            image_emb: [B, C] - 图像嵌入（原始图像特征维度）
            radiomics_emb: [B, C] - radiomics嵌入（投影到图像特征维度）
            
        Returns:
            similarity_scores: [B, B] - 相似性分数矩阵
        """
        if self.similarity_method == 'cosine':
            # 余弦相似性
            image_emb_norm = F.normalize(image_emb, p=2, dim=1)
            radiomics_emb_norm = F.normalize(radiomics_emb, p=2, dim=1)
            similarity_scores = torch.mm(image_emb_norm, radiomics_emb_norm.t())
            
        elif self.similarity_method == 'dot_product':
            # 点积相似性
            similarity_scores = torch.mm(image_emb, radiomics_emb.t())
            
        elif self.similarity_method == 'euclidean':
            # 欧几里得距离（转换为相似性）
            # 计算所有图像和radiomics特征之间的距离
            image_expanded = image_emb.unsqueeze(1)  # [B, 1, 128]
            radiomics_expanded = radiomics_emb.unsqueeze(0)  # [1, B, 128]
            distances = torch.norm(image_expanded - radiomics_expanded, dim=2)  # [B, B]
            similarity_scores = -distances  # 距离越小，相似性越高
            
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")
        
        # 应用温度参数
        similarity_scores = similarity_scores / self.temperature
        
        return similarity_scores
    
    def compute_similarity_loss(self, similarity_scores, labels=None):
        """
        计算相似性损失
        
        Args:
            similarity_scores: [B, B] - 相似性分数矩阵
            labels: [B] - 标签（可选，用于对比学习）
            
        Returns:
            loss: 相似性损失
        """
        B = similarity_scores.shape[0]
        
        if labels is not None:
            # 使用标签进行对比学习
            # 对角线元素是正样本，其他是负样本
            positive_mask = torch.eye(B, device=similarity_scores.device).bool()
            negative_mask = ~positive_mask
            
            # 正样本分数
            positive_scores = similarity_scores[positive_mask]
            
            # 负样本分数
            negative_scores = similarity_scores[negative_mask].view(B, B-1)
            
            # 计算对比损失
            positive_exp = torch.exp(positive_scores)
            negative_exp = torch.exp(negative_scores).sum(dim=1)
            
            # 避免数值不稳定
            positive_exp = torch.clamp(positive_exp, min=1e-8)
            negative_exp = torch.clamp(negative_exp, min=1e-8)
            
            # 对比损失
            loss = -torch.log(positive_exp / (positive_exp + negative_exp)).mean()
            
        else:
            # 使用对角线作为正样本
            positive_scores = torch.diag(similarity_scores)
            
            # 计算softmax损失
            logits = similarity_scores
            labels = torch.arange(B, device=similarity_scores.device)
            
            # 使用交叉熵损失
            loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_consistency_loss(self, image_emb, radiomics_emb):
        """
        计算一致性损失（L2距离）
        
        Args:
            image_emb: [B, C, H, W] - 图像嵌入（原始图像特征）
            radiomics_emb: [B, C, H, W] - radiomics嵌入（投影到图像特征维度并重塑为空间格式）
            
        Returns:
            consistency_loss: 一致性损失
        """
        # 计算L2距离（在空间维度上）
        l2_distance = F.mse_loss(image_emb, radiomics_emb)
        
        # 计算余弦相似性（在通道维度上，对每个空间位置）
        # 将特征重塑为 [B*H*W, C] 来计算余弦相似性
        B, C, H, W = image_emb.shape
        image_flat = image_emb.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        radiomics_flat = radiomics_emb.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        
        cosine_sim = F.cosine_similarity(image_flat, radiomics_flat, dim=1)  # [B*H*W]
        cosine_loss = (1 - cosine_sim).mean()
        
        # 组合损失
        consistency_loss = l2_distance + cosine_loss
        
        return consistency_loss


class SimilarityPretrainLoss(nn.Module):
    """
    相似性预训练损失函数
    组合多种损失函数进行预训练
    """
    
    def __init__(self, similarity_weight=1.0, consistency_weight=0.5, temperature=0.07):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.consistency_weight = consistency_weight
        self.temperature = temperature
    
    def forward(self, image_features, radiomics_features, labels=None):
        """
        计算预训练损失
        
        Args:
            image_features: [B, C, H, W] - 图像特征
            radiomics_features: [B, num_rois, radiomics_dim] - radiomics特征
            labels: [B] - 标签（可选）
            
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 创建相似性模块（现在radiomics投影到图像特征维度）
        similarity_module = ImageRadiomicsSimilarityModule(
            image_feature_dim=image_features.shape[1],
            radiomics_feature_dim=radiomics_features.shape[-1],
            similarity_method='cosine'
        ).to(image_features.device)
        
        # 计算相似性分数和嵌入
        similarity_scores, image_emb, radiomics_emb = similarity_module(
            image_features, radiomics_features
        )
        
        # 计算相似性损失
        similarity_loss = similarity_module.compute_similarity_loss(
            similarity_scores, labels
        )
        
        # 计算一致性损失
        consistency_loss = similarity_module.compute_consistency_loss(
            image_emb, radiomics_emb
        )
        
        # 总损失
        total_loss = (self.similarity_weight * similarity_loss + 
                     self.consistency_weight * consistency_loss)
        
        # 损失字典
        loss_dict = {
            'similarity_loss': similarity_loss,
            'consistency_loss': consistency_loss,
            'total_pretrain_loss': total_loss
        }
        
        return total_loss, loss_dict


def create_pretrain_optimizer(model, learning_rate=1e-4, weight_decay=1e-5):
    """
    创建预训练优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        
    Returns:
        optimizer: 优化器
    """
    # 只优化相似性相关的参数
    pretrain_params = []
    for name, param in model.named_parameters():
        if 'similarity' in name or 'pretrain' in name:
            pretrain_params.append(param)
    
    if not pretrain_params:
        # 如果没有专门的相似性参数，优化所有参数
        pretrain_params = list(model.parameters())
    
    optimizer = torch.optim.AdamW(
        pretrain_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    return optimizer


def pretrain_step(model, batch, similarity_loss_fn, optimizer, device):
    """
    执行一步预训练
    
    Args:
        model: 模型
        batch: 批次数据
        similarity_loss_fn: 相似性损失函数
        optimizer: 优化器
        device: 设备
        
    Returns:
        loss_dict: 损失字典
    """
    model.train()
    
    # 获取图像和radiomics特征
    images = batch['images'].to(device)
    radiomics = batch['radiomics'].to(device)
    
    # 前向传播获取特征
    with torch.no_grad():
        # 获取图像特征（不计算梯度，只用于特征提取）
        image_features = model.backbone(images)
        # 选择某个层级的特征，比如res4
        image_feat = image_features['res4']  # [B, C, H, W]
    
    # 处理radiomics特征
    if isinstance(radiomics, list):
        # 如果radiomics是列表，需要处理
        radiomics_tensors = []
        for rad in radiomics:
            if isinstance(rad, torch.Tensor):
                radiomics_tensors.append(rad)
            else:
                # 创建dummy radiomics
                dummy_rad = torch.randn(1, 128).to(device)
                radiomics_tensors.append(dummy_rad)
        
        # 填充到相同长度
        max_rois = max(rad.shape[0] for rad in radiomics_tensors)
        padded_radiomics = []
        for rad in radiomics_tensors:
            if rad.shape[0] < max_rois:
                padding = torch.zeros(max_rois - rad.shape[0], rad.shape[1]).to(device)
                padded_rad = torch.cat([rad, padding], dim=0)
            else:
                padded_rad = rad
            padded_radiomics.append(padded_rad)
        
        radiomics_features = torch.stack(padded_radiomics)  # [B, max_rois, radiomics_dim]
    else:
        radiomics_features = radiomics
    
    # 计算相似性损失
    total_loss, loss_dict = similarity_loss_fn(image_feat, radiomics_features)
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss_dict

