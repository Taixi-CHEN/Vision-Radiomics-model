import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class FusionClassificationHead(nn.Module):
    """
    基于融合特征的分类头，用于tumor/normal二分类
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,  # tumor vs normal
        dropout_rate: float = 0.1,
        fusion_method: str = 'concat'  # 'concat', 'attention', 'mean'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # 特征融合层
        if fusion_method == 'concat':
            self.fusion_layer = nn.Linear(input_dim, hidden_dim)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
            self.fusion_layer = nn.Linear(input_dim, hidden_dim)
        elif fusion_method == 'mean':
            self.fusion_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 融合后的多尺度特征字典 {'res2': tensor, 'res3': tensor, ...}
            
        Returns:
            logits: [B, num_classes] 分类logits
        """
        # 收集所有层级的特征
        feature_list = []
        for key in ['res2', 'res3', 'res4', 'res5']:
            if key in features:
                feat = features[key]  # [B, C, H, W]
                # 全局平均池化
                feat_pooled = F.adaptive_avg_pool2d(feat, (1, 1))  # [B, C, 1, 1]
                feat_pooled = feat_pooled.view(feat_pooled.size(0), -1)  # [B, C]
                feature_list.append(feat_pooled)
        
        if not feature_list:
            raise ValueError("No valid features found for classification")
        
        # 特征融合
        if self.fusion_method == 'concat':
            # 拼接所有特征
            fused_feat = torch.cat(feature_list, dim=1)  # [B, sum(C)]
            fused_feat = self.fusion_layer(fused_feat)  # [B, hidden_dim]
            
        elif self.fusion_method == 'attention':
            # 使用注意力机制融合
            # 将所有特征堆叠为序列
            feat_stack = torch.stack(feature_list, dim=1)  # [B, num_layers, C]
            
            # 自注意力
            attn_output, _ = self.attention(feat_stack, feat_stack, feat_stack)
            
            # 平均池化
            fused_feat = attn_output.mean(dim=1)  # [B, C]
            fused_feat = self.fusion_layer(fused_feat)  # [B, hidden_dim]
            
        elif self.fusion_method == 'mean':
            # 简单平均
            feat_stack = torch.stack(feature_list, dim=1)  # [B, num_layers, C]
            fused_feat = feat_stack.mean(dim=1)  # [B, C]
            fused_feat = self.fusion_layer(fused_feat)  # [B, hidden_dim]
        
        # 分类
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        return logits


class ClassificationLoss(nn.Module):
    """
    分类损失函数
    """
    def __init__(
        self,
        num_classes: int = 2,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算分类损失
        
        Args:
            logits: [B, num_classes] 预测logits
            labels: [B] 真实标签 (0: normal, 1: tumor)
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 交叉熵损失
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(
                logits, 
                labels, 
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )
        else:
            ce_loss = F.cross_entropy(
                logits, 
                labels, 
                label_smoothing=self.label_smoothing
            )
        
        # 计算准确率
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            # 计算每个类别的准确率
            normal_acc = ((predictions == 0) & (labels == 0)).float().sum() / (labels == 0).float().sum().clamp(min=1)
            tumor_acc = ((predictions == 1) & (labels == 1)).float().sum() / (labels == 1).float().sum().clamp(min=1)
        
        return {
            'classification_loss': ce_loss,
            'classification_accuracy': accuracy,
            'normal_accuracy': normal_acc,
            'tumor_accuracy': tumor_acc,
            'predictions': predictions
        }


def extract_tumor_normal_label(mask_filename: str) -> int:
    """
    从mask文件名中提取tumor/normal标签
    
    Args:
        mask_filename: mask文件名，如 "26_Pathology_lung_tumor.png"
        
    Returns:
        label: 0 for normal, 1 for tumor
    """
    filename_lower = mask_filename.lower()
    
    # 检查是否包含tumor
    if 'tumor' in filename_lower:
        return 1  # tumor
    
    # 其他情况（stroma, normal等）都归为normal
    return 0  # normal


def extract_labels_from_batch(batched_inputs: List[Dict]) -> torch.Tensor:
    """
    从批次输入中提取tumor/normal标签
    
    Args:
        batched_inputs: 批次输入数据
        
    Returns:
        labels: [B] 标签tensor
    """
    labels = []
    
    for batch_input in batched_inputs:
        # 尝试从不同来源获取mask文件名
        mask_filename = None
        
        # 1. 从image_id获取
        if 'image_id' in batch_input:
            mask_filename = batch_input['image_id']
        
        # 2. 从file_name获取
        elif 'file_name' in batch_input:
            mask_filename = batch_input['file_name']
        
        # 3. 从instances中获取
        elif 'instances' in batch_input and batch_input['instances'] is not None:
            instances = batch_input['instances']
            if hasattr(instances, 'image_id'):
                mask_filename = instances.image_id
            elif hasattr(instances, 'file_name'):
                mask_filename = instances.file_name
        
        # 4. 从其他可能的字段获取
        elif 'filename' in batch_input:
            mask_filename = batch_input['filename']
        elif 'mask_file' in batch_input:
            mask_filename = batch_input['mask_file']
        
        if mask_filename is not None:
            label = extract_tumor_normal_label(mask_filename)
            labels.append(label)
        else:
            # 默认标签为normal
            labels.append(0)
            print(f"Warning: Could not extract label from batch input, using default (normal)")
    
    # 转换为tensor
    device = next(iter(batched_inputs[0].values())).device if batched_inputs else torch.device('cpu')
    if isinstance(device, torch.Tensor):
        device = device.device
    
    labels_tensor = torch.tensor(labels, device=device, dtype=torch.long)
    
    return labels_tensor


# 测试函数
def test_classification_head():
    """测试分类头"""
    # 创建模拟特征
    batch_size = 4
    features = {
        'res2': torch.randn(batch_size, 192, 64, 64),
        'res3': torch.randn(batch_size, 384, 32, 32),
        'res4': torch.randn(batch_size, 768, 16, 16),
        'res5': torch.randn(batch_size, 1536, 8, 8)
    }
    
    # 创建分类头
    classifier = FusionClassificationHead(
        input_dim=192 + 384 + 768 + 1536,  # 所有特征拼接后的维度
        hidden_dim=512,
        num_classes=2
    )
    
    # 前向传播
    logits = classifier(features)
    print(f"Classification logits shape: {logits.shape}")
    
    # 创建损失函数
    loss_fn = ClassificationLoss(num_classes=2)
    
    # 模拟标签
    labels = torch.tensor([0, 1, 0, 1])  # normal, tumor, normal, tumor
    
    # 计算损失
    loss_dict = loss_fn(logits, labels)
    print(f"Loss dict: {loss_dict}")


if __name__ == "__main__":
    test_classification_head()
