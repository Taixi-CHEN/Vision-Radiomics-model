import random
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from kornia.contrib import distance_transform

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .build import register_model
from detectron2.layers import ShapeSpec
from ..utils import configurable, get_class_names, get_iou, Spatial_ImageList
from ..vision.backbone import build_backbone, Backbone
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, SetCriterion, HungarianMatcher, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity
from utilities.prompt_engineering import prompt_engineering
from utilities.constants import COCO_PANOPTIC_CLASSES, BIOMED_CLASSES
from .G_Jamba.Jamba.jamba.model import Jamba
from ..modules.similarity_pretrain import SimilarityPretrainLoss, ImageRadiomicsSimilarityModule
from ..modules.classification_head import FusionClassificationHead, ClassificationLoss, extract_labels_from_batch
from PIL import Image

class RadiomicsFusionModule(nn.Module):
    """
    Module to fuse radiomics features with image features using cross attention
    Improved to handle ROI-level radiomics features with cross attention mechanism
    """
    def __init__(self, image_feature_dim, feature_channels: dict, radiomics_seq_length=8, radiomics_feature_dim=16):
        super().__init__()
        self.image_feature_dim = image_feature_dim
        self.feature_channels = feature_channels  # e.g., {"res2": C2, "res3": C3, ...}
        self.radiomics_seq_length = radiomics_seq_length
        self.radiomics_feature_dim = radiomics_feature_dim
        self.radiomics_total_dim = radiomics_seq_length * radiomics_feature_dim  # 8 * 16 = 128
        
        # ROI-level processing
        self.max_rois = 32  # Maximum number of ROIs to handle

        self.post_concat_proj = nn.ModuleDict({
            k: nn.Conv2d(c+128, c, kernel_size=1) for k, c in self.feature_channels.items()
        })
    
    def _fuse_single_image(self, feat, roi_attended, key):
        """
        Fuse radiomics features with a single image's features
        Args:
            feat: [1, C, H, W] - single image features
            roi_attended: [num_rois, radiomics_total_dim] - processed radiomics for this image
            key: feature level key
        Returns:
            [1, C, H, W] - fused features for this image
        """
        B, C, H, W = feat.shape
        num_rois = roi_attended.shape[0]
        # if torch.isnan(roi_attended).any():
        #     return feat
        if torch.isnan(roi_attended).any():
            # print(f"WARNING: Found Inf in radiomics_proj for level {key}")
            roi_attended = torch.nan_to_num(roi_attended, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"roi_attended: {torch.isnan(roi_attended).any()}")
        
        if not hasattr(self, 'radiomics_input_norm'):
            self.radiomics_input_norm = nn.LayerNorm(self.radiomics_total_dim).to(roi_attended.device)

        roi_attended_norm = self.radiomics_input_norm(roi_attended)

        if torch.isnan(roi_attended_norm).any():
            # print(f"WARNING: Found Inf in radiomics_proj for level {key}")
            roi_attended_norm = torch.nan_to_num(roi_attended_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        radiomics_avg = roi_attended_norm.mean(dim=0, keepdim=True)  # [1, C]

        radiomics_broadcast = radiomics_avg.unsqueeze(-1).unsqueeze(-1).expand(1, 128, H, W)

        if torch.isnan(radiomics_broadcast).any():
            # print(f"WARNING: Found Inf in radiomics_proj for level {key}")
            radiomics_broadcast = torch.nan_to_num(radiomics_broadcast, nan=0.0, posinf=1.0, neginf=-1.0)


        fused_feat_cat = torch.cat((feat, radiomics_broadcast), dim=1)  
        # fused_feat_cat = self.post_concat_proj['res2'](fused_feat_cat)
        fused_feat_cat = self.post_concat_proj[key](fused_feat_cat)
        
        if torch.isnan(fused_feat_cat).any():
            # print(f"WARNING: Found Inf in radiomics_proj for level {key}")
            fused_feat_cat = torch.nan_to_num(fused_feat_cat, nan=0.0, posinf=1.0, neginf=-1.0)
        return fused_feat_cat
        
    def forward(self, image_features, radiomics_features):
        """
        Args:
            image_features: dict with multi-scale features from backbone
            radiomics_features: tensor of shape [B, num_rois, seq_length*feature_dim] or [B, seq_length*feature_dim] or list of [num_rois, seq_length*feature_dim]
        Returns:
            dict with fused multi-scale features (only res2 is fused, others are unchanged)
        """
        # Initialize output with all original features
        fused_features = image_features.copy()

        available_levels = ['res2', 'res3', 'res4', 'res5']
        levels_to_fuse = [level for level in available_levels if level in image_features]
        
        # # Only fuse res2 layer, skip others
        # if 'res2' not in image_features:
        #     print("WARNING: res2 layer not found in image_features, returning original features")
        #     return fused_features
        
        # # Handle different input formats
        if isinstance(radiomics_features, list):
        #     # List format: [tensor1, tensor2, ...] where each tensor is [num_rois, seq_length*feature_dim]
            B = len(radiomics_features)

         # Process all available layers
            for level in levels_to_fuse:
                feat = image_features[level]
                print(f"Processing level {level}, shape: {feat.shape}")
                
                if len(feat.shape) == 4:
                    B_img, C, H, W = feat.shape
                elif len(feat.shape) == 2:
                    B_img, C = feat.shape
                    H, W = 1, 1  # 默认尺寸
                else:
                    raise ValueError(f"Unexpected feat shape: {feat.shape}")
                
                fused_images = []
                
                for b in range(B):
                    image_radiomics = radiomics_features[b]  # [num_rois, seq_length*feature_dim]
                    num_rois = image_radiomics.shape[0]
                    
                    # ROI self-attention for this image
                    # 直接使用原始radiomics特征，跳过self-attention
                    roi_attended = image_radiomics  # [num_rois, radiomics_total_dim]
                    
                    fused_img = self._fuse_single_image(feat[b:b+1], roi_attended, level)  # [1, C, H, W]
                    fused_images.append(fused_img)
                
                fused_features[level] = torch.cat(fused_images, dim=0)  # [B, C, H, W]
                print(f"Fused level {level}, output shape: {fused_features[level].shape}")
                
            return fused_features    
        else:
            raise ValueError(f"Unsupported radiomics features shape: {radiomics_features.shape}")


class GeneralizedSEEMRadiomics(nn.Module):
    # 类级别的Jamba模型，避免重复加载
    _jamba_model_loaded = False
    _jamba_model_instance = None

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        phrase_prob: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        train_dataset_name: str,
        interactive_mode: str,
        interactive_iter: str,
        dilation_kernel: torch.Tensor,
        train_max_iter: int,
        binary_classes: bool,
        standard_text_for_eval: bool,
        # radiomics specific
        radiomics_feature_dim: int = 16,  # Jamba output dimension
        radiomics_seq_length: int = 8,  # Sequence length for radiomics features
        jamba_model_path: str = None,  # 将使用相对路径自动构建
        pretrain_mode: bool = False,

    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            radiomics_feature_dim: int, dimension of radiomics features from Jamba
            jamba_model_path: str, path to pre-trained Jamba model
        """
        super().__init__()
        
        # 数值稳定的softmax函数
        def safe_softmax(x, dim=-1, temperature=1.0):
            """数值稳定的softmax函数，通过减去最大值来防止exp溢出"""
            x = x / temperature
            x_max = torch.max(x, dim=dim, keepdim=True)[0]
            x_shifted = x - x_max
            exp_x = torch.exp(x_shifted)
            softmax_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
            if torch.isnan(softmax_x).any():
                print("DEBUG: Found NaN in safe_softmax, applying fix")
                softmax_x = torch.nan_to_num(softmax_x, nan=1.0/softmax_x.shape[dim])
            return softmax_x
        
        self.safe_softmax = safe_softmax
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        # caption argument
        self.task_switch = task_switch
        self.phrase_prob = phrase_prob
        self.train_max_iter = train_max_iter

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = get_class_names(train_dataset_name)
        if binary_classes:
            self.train_class_names = ['target', 'background']
        self.interactive_mode = interactive_mode
        self.interactive_iter = interactive_iter

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.register_buffer("dilation_kernel", dilation_kernel)

        self.standard_text_for_eval = standard_text_for_eval
        
        # Radiomics integration
        self.radiomics_feature_dim = radiomics_feature_dim
        self.radiomics_seq_length = radiomics_seq_length
        self.radiomics_total_dim = radiomics_seq_length * radiomics_feature_dim  # 8 * 16 = 128
        self.jamba_model_path = jamba_model_path
        
        # Initialize Jamba model for radiomics processing
        self.jamba_model = None
        if jamba_model_path:
            self._load_jamba_model()
        
        # 确保jamba模型被注册为模块的一部分（即使为None）
        if self.jamba_model is not None:
            # 将jamba模型注册为子模块，这样它会被包含在state_dict中
            self.add_module('jamba_model', self.jamba_model)
        
        # Get backbone output dimensions for fusion
        backbone_output_shape = backbone.output_shape()
        # Only fuse with res2 layer
        # After fusion, res2 will have 2C channels (original + radiomics)
        # feature_channels = {'res2': backbone_output_shape['res2'].channels}  # 2C
        feature_channels = {}
        for key in ['res2', 'res3', 'res4', 'res5']:
            if key in backbone_output_shape:
                feature_channels[key] = backbone_output_shape[key].channels
        image_feature_dim = backbone_output_shape['res2'].channels
        
        print(f"DEBUG: backbone_output_shape: {backbone_output_shape}")
        print(f"DEBUG: res2 channels: {backbone_output_shape['res2'].channels}")
        print(f"DEBUG: feature_channels: {feature_channels}")
        
        # 检查实际backbone输出与配置是否匹配
        actual_res2_channels = backbone_output_shape['res2'].channels
        expected_res2_channels = 192  # 根据配置文件EMBED_DIM=192
        if actual_res2_channels != expected_res2_channels:
            print(f"WARNING: Backbone output mismatch!")
            print(f"WARNING: Expected res2 channels: {expected_res2_channels} (from config EMBED_DIM)")
            print(f"WARNING: Actual res2 channels: {actual_res2_channels} (from backbone.output_shape())")
            print(f"WARNING: This suggests backbone configuration or pretrained weights changed the structure")
            print(f"WARNING: Using actual channels ({actual_res2_channels}) for fusion module")
        
        # 为预训练使用encoder输出维度（512维）
        pretrain_image_feature_dim = 512
        
        # Initialize radiomics fusion module with improved parameters
        self.radiomics_fusion = RadiomicsFusionModule(
            image_feature_dim=image_feature_dim,
            feature_channels=feature_channels,
            radiomics_seq_length=self.radiomics_seq_length,
            radiomics_feature_dim=radiomics_feature_dim,
        )
        self.enable_radiomics_fusion = False
        # Pretrain mode setup
        self.pretrain_mode = pretrain_mode
        

        total_feature_dim = 0
        for _k, _c in feature_channels.items():
            total_feature_dim += _c
        self.cls_head_fc1 = nn.Linear(total_feature_dim, 512)
        self.cls_head_fc2 = nn.Linear(512, 256)
        self.cls_head_fc3 = nn.Linear(256, 128)
        self.cls_head_fc4 = nn.Linear(128, 2)

        
        self.cls_head_fc1.weight.requires_grad = True
        self.cls_head_fc1.bias.requires_grad = True
        print(f"Unfrozen cls_head_fc1 parameters: {self.cls_head_fc1.weight.numel() + self.cls_head_fc1.bias.numel()}")
        
        self.cls_head_fc2.weight.requires_grad = True
        self.cls_head_fc2.bias.requires_grad = True
        print(f"Unfrozen cls_head_fc2 parameters: {self.cls_head_fc2.weight.numel() + self.cls_head_fc2.bias.numel()}")
        

    def print_trainable_parameters(self):
        """Print detailed information about trainable parameters"""
        print("\n=== Trainable Parameters Summary ===")
        
        # Check each major component
        components = {
            'backbone': self.backbone,
            'sem_seg_head': self.sem_seg_head,
            'jamba_model': self.jamba_model,
            'radiomics_fusion': self.radiomics_fusion,
        }
        
        
        print("=" * 40)

    def _load_jamba_model(self):
        """Load pre-trained Jamba model for radiomics processing"""
        # 使用类级别的模型，避免重复加载
        if GeneralizedSEEMRadiomics._jamba_model_loaded:
            print("Jamba model already loaded, reusing...")
            self.jamba_model = GeneralizedSEEMRadiomics._jamba_model_instance
            return
            
        try:
            # 直接使用已经导入的Jamba类（datasets冲突已在文件开头处理）
            # Jamba类已经在文件开头通过相对导入加载
            
            # self.jamba_model = Jamba(
            #     input_dim=16,  
            #     d_model = 16,  # 使用特征维度
            #     depth=2,
            #     num_tokens=1000,  # 使用实际的token数量
            #     d_state=8,
            #     d_conv=4,
            #     heads=2,
            #     num_experts=2,
            #     num_experts_per_token=2,
            # )

            self.jamba_model = Jamba(
                dim=16,  # 主要参数
                d_model=16,  # 使用特征维度
                depth=2,
                num_tokens=1000,  # 使用实际的token数量
                d_state=8,
                d_conv=4,
                heads=4,
                num_experts=4,
                num_experts_per_token=4,
            )
            
            # 构建相对路径：当前文件在 modeling/architectures/ 目录下
            current_dir = os.path.dirname(os.path.abspath(__file__))  # modeling/architectures/
            jamba_model_path = os.path.join(current_dir, "G_Jamba", "Jamba", "jamba_feature_reshape_model.pth")
            
            # 尝试多个可能的模型文件路径
            possible_model_paths = [
                jamba_model_path,  # 基于当前文件位置的相对路径
                "/home/taixi/G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # Linux路径
                "G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # 相对路径
                "./G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # 当前目录下的相对路径
                "../G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # 上级目录下的相对路径
                "modeling/architectures/G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # 新路径
                "./modeling/architectures/G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # 当前目录下的新路径
                "../modeling/architectures/G-Jamba/Jamba/jamba_feature_reshape_model.pth",  # 上级目录下的新路径
                self.jamba_model_path if self.jamba_model_path else None,  # 配置中的路径
            ]
            
            model_loaded = False
            for model_path in possible_model_paths:
                if model_path and os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location='cpu')
                    self.jamba_model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded Jamba model from {model_path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("Warning: Jamba model file not found, using randomly initialized model")
                
            self.jamba_model.eval()  # Set to eval mode for inference
            self.add_module('jamba_model', self.jamba_model)
            
            # 保存到类级别，供其他实例使用
            GeneralizedSEEMRadiomics._jamba_model_instance = self.jamba_model
            GeneralizedSEEMRadiomics._jamba_model_loaded = True
            
            # 恢复原始sys.path
            # sys.path = original_path
            
        except Exception as e:
            print(f"Warning: Could not load Jamba model: {e}")
            print("Using randomly initialized radiomics processing")
            self.jamba_model = None
            
            

    def process_radiomics(self, radiomics_data):
        """
        Process radiomics data using Jamba model
        Handle ROI-level structure: [B, num_rois, seq_len*feat_dim] -> [B, num_rois, seq_len, feat_dim]
        Process each image separately to avoid padding issues
        Args:
            radiomics_data: tensor of shape [B, num_rois, seq_len*feat_dim] or [B, input_dim]
        Returns:
            tensor of shape [B, num_rois, seq_len*radiomics_feature_dim]
        """
        # 检查输入数据中的NaN和Inf
        if torch.isnan(radiomics_data).any():
            print(f"WARNING: Found NaN in radiomics input, shape: {radiomics_data.shape}")
            print(f"WARNING: NaN positions: {torch.isnan(radiomics_data).sum()}")
            print(f"WARNING: Input data stats - min: {radiomics_data.min():.6f}, max: {radiomics_data.max():.6f}")
            radiomics_data = torch.nan_to_num(radiomics_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isinf(radiomics_data).any():
            print(f"WARNING: Found Inf in radiomics input, shape: {radiomics_data.shape}")
            print(f"WARNING: Inf positions: {torch.isinf(radiomics_data).sum()}")
            print(f"WARNING: Input data stats - min: {radiomics_data.min():.6f}, max: {radiomics_data.max():.6f}")
            radiomics_data = torch.nan_to_num(radiomics_data, nan=0.0, posinf=1.0, neginf=-1.0)
        # Handle ROI-level structure: [B, num_rois, seq_len*feat_dim]
        if len(radiomics_data.shape) == 3:  # [B, num_rois, seq_len*feat_dim]
            B, num_rois, flat_dim = radiomics_data.shape
            seq_len = 8
            feat_dim = 16
            
            # Process each image separately to avoid padding
            processed_images = []
            
            for b in range(B):
                # Get radiomics for this image: [num_rois, seq_len*feat_dim]
                image_radiomics = radiomics_data[b]  # [num_rois, seq_len*feat_dim]
                
                # Reshape to [num_rois, seq_len, feat_dim]
                image_reshaped = image_radiomics.reshape(num_rois, seq_len, feat_dim)
                
                if self.jamba_model is not None:
                    # Process with Jamba model (frozen, no gradients)
                    # with torch.no_grad():
                    processed_image = self.jamba_model(image_reshaped)  # [num_rois, seq_len, radiomics_feature_dim]
                    print(f"processed_image: {processed_image.shape}")
                    
                    # 检查Jamba模型输出中的NaN和Inf
                    if torch.isnan(processed_image).any():
                        print(f"WARNING: Found NaN in Jamba output for image {b}, shape: {processed_image.shape}")
                        processed_image = torch.nan_to_num(processed_image, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if torch.isinf(processed_image).any():
                        print(f"WARNING: Found Inf in Jamba output for image {b}, shape: {processed_image.shape}")
                        processed_image = torch.nan_to_num(processed_image, nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    # Fallback: simple projection
                    if not hasattr(self, 'fallback_proj'):
                        self.fallback_proj = nn.Linear(feat_dim, self.radiomics_feature_dim).to(radiomics_data.device)
                    
                    # Apply projection to each timestep
                    num_rois_img, seq_len, input_dim = image_reshaped.shape
                    radiomics_flat = image_reshaped.reshape(-1, input_dim)
                    processed_flat = self.fallback_proj(radiomics_flat)
                    processed_image = processed_flat.reshape(num_rois_img, seq_len, self.radiomics_feature_dim)
                
                # Flatten to [num_rois, seq_len*radiomics_feature_dim]
                processed_flat = processed_image.reshape(num_rois, -1)
                processed_images.append(processed_flat)
            
            # Stack all processed images: [B, num_rois, seq_len*radiomics_feature_dim]
            return torch.stack(processed_images, dim=0)
            
        elif len(radiomics_data.shape) == 2:  # [B, input_dim] - legacy format
            # Handle legacy format for backward compatibility
            B, input_dim = radiomics_data.shape
            seq_len = 8
            feat_dim = 16
            
            # Reshape to [B, seq_len, feat_dim]
            if input_dim >= seq_len * feat_dim:
                radiomics_reshaped = radiomics_data[:, :seq_len * feat_dim].reshape(B, seq_len, feat_dim)
            else:
                # Pad if necessary
                padded = torch.zeros(B, seq_len * feat_dim, device=radiomics_data.device, dtype=radiomics_data.dtype)
                padded[:, :input_dim] = radiomics_data
                radiomics_reshaped = padded.reshape(B, seq_len, feat_dim)
            
            if self.jamba_model is not None:
                # Process with Jamba model (frozen, no gradients)
                # with torch.no_grad():
                processed = self.jamba_model(radiomics_reshaped)  # [B, seq_len, radiomics_feature_dim]
            else:
                if not hasattr(self, 'fallback_proj'):
                    self.fallback_proj = nn.Linear(feat_dim, self.radiomics_feature_dim).to(radiomics_data.device)
                
                B, seq_len, input_dim = radiomics_reshaped.shape
                radiomics_flat = radiomics_reshaped.reshape(-1, input_dim)
                processed_flat = self.fallback_proj(radiomics_flat)
                processed = processed_flat.reshape(B, seq_len, self.radiomics_feature_dim)
            
            # Flatten to [B, seq_len*radiomics_feature_dim]
            return processed.reshape(B, -1)
        
        else:
            raise ValueError(f"Unsupported radiomics data shape: {radiomics_data.shape}")

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        loss_weights = {'mask': {'ce': dec_cfg['CLASS_WEIGHT'], 'dice': dec_cfg['DICE_WEIGHT'], 'bce': dec_cfg['MASK_WEIGHT']},
                        'bbox': {'l1': dec_cfg['BBOX_WEIGHT'], 'giou': dec_cfg['GIOU_WEIGHT']},
                        'spatial': {'ce': dec_cfg['SCLASS_WEIGHT'], 'dice': dec_cfg['SDICE_WEIGHT'], 'bce': dec_cfg['SMASK_WEIGHT']},
                        'grounding': {'ce': dec_cfg['GCLASS_WEIGHT'], 'dice': dec_cfg['GDICE_WEIGHT'], 'bce': dec_cfg['GMASK_WEIGHT']},
                        'openimage': {'ce': dec_cfg['OCLASS_WEIGHT'], 'dice': dec_cfg['ODICE_WEIGHT'], 'bce': dec_cfg['OMASK_WEIGHT']}}

        openimage_switch = {'grounding': dec_cfg['OPENIMAGE']['GROUNDING'].get('ENABLED', False),
                            'mask': dec_cfg['OPENIMAGE'].get('ENABLED', False)}

        task_switch = {'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg['MASK'].get('ENABLED', True),
                       'spatial': dec_cfg['SPATIAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
                       'openimage': openimage_switch}

        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'grounding': dec_cfg.get('TOP_GROUNDING_LAYERS', 10),
                        'openimage': dec_cfg.get('TOP_OPENIMAGE_LAYERS', 10),
                        'spatial': dec_cfg.get('TOP_SPATIAL_LAYERS', 10)}

        spatial_cost = {"class_weight": dec_cfg['COST_SPATIAL']['CLASS_WEIGHT'],
                        "mask_weight": dec_cfg['COST_SPATIAL']['MASK_WEIGHT'],
                        "dice_weight": dec_cfg['COST_SPATIAL']['DICE_WEIGHT']}

        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        
        # Modify input_shape to reflect radiomics fusion (res2 becomes 2C)
        backbone_output_shape = backbone.output_shape()
        modified_input_shape = backbone_output_shape.copy()
        
        sem_seg_head = build_xdecoder_head(cfg, modified_input_shape, lang_encoder, extra=extra)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=loss_weights['mask']['ce'],
            cost_mask=loss_weights['mask']['bce'],
            cost_dice=loss_weights['mask']['dice'],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            spatial_cost=spatial_cost,
        )

        # init weight dict and criterion loss functions.
        losses = {'seg': [], 'openimage': []}
        if task_switch['mask']:
            losses['seg'] += ["labels"]  # 只保留labels损失
            losses['seg'] += ["labels", "masks"]  # 暂时注释掉masks损失
        if task_switch['spatial']:
            losses['seg'] += ["spatials"]  # 暂时注释掉spatial损失
        if task_switch['grounding']:
            losses['seg'] += ["groundings"]  # 暂时注释掉grounding损失
        if task_switch['openimage']:
            losses['openimage'] += ["labels_openimage", "masks"]  # 暂时注释掉openimage损失
        if task_switch['openimage']['grounding']:
            losses['openimage'] += ["groundings"]  # 暂时注释掉openimage grounding损失

        weight_dict = {}
        # 注释掉其他任务的权重
        for key, turn_on in task_switch.items():
            if turn_on:
                if isinstance(loss_weights[key], dict):
                    # HACK it should support bbox in the future
                    for key_, weight in loss_weights[key].items():
                        weight_dict["loss_{}_{}_0".format(key, key_)] = weight # NOTE: hard code for segmentation that has multiple loss
                else:
                    weight_dict["loss_{}_0".format(key)] = loss_weights[key]

        # generate full weight dict and remove not computed layers. 
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                for k, v in weight_dict.items():
                    if (i+1) > (top_x_layers[k.split('_')[1]] - 1):
                        continue
                    aux_weight_dict.update({k.replace('_0', f"_{i+1}"): v})
            weight_dict.update(aux_weight_dict)

        grd_weight = {'text': dec_cfg['GROUNDING']['TEXT_WEIGHT'], 'class': dec_cfg['GROUNDING']['CLASS_WEIGHT']}
        # generate critenrion for loss function.
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=[],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=grd_weight,
        )

        # extra logistic
        train_dataset_name = cfg['DATASETS']['TRAIN'][0] # HACK for only one training set.
        train_max_iter = dec_cfg['SPATIAL'].get('MAX_ITER', 3)
        phrase_prob = dec_cfg['CAPTION'].get('PHRASE_PROB', 0.5)
        interactive_mode = cfg['STROKE_SAMPLER']['EVAL']['MODE']
        interactive_iter = cfg['STROKE_SAMPLER']['EVAL']['MAX_ITER']

        dilation = 3
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        # Radiomics specific parameters
        radiomics_feature_dim = cfg.get('RADIOMICS', {}).get('FEATURE_DIM', 16)
        radiomics_seq_length = cfg.get('RADIOMICS', {}).get('SEQ_LENGTH', 8)
        jamba_model_path = cfg.get('RADIOMICS', {}).get('JAMBA_MODEL_PATH', None)
        
        # Pretrain specific parameters
        pretrain_mode = cfg.get('PRETRAIN_MODE', False)


        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "interactive_mode": interactive_mode,
            "interactive_iter": interactive_iter,
            "dilation_kernel": dilation_kernel,
            "train_max_iter": train_max_iter,
            "binary_classes": enc_cfg['BINARY_CLASSES'],
            "standard_text_for_eval": cfg['STANDARD_TEXT_FOR_EVAL'],
            # radiomics specific
            "radiomics_feature_dim": radiomics_feature_dim,
            "radiomics_seq_length": radiomics_seq_length,
            "jamba_model_path": jamba_model_path,
            # pretrain specific
            "pretrain_mode": pretrain_mode,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode='default'):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * "radiomics": Tensor, radiomics features (optional)
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        if self.training:
            losses = {}
            if self.task_switch['mask'] or self.task_switch['grounding'] or self.task_switch['spatial']:
                losses_seg = self.forward_seg(batched_inputs)
                losses.update(losses_seg)
            if self.task_switch['openimage'] and self.task_switch['openimage']['mask']:
                losses_openimage = self.forward_openimage(batched_inputs['openimage'])
                losses_openimage = {key.replace('mask', 'openimage'):value for key, value in losses_openimage.items()}
                losses_openimage = {key.replace('grounding', 'grounding_openimage'):value for key, value in losses_openimage.items()}
                losses.update(losses_openimage)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else: # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            if mode == 'interactive':
                return self.evaluate_interactive(batched_inputs)
            elif mode == 'interactive_grounding':
                return self.evaluate_interactive_grounding(batched_inputs)
            elif mode == 'grounding_spatial':
                return self.evaluate_grounding_sptial(batched_inputs, mode)
            elif mode in ['grounding_phrasecut', 'grounding_refcoco']:
                return self.evaluate_grounding(batched_inputs, mode)
            else:
                return self.evaluate(batched_inputs)

    
        
    def forward_seg(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.train_class_names, is_eval=False)

        extra = {}
        # mask classification target
        if "instances" in batched_inputs[0]:
            # input bounding box is checked to be correct.
            targets = self.prepare_targets(batched_inputs, images)

            if self.task_switch['grounding']:
                grounding_tokens = [x['grounding_query_embs'] for x in targets] # need to pad for more than one grounding token
                grounding_tokens = nn.utils.rnn.pad_sequence(grounding_tokens, padding_value=-1)
                non_zero_query_mask = (grounding_tokens.sum(dim=-1) == -grounding_tokens.shape[-1])
                grounding_tokens[non_zero_query_mask] = 0

                extra['grounding_tokens'] = grounding_tokens
                extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

            if self.task_switch['spatial']:
                pos_masks = [x['spatial_query']['rand_shape'].to(self.device) for x in batched_inputs]
                neg_masks = [(x['spatial_query']['rand_shape'].to(self.device) & False) for x in batched_inputs]
                fp_masks = nn.utils.rnn.pad_sequence([(x['spatial_query']['rand_shape'].to(self.device) & False) for x in batched_inputs], padding_value=False, batch_first=True)
                extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks, 'false_positive_mask': fp_masks})

        # 🔥 KEY MODIFICATION: Extract and fuse radiomics features
        features = self.backbone(images.tensor)
        
        if self.enable_radiomics_fusion == True:
            print(f"using radiomics..........................................................")
            # Process radiomics features if available
            if "radiomics" in batched_inputs[0]:
                # Process each image's radiomics separately without padding
                radiomics_tensors = [x["radiomics"].to(self.device) for x in batched_inputs]
                
                # Process each image separately
                processed_radiomics = []
                for tensor in radiomics_tensors:
                    # Process single image's radiomics
                    if len(tensor.shape) == 2:  # [num_rois, seq_len*feat_dim]
                        # Add batch dimension for processing
                        tensor_batch = tensor.unsqueeze(0)  # [1, num_rois, seq_len*feat_dim]
                        processed = self.process_radiomics(tensor_batch)
                        processed_radiomics.append(processed.squeeze(0))  # [num_rois, seq_len*feat_dim]
                    else:
                        processed_radiomics.append(tensor)
                
                # Process radiomics individually (no stacking needed)
                radiomics_features = processed_radiomics  # List of [num_rois, seq_len*feat_dim] tensors


                if self.enable_radiomics_fusion == True:
                    available_levels = ['res2', 'res3', 'res4', 'res5']
                    levels_to_fuse = [level for level in available_levels if level in features]
                    
                    if levels_to_fuse:
                        print(f"DEBUG: Found backbone layers for fusion: {levels_to_fuse}")
                        for level in levels_to_fuse:
                            print(f"DEBUG: {level} shape: {features[level].shape}")
                        # Fuse radiomics features with image features
                        features = self.radiomics_fusion(features, radiomics_features)
                    else:
                        print("WARNING: No backbone layers found in features, skipping radiomics fusion")
                        print(f"DEBUG: Available feature keys: {list(features.keys())}")

        else:
            print(f"NOt using radiomics..........................................................")
        
        
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        # forward spatial only without gradient
        if self.task_switch['spatial']:
            with torch.no_grad():
                # generate random integeter between [0,3]
                rand_iter_num = random.randint(0, self.train_max_iter)
                for i in range(rand_iter_num):
                    outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='spatial')
                    extra.update(outputs)
                    extra.update(self.prepare_next_spaital_mask(extra, batched_inputs))

        outputs = self.sem_seg_head.predictor(multi_scale_features, mask_features, extra=extra, task='seg')

        extra = {'lang_logit': self.sem_seg_head.predictor.lang_encoder.logit_scale,
                 'class_embeddings': getattr(self.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('default')),
                 'false_positive_mask': extra['false_positive_mask']}
        # bipartite matching-based loss
        self.criterion.losses = self.losses['seg'] # seg criterion losses

        if self.task_switch['mask']:
            losses = self.criterion(outputs, targets, extra)
        else:
            losses = self.criterion.forward_vlp(outputs, targets, extra)

        del outputs
        return losses

    # def evaluate(self, batched_inputs):
    def evaluate(self, batched_inputs, save_masks=True, save_dir="evaluation_masks"):
        """
        Evaluate model and optionally save predicted masks
        
        Args:
            batched_inputs: Input batch
            save_masks: Whether to save predicted masks as images
            save_dir: Directory to save mask images
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        # Backbone is frozen, run in eval mode
        with torch.no_grad():
            features = self.backbone(images.tensor)

       
        if self.enable_radiomics_fusion == True:
            print(f"using radiomics..........................................................")
            if "radiomics" in batched_inputs[0]:
                # Process each image's radiomics separately without padding
                radiomics_tensors = [x["radiomics"].to(self.device) for x in batched_inputs]
                
                # Process each image separately
                processed_radiomics = []
                for tensor in radiomics_tensors:
                    # Process single image's radiomics
                    if len(tensor.shape) == 2:  # [num_rois, seq_len*feat_dim]
                        # Add batch dimension for processing
                        tensor_batch = tensor.unsqueeze(0)  # [1, num_rois, seq_len*feat_dim]
                        processed = self.process_radiomics(tensor_batch)
                        processed_radiomics.append(processed.squeeze(0))  # [num_rois, seq_len*feat_dim]
                    else:
                        processed_radiomics.append(tensor)

                
                
                # Process radiomics individually (no stacking needed)
                radiomics_features = processed_radiomics  # List of [num_rois, seq_len*feat_dim] tensors
                available_levels = ['res2', 'res3', 'res4', 'res5']
                levels_to_fuse = [level for level in available_levels if level in features]
                
                if levels_to_fuse:
                    print(f"DEBUG: Found backbone layers for fusion: {levels_to_fuse}")
                    for level in levels_to_fuse:
                        print(f"DEBUG: {level} shape: {features[level].shape}")
                    # Fuse radiomics features with image features
                    features = self.radiomics_fusion(features, radiomics_features)
                else:
                    print("WARNING: No backbone layers found in features, skipping radiomics fusion")
                    print(f"DEBUG: Available feature keys: {list(features.keys())}")
                # Fuse radiomics features with image features
                features = self.radiomics_fusion(features, radiomics_features)

        else:
            print(f"NOt using radiomics..........................................................")
            
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        box_pred_results = outputs["pred_boxes"] if self.task_switch['bbox'] else [None for i in range(len(mask_pred_results))]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        input_size = mask_pred_results.shape[-2:]
        del outputs

        if save_masks:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving evaluation masks to: {save_dir}")

        processed_results = []
        for idx, (mask_cls_result, mask_pred_result, box_pred_result, input_per_image, image_size) in enumerate(zip(
            mask_cls_results, mask_pred_results, box_pred_results, batched_inputs, images.image_sizes
        )):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

                if save_masks:
                    sem_mask = r.cpu().numpy()  # [num_classes, H, W]
                    # Convert to single channel mask by taking argmax
                    sem_mask_single = np.argmax(sem_mask, axis=0)  # [H, W]
                    # Normalize to 0-255 range
                    sem_mask_normalized = (sem_mask_single * 255 / (sem_mask.shape[0] - 1)).astype(np.uint8)
                    sem_img = Image.fromarray(sem_mask_normalized)
                    sem_path = os.path.join(save_dir, f"semantic_mask_{idx:04d}.png")
                    sem_img.save(sem_path)
                    print(f"Saved semantic mask: {sem_path}")

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
                if save_masks:
                    panoptic_seg, segments_info = panoptic_r
                    panoptic_mask = panoptic_seg.cpu().numpy()  # [H, W]
                    # Normalize to 0-255 range
                    if panoptic_mask.max() > 0:
                        panoptic_mask_normalized = (panoptic_mask * 255 / panoptic_mask.max()).astype(np.uint8)
                    else:
                        panoptic_mask_normalized = panoptic_mask.astype(np.uint8)
                    panoptic_img = Image.fromarray(panoptic_mask_normalized)
                    panoptic_path = os.path.join(save_dir, f"panoptic_mask_{idx:04d}.png")
                    panoptic_img.save(panoptic_path)
                    print(f"Saved panoptic mask: {panoptic_path}")
            
            # instance segmentation inference
            if self.instance_on:
                if self.task_switch['bbox']:
                    box_pred_result = bbox_postprocess(box_pred_result, input_size, image_size, height, width)
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, box_pred_result)
                processed_results[-1]["instances"] = instance_r

                if save_masks:
                    instances = instance_r
                    if len(instances) > 0:
                        # Create combined instance mask
                        instance_mask = torch.zeros((height, width), dtype=torch.long, device=self.device)
                        for i, (mask, score, class_id) in enumerate(zip(instances.pred_masks, instances.scores, instances.pred_classes)):
                            if score > 0.5:  # Only save high-confidence instances
                                instance_mask[mask > 0.5] = i + 1  # Instance ID starts from 1
                        
                        instance_mask_np = instance_mask.cpu().numpy()
                        # Normalize to 0-255 range
                        if instance_mask_np.max() > 0:
                            instance_mask_normalized = (instance_mask_np * 255 / instance_mask_np.max()).astype(np.uint8)
                        else:
                            instance_mask_normalized = instance_mask_np.astype(np.uint8)
                        instance_img = Image.fromarray(instance_mask_normalized)
                        instance_path = os.path.join(save_dir, f"instance_mask_{idx:04d}.png")
                        instance_img.save(instance_path)
                        print(f"Saved instance mask: {instance_path}")
                    else:
                        print(f"No instances detected for image {idx}")

        return processed_results

    def evaluate_grounding(self, batched_inputs, mode):
        # Mirror seem_model_v1.evaluate_grounding, reusing backbone and head
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        extra = {}
        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            if self.standard_text_for_eval and 'grounding_info' in batch_per_image:
                standard_texts = []
                for grd in batch_per_image['grounding_info']:
                    mask_file = grd['mask_file'].split('.')[0].split('/')[-1]
                    target = mask_file.split('_')[-1].replace('+', ' ')
                    site = mask_file.split('_')[-2].replace('+', ' ')
                    modality = mask_file.split('_')[-3].replace('+', ' ')
                    standard_texts.append(f'{target} in {site} {modality}')
                grd_texts = standard_texts
                batch_per_image['groundings']['texts'] = standard_texts

            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            non_zero_query_mask = torch.zeros(query_emb[:,None].shape[:-1], dtype=torch.bool, device=query_emb.device)

            extra['grounding_tokens'] = query_emb[:,None]
            extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

            # Backbone is frozen, run in eval mode
            with torch.no_grad():
                features = self.backbone(images.tensor)
        

            if self.enable_radiomics_fusion == True:
                if 'radiomics' in batch_per_image:
                    radiomics_data = batch_per_image['radiomics'].to(self.device)
                    # Add batch dimension if needed
                    if len(radiomics_data.shape) == 2:
                        radiomics_data = radiomics_data.unsqueeze(0)  # [1, num_rois, seq_len*feat_dim]
                    
                    processed_radiomics = self.process_radiomics(radiomics_data)
                    # Convert to list format expected by RadiomicsFusionModule
                    radiomics_features = [processed_radiomics[0]]  # List of [num_rois, seq_len*feat_dim]
                    
                    # Check if any backbone layers exist before fusion
                    available_levels = ['res2', 'res3', 'res4', 'res5']
                    levels_to_fuse = [level for level in available_levels if level in features]
                    
                    if levels_to_fuse:
                        print(f"DEBUG: Found backbone layers for fusion: {levels_to_fuse}")
                        features = self.radiomics_fusion(features, radiomics_features)
                    else:
                        print("WARNING: No backbone layers found in features, skipping radiomics fusion")
            
            outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

            pred_gmasks = outputs['pred_gmasks'][idx]
            v_emb = outputs['pred_gtexts'][idx]
            t_emb = gtext['class_emb']

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

            temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

            matched_id = out_prob.max(0)[1]
            mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            # Ensure evaluator gets CPU tensors to avoid device mismatch with GT
            processed_results[-1]['grounding_mask'] = mask_pred_result.to('cpu')

        return processed_results

    def evaluate_grounding_sptial(self, batched_inputs, mode):
        # Mirror seem_model_v1.evaluate_grounding_sptial for spatial grounding
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"

        extra = {}
        dilation = 3
        pos_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device)).unbind(0)
        pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility).tensor
        pos_masks = (F.conv2d(pos_masks.float(), self.dilation_kernel, padding=dilation//2) > 0).unbind(0)

        neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.device) & False).unbind(0)
        neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility).tensor.unbind(0)

        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            grd_masks = []
            for idx2, anno_text in enumerate(grd_texts):
                extra.update({'spatial_query_pos_mask': [pos_masks[idx2]], 'spatial_query_neg_mask': [neg_masks[idx2]]})

                gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([anno_text[0]], name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']

                grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
                non_zero_query_mask = torch.zeros(grd_emb[:,None].shape[:-1], dtype=torch.bool, device=grd_emb.device)
                extra['grounding_tokens'] = grd_emb[:,None]
                extra['grounding_nonzero_mask'] = non_zero_query_mask.t()

                # Backbone is frozen, run in eval mode
                with torch.no_grad():
                    features = self.backbone(images.tensor)
                
                # Apply radiomics fusion in evaluation mode
                if self.enable_radiomics_fusion == True:
                    # if 'radiomics' in batch_per_image:
                    #     radiomics_data = batch_per_image['radiomics'].to(self.device)
                    #     # Add batch dimension if needed
                    #     if len(radiomics_data.shape) == 2:
                    #         radiomics_data = radiomics_data.unsqueeze(0)  # [1, num_rois, seq_len*feat_dim]
                        
                    #     processed_radiomics = self.process_radiomics(radiomics_data)
                    #     # Convert to list format expected by RadiomicsFusionModule
                    #     radiomics_features = [processed_radiomics[0]]  # List of [num_rois, seq_len*feat_dim]
                    #     features = self.radiomics_fusion(features, radiomics_features)
                    if 'radiomics' in batch_per_image:
                        radiomics_data = batch_per_image['radiomics'].to(self.device)
                        # Add batch dimension if needed
                        if len(radiomics_data.shape) == 2:
                            radiomics_data = radiomics_data.unsqueeze(0)  # [1, num_rois, seq_len*feat_dim]
                        
                        processed_radiomics = self.process_radiomics(radiomics_data)
                        # Convert to list format expected by RadiomicsFusionModule
                        radiomics_features = [processed_radiomics[0]]  # List of [num_rois, seq_len*feat_dim]
                        
                        # Check if any backbone layers exist before fusion
                        available_levels = ['res2', 'res3', 'res4', 'res5']
                        levels_to_fuse = [level for level in available_levels if level in features]
                        
                        if levels_to_fuse:
                            print(f"DEBUG: Found backbone layers for fusion: {levels_to_fuse}")
                            features = self.radiomics_fusion(features, radiomics_features)
                        else:
                            print("WARNING: No backbone layers found in features, skipping radiomics fusion")
                
                outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

                pred_gmasks = outputs['pred_gmasks'][idx]
                v_emb = outputs['pred_gtexts'][idx]
                t_emb = gtext['class_emb']

                t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
                v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

                temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
                out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

                matched_id = out_prob.max(0)[1]
                grd_masks += [pred_gmasks[matched_id,:,:]]

            mask_pred_results += [torch.cat(grd_masks)]

        for i in range(len(mask_pred_results)):
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            # Ensure evaluator gets CPU tensors to avoid device mismatch with GT
            processed_results[-1]['grounding_mask'] = mask_pred_result.to('cpu')

        return processed_results

    # Include all other methods from seem_model_v1.py (evaluate_interactive, prepare_targets, etc.)
    # For brevity, I'll include the key methods that need radiomics integration
    
    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):            
            target_dict = {}
            if self.task_switch['mask']:
                targets_per_image = batch_per_image['instances'].to(self.device)
                # pad gt
                gt_masks = targets_per_image.gt_masks.tensor
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

                gt_boxes = targets_per_image.gt_boxes.tensor
                ratio = torch.tensor([w_pad,h_pad,w_pad,h_pad]).to(gt_boxes.device)[None,:]
                gt_boxes = gt_boxes / ratio
                xc,yc,w,h = (gt_boxes[:,0] + gt_boxes[:,2])/2, (gt_boxes[:,1] + gt_boxes[:,3])/2, gt_boxes[:,2] - gt_boxes[:,0], gt_boxes[:,3] - gt_boxes[:,1]
                gt_boxes = torch.stack([xc,yc,w,h]).permute(1,0)

                target_dict.update({
                        "labels": targets_per_image.gt_classes,
                        "is_things": targets_per_image.is_things,
                        "masks": padded_masks,
                        "boxes": gt_boxes,
                        })

            if self.task_switch['spatial']:
                # prepare targets for spatial query
                target_dict['gt_spatial_masks'] = batch_per_image['spatial_query']['gt_masks']

            if self.task_switch['grounding']:
                grd_masks = batch_per_image['groundings']['masks']
                grd_texts = batch_per_image['groundings']['texts']
                grd_hash = batch_per_image['groundings']['hash']
                grd_task = batch_per_image['groundings']['mode']
                
                if len(grd_masks) == 0:
                    padded_masks = None
                else:
                    padded_masks = torch.zeros((grd_masks.shape[0], h_pad, w_pad), dtype=grd_masks.dtype, device=grd_masks.device)
                    padded_masks[:, : grd_masks.shape[1], : grd_masks.shape[2]] = grd_masks

                gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
                
                unique_hash_id = np.unique(grd_hash, return_index=True)[1]
                selected_mask = np.zeros(len(grd_hash)).astype(bool)
                selected_mask[unique_hash_id] = True

                selected_token_emb = token_emb[selected_mask]
                selected_attn_mask = tokens['attention_mask'][selected_mask]
                query_emb = selected_token_emb[selected_attn_mask.bool()]
                
                class_idx = tokens['attention_mask'].sum(dim=-1) - 1
                class_idx = torch.stack((torch.arange(len(class_idx), device=class_idx.device), class_idx)).tolist()
                class_emb = token_emb[class_idx]
                
                target_dict['grounding_masks'] = padded_masks
                target_dict['grounding_query_embs'] = query_emb
                target_dict['grounding_class_embs'] = class_emb
                target_dict['grounding_hash'] = grd_hash
                target_dict['grounding_task'] = grd_task

            new_targets.append(target_dict)
        return new_targets

    # Include other methods from seem_model_v1.py as needed
    # (semantic_inference, panoptic_inference, instance_inference, etc.)
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = self.safe_softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = self.safe_softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = self.safe_softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if box_pred is not None:
            box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

            if box_pred is not None:
                box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        if box_pred is not None:
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result

    def get_jamba_parameters(self):
        """Get Jamba model parameters for optimizer"""
        if hasattr(self, 'jamba_model') and self.jamba_model is not None:
            return list(self.jamba_model.parameters())
        return []
    
    def named_jamba_parameters(self):
        """Get named Jamba model parameters for optimizer"""
        if hasattr(self, 'jamba_model') and self.jamba_model is not None:
            return list(self.jamba_model.named_parameters())
        return []


@register_model
def get_seem_radiomics_model(cfg, **kwargs):
    return GeneralizedSEEMRadiomics(cfg)



