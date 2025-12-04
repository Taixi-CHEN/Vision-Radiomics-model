# --------------------------------------------------------
# X-Decoder with Radiomics Integration -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# Modified for radiomics integration
# --------------------------------------------------------

import logging
import time
import datetime
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union
from infinibatch import iterators

from trainer.default_trainer import DefaultTrainer

from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import MetadataCatalog

from modeling import build_model
from modeling.utils import get_class_names
from modeling.BaseModel import BaseModel
from datasets_bio import build_evaluator, build_eval_dataloader, build_train_dataloader
from utilities.distributed import is_main_process
from utilities.constants import COCO_PANOPTIC_CLASSES
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils.misc import hook_metadata, hook_switcher, hook_opt


import math
from detectron2.data import DatasetCatalog


logger = logging.getLogger(__name__)


class XDecoderRadiomicsPipeline:
    """
    Pipeline for X-Decoder with Radiomics Integration
    Extends the original XDecoderPipeline to handle radiomics features
    """
    def __init__(self, opt):
        self._opt = opt
        pretrain_path = self._opt.get('PRETRAIN_WEIGHTS_PATH', '')
        if pretrain_path:
            print(f"Pretrain weights path: {pretrain_path}")
        else:
            print("No pretrain weights specified")
        # print(self._opt['RESUME_FROM'])

    def initialize_model(self):
        model_name = "default"
        model = build_model(self._opt)
        model.train()

        if is_main_process():
            logger.info(model)

        pretrain_weights_path = self._opt.get('PRETRAIN_WEIGHTS_PATH', None)
        if pretrain_weights_path and os.path.exists(pretrain_weights_path):
            if is_main_process():
                logger.info(f"Loading pretrained weights from: {pretrain_weights_path}")
            
            # Load pretrained weights using the model's load_pretrain_weights method
            if hasattr(model, 'load_pretrain_weights'):
                try:
                    epoch, losses = model.load_pretrain_weights(pretrain_weights_path, load_similarity_module=False)
                    if is_main_process():
                        logger.info(f"Successfully loaded pretrained weights from epoch {epoch}")
                        if losses:
                            logger.info(f"Pretrain losses: {losses}")
                except Exception as e:
                    if is_main_process():
                        logger.warning(f"Failed to load pretrained weights: {e}")
                        logger.warning("Continuing with randomly initialized weights")
            else:
                if is_main_process():
                    logger.warning("Model does not support load_pretrain_weights method")
        
        # Switch to normal training mode if we loaded pretrained weights
        if hasattr(model, 'pretrain_mode'):
            model.pretrain_mode = False
            if is_main_process():
                logger.info("Switched to normal training mode")

        raw_models = {model_name: BaseModel(self._opt, model)}
        return raw_models

    def initialize_dataloader(self, opt, split='train'):
        if split == 'train':
            dataloader = build_train_dataloader(opt)
        else:
            dataloader = build_eval_dataloader(opt, split)
        return dataloader

    def initialize_trainer(self, opt, model, dataloader):
        trainer = DefaultTrainer(opt, model, dataloader)
        return trainer

    def build_dataloader(self, opt, split='train'):
        if split == 'train':
            dataloader = build_train_dataloader(opt)
        else:
            dataloader = build_eval_dataloader(opt, split)
        return dataloader

    def get_dataloaders(
        self, trainer: DefaultTrainer,
        dataset_label: str,
        is_evaluation: bool
    ) -> Union[DataLoader, iterators.CheckpointableIterator]:
        distributed = self._opt['world_size'] > 1
        if is_evaluation:
            if not hasattr(self, 'valid_loader'):
                dataloaders = build_eval_dataloader(self._opt)
                self.valid_loader = dataloaders
            else:
                dataloaders = self.valid_loader
            idx = 0 if dataset_label=='dev' else self._opt['DATASETS']['TEST'].index(dataset_label)
            dataloader = dataloaders[idx]
            self.evaluator = build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR'])
        else:
            if not hasattr(self, 'train_loader'):
                dataloader = build_train_dataloader(self._opt)
                self.train_loader = dataloader
                # logger.info(f'num of train samples: {len(dataloader)}')
            else:
                dataloader = self.train_loader
                
            # temp solution for lr scheduler
            # compute steps without calling len() on the grouped dataset
            train_names = self._opt['DATASETS']['TRAIN']
            if isinstance(train_names, str):
                train_names = [train_names]

            num_imgs = sum(len(DatasetCatalog.get(n)) for n in train_names)

            # total batch per update (either explicit TOTAL or per-gpu * world size)
            if "TRAIN" in self._opt and "BATCH_SIZE_TOTAL" in self._opt["TRAIN"]:
                batch_total = int(self._opt["TRAIN"]["BATCH_SIZE_TOTAL"])
            else:
                per_gpu = int(self._opt.get("TRAIN", {}).get("BATCH_SIZE_PER_GPU", 1))
                world   = int(self._opt.get("world_size", 1))
                batch_total = max(1, per_gpu * world)

            steps_acc   = int(self._opt.get("GRADIENT_ACCUMULATE_STEP", 1))
            steps_total = math.ceil(num_imgs / batch_total)
            steps_update = math.ceil(steps_total / max(1, steps_acc))

            self._opt.setdefault("LR_SCHEDULER_PARAMS", {})["steps_update_per_epoch"] = steps_update
            logger.info(f"num of train images: {num_imgs} | batch_total: {batch_total} | "
                        f"grad_acc: {steps_acc} | steps/epoch: {steps_total} | "
                        f"optimizer-updates/epoch: {steps_update}")

        return dataloader

    def build_train_dataloader(self, opt):
        dataloader = build_train_dataloader(opt)
        if is_main_process():
            num_imgs = len(DatasetCatalog.get(opt['DATASETS']['TRAIN'][0]))
            batch_total = opt['TRAIN']['BATCH_SIZE_TOTAL']
            steps_acc = opt['SOLVER'].get('STEPS_ACC', 1)
            steps_total = math.ceil(num_imgs / batch_total)
            steps_update = math.ceil(steps_total / max(1, steps_acc))

            self._opt.setdefault("LR_SCHEDULER_PARAMS", {})["steps_update_per_epoch"] = steps_update
            logger.info(f"num of train images: {num_imgs} | batch_total: {batch_total} | "
                        f"grad_acc: {steps_acc} | steps/epoch: {steps_total} | "
                        f"optimizer-updates/epoch: {steps_update}")

        return dataloader

    @staticmethod
    def forward_func(trainer, batch):
        """
        Forward function for training with radiomics support
        """
        # The model will automatically handle radiomics features if present in the batch
        loss = trainer.models['default'](batch)
        return loss

    def forward_step(
        self,
        trainer: DefaultTrainer,
        batch,
        grad_acc_batches: List,
        grad_acc_index: int,
        is_distributed: bool,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
        """
        Forward step with radiomics support
        """
        loss_info, sample_size_info, extra_info = {}, {}, {}
        
        # Move batch to device (including radiomics features)
        batch = move_batch_to_device(batch, self._opt['device'])
        
        # Cast to half precision if using FP16
        if self._opt['FP16']:
            batch = cast_batch_to_half(batch)
        
        # Compute loss (model will handle radiomics automatically)
        loss = trainer.compute_loss(self.forward_func, batch)
        loss_info = {k: v.detach().item() for k, v in loss.items()}
        sample_size_info = {'num_samples': len(batch)}
        # loss = sum(loss for loss in loss.values())
        if 'classification_loss' in loss:
            total_loss = loss['classification_loss']
            print(f"Using only classification loss: {total_loss.item():.4f}")
        else:
            # Fallback to sum of all losses if classification_loss not found
            total_loss = sum(loss for loss in loss.values())
            print(f"Warning: classification_loss not found, using sum of all losses: {total_loss.item():.4f}")
        
        # Backward pass and update
        # trainer.backward_loss(loss, model_names=['default'])
        trainer.backward_loss(total_loss, model_names=['default'])
        trainer.update_model(model_name='default')
        
        return loss_info, sample_size_info, extra_info

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
        save_folder,
        epoch: int = None,
    ) -> Tuple[Dict, Dict[str, float], bool]:
        """
        Evaluate model with radiomics support
        """
        model = trainer.raw_models['default'].eval()
        self._opt = hook_opt(self._opt)
        dataset_names = self._opt['DATASETS']['TEST']
        scores = {}
        summary = {}

        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            self.evaluator.reset()
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                names = get_class_names(dataset_label)
                if self._opt['MODEL']['ENCODER']['BINARY_CLASSES']:
                    names = ['target', 'background']
                model.model.metadata = MetadataCatalog.get(dataset_label)
                model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
                eval_type = model.model.metadata.evaluator_type
                if 'background' in names:
                    model.model.sem_seg_head.num_classes = len(names) - 1
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
                hook_switcher(model, dataset_label)
                total = len(eval_batch_gen)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                start_data_time = time.perf_counter()
                classification_only = hasattr(model.model, 'cls_head_fc1') and hasattr(model.model, 'cls_head_fc2')

                for idx, batch in enumerate(eval_batch_gen):
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0
                    
                    total_data_time += time.perf_counter() - start_data_time
                    start_compute_time = time.perf_counter()
                    
                    # Process batch with radiomics support
                    if hasattr(batch, '__iter__') and not isinstance(batch, (str, bytes)):
                        batch = list(batch)
                    
                    # Move batch to device if needed
                    if hasattr(batch[0], 'to'):
                        batch = [item.to(self._opt['device']) for item in batch]

                    if classification_only:
                        outputs = model.model(batch)
                    
                    # # Forward pass
                    # if eval_type in [
                    #     "grounding_refcoco",
                    #     "grounding_phrasecut",
                    #     "grounding_spatial",
                    #     "grounding_entity",
                    # ]:
                    #     outputs = model(batch, mode=eval_type)
                    #     # mask_save_dir = os.path.join(save_folder, f"masks_epoch_{epoch:03d}" if epoch else "masks")
                    #     # outputs = model.model.evaluate(batch, save_masks=True, save_dir=mask_save_dir)
                    # else:
                    #     outputs = model(batch)
                        # mask_save_dir = os.path.join(save_folder, f"masks_epoch_{epoch:03d}" if epoch else "masks")
                        # outputs = model.model.evaluate(batch, save_masks=True, save_dir=mask_save_dir)
                    
                    total_compute_time += time.perf_counter() - start_compute_time
                    start_eval_time = time.perf_counter()

                    if classification_only:
                        sample = batch[0]
                        print(f"Available fields in batch sample:")
                        for key, value in sample.items():
                            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
                        print("=" * 50)
                        # Compute simple accuracy for classification
                        labels = []
                        for sample in batch:
                            name = ''
                            if 'grounding_info' in sample and sample['grounding_info']:
                                grounding_info = sample['grounding_info']
                                if isinstance(grounding_info, list) and len(grounding_info) > 0:
                                    if 'mask_file' in grounding_info[0]:
                                        name = str(grounding_info[0]['mask_file'])
                                        print("TRUE\n")
                                elif isinstance(grounding_info, dict) and 'mask_file' in grounding_info:
                                    name = str(grounding_info['mask_file'])
                                    print("ELSE")
                            # if 'file_name' in sample:
                            #     name = str(sample['file_name'])
                            # elif 'filename' in sample:
                            #     name = str(sample['filename'])
                            # elif 'mask_file' in sample:
                            #     name = str(sample['mask_file'])
                            print(name)
                            lbl = 1 if ('tumor' in name) else 0
                            if lbl == 1:
                                print("TRUE\n")
                            else:
                                print("False\n")

                            labels.append(lbl)
                        preds = [o.get('prediction', 0) for o in outputs]
                        correct = sum(int(p == l) for p, l in zip(preds, labels))
                        scores.setdefault(dataset_label, {'classification': {'correct': 0, 'samples': 0}})
                        scores[dataset_label]['classification']['correct'] += correct
                        scores[dataset_label]['classification']['samples'] += len(labels)
                    else:
                        self.evaluator.process(batch, outputs)
                    
                    # print(labels)
                    
                    # Evaluate outputs
                    # self.evaluator.process(batch, outputs)
                    
                    total_eval_time += time.perf_counter() - start_eval_time
                    start_data_time = time.perf_counter()

                if classification_only:
                    cls = scores.get(dataset_label, {}).get('classification', None)
                    if cls and cls['samples'] > 0:
                        acc = cls['correct'] / cls['samples']
                    else:
                        acc = 0.0
                    score = {'classification': {'accuracy': acc, 'samples': cls['samples'] if cls else 0}}
                    scores[dataset_label] = score
                else:
                    score = self.evaluator.evaluate()
                    scores[dataset_label] = score
                
                # # Get evaluation results
                # score = self.evaluator.evaluate()
                # scores[dataset_label] = score
                
                if is_main_process():
                    logger.info(f"Evaluation results for {dataset_label}:")
                    logger.info(score)

        # if is_main_process():
        #     summary = {k: v for k, v in scores.items()}
        #     with open(os.path.join(save_folder, "eval_results.json"), "w") as f:
        #         json.dump(summary, f, indent=2)
        #     logger.info(f"Evaluation results saved to {save_folder}/eval_results.json")
        if is_main_process():
            summary = {k: v for k, v in scores.items()}
            # Add epoch information to summary
            if epoch is not None:
                summary['epoch'] = epoch
            
            # Generate filename based on epoch
            if epoch is not None:
                filename = f"eval_results_epoch_{epoch:03d}.json"
            else:
                filename = "eval_results.json"
            
            eval_file_path = os.path.join(save_folder, filename)
            with open(eval_file_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Evaluation results saved to {eval_file_path}")

        return scores, summary, True

    def validate_batch_format(self, batch):
        """
        Validate that the batch contains the required radiomics features
        """
        if not batch:
            return False
            
        # Check if radiomics features are present
        sample = batch[0] if isinstance(batch, list) else batch
        has_radiomics = 'radiomics' in sample
        
        if has_radiomics:
            logger.debug("Batch contains radiomics features")
            # Validate radiomics feature format
            radiomics = sample['radiomics']
            if not isinstance(radiomics, torch.Tensor):
                logger.warning("Radiomics features should be torch.Tensor")
                return False
            if len(radiomics.shape) != 1:
                logger.warning(f"Expected 1D radiomics tensor, got shape {radiomics.shape}")
                return False
        else:
            logger.debug("Batch does not contain radiomics features - using standard mode")
            
        return True

    def preprocess_batch(self, batch):
        """
        Preprocess batch to ensure radiomics features are properly formatted
        """
        if not batch:
            return batch
            
        # Validate batch format
        if not self.validate_batch_format(batch):
            logger.warning("Batch format validation failed")
            
        # Ensure radiomics features are tensors
        for item in batch:
            if 'radiomics' in item:
                if not isinstance(item['radiomics'], torch.Tensor):
                    item['radiomics'] = torch.tensor(item['radiomics'])
                    
        return batch

    def forward_step_with_validation(
        self,
        trainer: DefaultTrainer,
        batch,
        grad_acc_batches: List,
        grad_acc_index: int,
        is_distributed: bool,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
        """
        Forward step with batch validation and preprocessing
        """
        # Preprocess batch
        batch = self.preprocess_batch(batch)
        
        # Call standard forward step
        return self.forward_step(trainer, batch, grad_acc_batches, grad_acc_index, is_distributed)

    def log_radiomics_info(self, batch):
        """
        Log information about radiomics features in the batch
        """
        if not batch or 'radiomics' not in batch[0]:
            return
            
        radiomics_features = [item['radiomics'] for item in batch if 'radiomics' in item]
        if radiomics_features:
            feature_shape = radiomics_features[0].shape
            logger.info(f"Processing batch with {len(radiomics_features)} radiomics samples, "
                       f"feature dimension: {feature_shape}")

    def forward_step_with_logging(
        self,
        trainer: DefaultTrainer,
        batch,
        grad_acc_batches: List,
        grad_acc_index: int,
        is_distributed: bool,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
        """
        Forward step with radiomics logging
        """
        # Log radiomics information
        self.log_radiomics_info(batch)
        
        # Call standard forward step
        return self.forward_step(trainer, batch, grad_acc_batches, grad_acc_index, is_distributed)


# Alias for backward compatibility
XDecoderPipeline = XDecoderRadiomicsPipeline


# --------------------------------------------------------
# X-Decoder with Radiomics Integration -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# Modified for radiomics integration
# --------------------------------------------------------

# import logging
# import time
# import datetime
# import json
# import os

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from typing import Tuple, Dict, List, Union
# from infinibatch import iterators

# from trainer.default_trainer import DefaultTrainer

# from detectron2.evaluation import inference_on_dataset
# from detectron2.utils.logger import log_every_n_seconds
# from detectron2.data import MetadataCatalog

# from modeling import build_model
# from modeling.utils import get_class_names
# from modeling.BaseModel import BaseModel
# from datasets_bio import build_evaluator, build_eval_dataloader, build_train_dataloader
# from utilities.distributed import is_main_process
# from utilities.constants import COCO_PANOPTIC_CLASSES
# from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

# from .utils.misc import hook_metadata, hook_switcher, hook_opt


# import math
# from detectron2.data import DatasetCatalog


# logger = logging.getLogger(__name__)


# class XDecoderRadiomicsPipeline:
#     """
#     Pipeline for X-Decoder with Radiomics Integration
#     Extends the original XDecoderPipeline to handle radiomics features
#     """
#     def __init__(self, opt):
#         self._opt = opt
#         pretrain_path = self._opt.get('PRETRAIN_WEIGHTS_PATH', '')
#         if pretrain_path:
#             print(f"Pretrain weights path: {pretrain_path}")
#         else:
#             print("No pretrain weights specified")
#         # print(self._opt['RESUME_FROM'])

#     def initialize_model(self):
#         model_name = "default"
#         model = build_model(self._opt)
#         model.train()

#         if is_main_process():
#             logger.info(model)

#         pretrain_weights_path = self._opt.get('PRETRAIN_WEIGHTS_PATH', None)
#         if pretrain_weights_path and os.path.exists(pretrain_weights_path):
#             if is_main_process():
#                 logger.info(f"Loading pretrained weights from: {pretrain_weights_path}")
            
#             # Load pretrained weights using the model's load_pretrain_weights method
#             if hasattr(model, 'load_pretrain_weights'):
#                 try:
#                     epoch, losses = model.load_pretrain_weights(pretrain_weights_path, load_similarity_module=False)
#                     if is_main_process():
#                         logger.info(f"Successfully loaded pretrained weights from epoch {epoch}")
#                         if losses:
#                             logger.info(f"Pretrain losses: {losses}")
#                 except Exception as e:
#                     if is_main_process():
#                         logger.warning(f"Failed to load pretrained weights: {e}")
#                         logger.warning("Continuing with randomly initialized weights")
#             else:
#                 if is_main_process():
#                     logger.warning("Model does not support load_pretrain_weights method")
        
#         # Switch to normal training mode if we loaded pretrained weights
#         if hasattr(model, 'pretrain_mode'):
#             model.pretrain_mode = False
#             if is_main_process():
#                 logger.info("Switched to normal training mode")

#         raw_models = {model_name: BaseModel(self._opt, model)}
#         return raw_models

#     def initialize_dataloader(self, opt, split='train'):
#         if split == 'train':
#             dataloader = build_train_dataloader(opt)
#         else:
#             dataloader = build_eval_dataloader(opt, split)
#         return dataloader

#     def initialize_trainer(self, opt, model, dataloader):
#         trainer = DefaultTrainer(opt, model, dataloader)
#         return trainer

#     def build_dataloader(self, opt, split='train'):
#         if split == 'train':
#             dataloader = build_train_dataloader(opt)
#         else:
#             dataloader = build_eval_dataloader(opt, split)
#         return dataloader

#     def get_dataloaders(
#         self, trainer: DefaultTrainer,
#         dataset_label: str,
#         is_evaluation: bool
#     ) -> Union[DataLoader, iterators.CheckpointableIterator]:
#         distributed = self._opt['world_size'] > 1
#         if is_evaluation:
#             if not hasattr(self, 'valid_loader'):
#                 dataloaders = build_eval_dataloader(self._opt)
#                 self.valid_loader = dataloaders
#             else:
#                 dataloaders = self.valid_loader
#             idx = 0 if dataset_label=='dev' else self._opt['DATASETS']['TEST'].index(dataset_label)
#             dataloader = dataloaders[idx]
#             self.evaluator = build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR'])
#         else:
#             if not hasattr(self, 'train_loader'):
#                 dataloader = build_train_dataloader(self._opt)
#                 self.train_loader = dataloader
#                 # logger.info(f'num of train samples: {len(dataloader)}')
#             else:
#                 dataloader = self.train_loader
                
#             # temp solution for lr scheduler
#             # compute steps without calling len() on the grouped dataset
#             train_names = self._opt['DATASETS']['TRAIN']
#             if isinstance(train_names, str):
#                 train_names = [train_names]

#             num_imgs = sum(len(DatasetCatalog.get(n)) for n in train_names)

#             # total batch per update (either explicit TOTAL or per-gpu * world size)
#             if "TRAIN" in self._opt and "BATCH_SIZE_TOTAL" in self._opt["TRAIN"]:
#                 batch_total = int(self._opt["TRAIN"]["BATCH_SIZE_TOTAL"])
#             else:
#                 per_gpu = int(self._opt.get("TRAIN", {}).get("BATCH_SIZE_PER_GPU", 1))
#                 world   = int(self._opt.get("world_size", 1))
#                 batch_total = max(1, per_gpu * world)

#             steps_acc   = int(self._opt.get("GRADIENT_ACCUMULATE_STEP", 1))
#             steps_total = math.ceil(num_imgs / batch_total)
#             steps_update = math.ceil(steps_total / max(1, steps_acc))

#             self._opt.setdefault("LR_SCHEDULER_PARAMS", {})["steps_update_per_epoch"] = steps_update
#             logger.info(f"num of train images: {num_imgs} | batch_total: {batch_total} | "
#                         f"grad_acc: {steps_acc} | steps/epoch: {steps_total} | "
#                         f"optimizer-updates/epoch: {steps_update}")

#         return dataloader

#     def build_train_dataloader(self, opt):
#         dataloader = build_train_dataloader(opt)
#         if is_main_process():
#             num_imgs = len(DatasetCatalog.get(opt['DATASETS']['TRAIN'][0]))
#             batch_total = opt['TRAIN']['BATCH_SIZE_TOTAL']
#             steps_acc = opt['SOLVER'].get('STEPS_ACC', 1)
#             steps_total = math.ceil(num_imgs / batch_total)
#             steps_update = math.ceil(steps_total / max(1, steps_acc))

#             self._opt.setdefault("LR_SCHEDULER_PARAMS", {})["steps_update_per_epoch"] = steps_update
#             logger.info(f"num of train images: {num_imgs} | batch_total: {batch_total} | "
#                         f"grad_acc: {steps_acc} | steps/epoch: {steps_total} | "
#                         f"optimizer-updates/epoch: {steps_update}")

#         return dataloader

#     @staticmethod
#     def forward_func(trainer, batch):
#         """
#         Forward function for training with radiomics support
#         """
#         # The model will automatically handle radiomics features if present in the batch
#         loss = trainer.models['default'](batch)
#         return loss

#     def forward_step(
#         self,
#         trainer: DefaultTrainer,
#         batch,
#         grad_acc_batches: List,
#         grad_acc_index: int,
#         is_distributed: bool,
#     ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
#         """
#         Forward step with radiomics support
#         """
#         loss_info, sample_size_info, extra_info = {}, {}, {}
        
#         # Move batch to device (including radiomics features)
#         batch = move_batch_to_device(batch, self._opt['device'])
        
#         # Cast to half precision if using FP16
#         if self._opt['FP16']:
#             batch = cast_batch_to_half(batch)
        
#         # Compute loss (model will handle radiomics automatically)
#         loss = trainer.compute_loss(self.forward_func, batch)
#         loss_info = {k: v.detach().item() for k, v in loss.items()}
#         sample_size_info = {'num_samples': len(batch)}
#         loss = sum(loss for loss in loss.values())
        
#         # Backward pass and update
#         trainer.backward_loss(loss, model_names=['default'])
#         trainer.update_model(model_name='default')
        
#         return loss_info, sample_size_info, extra_info

#     def evaluate_model(
#         self,
#         trainer: DefaultTrainer,
#         save_folder,
#         epoch: int = None,
#     ) -> Tuple[Dict, Dict[str, float], bool]:
#         """
#         Evaluate model with radiomics support
#         """
#         model = trainer.raw_models['default'].eval()
#         self._opt = hook_opt(self._opt)
#         dataset_names = self._opt['DATASETS']['TEST']
#         scores = {}
#         summary = {}

#         for dataset_label in dataset_names:
#             torch.cuda.empty_cache()
#             eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
#             self.evaluator.reset()
#             with torch.no_grad():
#                 names = get_class_names(dataset_label)
#                 if self._opt['MODEL']['ENCODER']['BINARY_CLASSES']:
#                     names = ['target', 'background']
#                 model.model.metadata = MetadataCatalog.get(dataset_label)
#                 model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
#                 eval_type = model.model.metadata.evaluator_type
#                 if 'background' in names:
#                     model.model.sem_seg_head.num_classes = len(names) - 1
#                 model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
#                 hook_switcher(model, dataset_label)
#                 total = len(eval_batch_gen)
#                 num_warmup = min(5, total - 1)
#                 start_time = time.perf_counter()
#                 total_data_time = 0
#                 total_compute_time = 0
#                 total_eval_time = 0
#                 start_data_time = time.perf_counter()
                
#                 for idx, batch in enumerate(eval_batch_gen):
#                     if idx == num_warmup:
#                         start_time = time.perf_counter()
#                         total_data_time = 0
#                         total_compute_time = 0
#                         total_eval_time = 0
                    
#                     total_data_time += time.perf_counter() - start_data_time
#                     start_compute_time = time.perf_counter()
                    
#                     # Process batch with radiomics support
#                     if hasattr(batch, '__iter__') and not isinstance(batch, (str, bytes)):
#                         batch = list(batch)
                    
#                     # Move batch to device if needed
#                     if hasattr(batch[0], 'to'):
#                         batch = [item.to(self._opt['device']) for item in batch]
                    
#                     # Forward pass
#                     if eval_type in [
#                         "grounding_refcoco",
#                         "grounding_phrasecut",
#                         "grounding_spatial",
#                         "grounding_entity",
#                     ]:
#                         outputs = model(batch, mode=eval_type)
#                         # mask_save_dir = os.path.join(save_folder, f"masks_epoch_{epoch:03d}" if epoch else "masks")
#                         # outputs = model.model.evaluate(batch, save_masks=True, save_dir=mask_save_dir)
#                     else:
#                         outputs = model(batch)
#                         # mask_save_dir = os.path.join(save_folder, f"masks_epoch_{epoch:03d}" if epoch else "masks")
#                         # outputs = model.model.evaluate(batch, save_masks=True, save_dir=mask_save_dir)
                    
#                     total_compute_time += time.perf_counter() - start_compute_time
#                     start_eval_time = time.perf_counter()
                    
#                     # Evaluate outputs
#                     self.evaluator.process(batch, outputs)
                    
#                     total_eval_time += time.perf_counter() - start_eval_time
#                     start_data_time = time.perf_counter()
                
#                 # Get evaluation results
#                 score = self.evaluator.evaluate()
#                 scores[dataset_label] = score
                
#                 if is_main_process():
#                     logger.info(f"Evaluation results for {dataset_label}:")
#                     logger.info(score)

#         # if is_main_process():
#         #     summary = {k: v for k, v in scores.items()}
#         #     with open(os.path.join(save_folder, "eval_results.json"), "w") as f:
#         #         json.dump(summary, f, indent=2)
#         #     logger.info(f"Evaluation results saved to {save_folder}/eval_results.json")
#         if is_main_process():
#             summary = {k: v for k, v in scores.items()}
#             # Add epoch information to summary
#             if epoch is not None:
#                 summary['epoch'] = epoch
            
#             # Generate filename based on epoch
#             if epoch is not None:
#                 filename = f"eval_results_epoch_{epoch:03d}.json"
#             else:
#                 filename = "eval_results.json"
            
#             eval_file_path = os.path.join(save_folder, filename)
#             with open(eval_file_path, "w") as f:
#                 json.dump(summary, f, indent=2)
#             logger.info(f"Evaluation results saved to {eval_file_path}")

#         return scores, summary, True

#     def validate_batch_format(self, batch):
#         """
#         Validate that the batch contains the required radiomics features
#         """
#         if not batch:
#             return False
            
#         # Check if radiomics features are present
#         sample = batch[0] if isinstance(batch, list) else batch
#         has_radiomics = 'radiomics' in sample
        
#         if has_radiomics:
#             logger.debug("Batch contains radiomics features")
#             # Validate radiomics feature format
#             radiomics = sample['radiomics']
#             if not isinstance(radiomics, torch.Tensor):
#                 logger.warning("Radiomics features should be torch.Tensor")
#                 return False
#             if len(radiomics.shape) != 1:
#                 logger.warning(f"Expected 1D radiomics tensor, got shape {radiomics.shape}")
#                 return False
#         else:
#             logger.debug("Batch does not contain radiomics features - using standard mode")
            
#         return True

#     def preprocess_batch(self, batch):
#         """
#         Preprocess batch to ensure radiomics features are properly formatted
#         """
#         if not batch:
#             return batch
            
#         # Validate batch format
#         if not self.validate_batch_format(batch):
#             logger.warning("Batch format validation failed")
            
#         # Ensure radiomics features are tensors
#         for item in batch:
#             if 'radiomics' in item:
#                 if not isinstance(item['radiomics'], torch.Tensor):
#                     item['radiomics'] = torch.tensor(item['radiomics'])
                    
#         return batch

#     def forward_step_with_validation(
#         self,
#         trainer: DefaultTrainer,
#         batch,
#         grad_acc_batches: List,
#         grad_acc_index: int,
#         is_distributed: bool,
#     ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
#         """
#         Forward step with batch validation and preprocessing
#         """
#         # Preprocess batch
#         batch = self.preprocess_batch(batch)
        
#         # Call standard forward step
#         return self.forward_step(trainer, batch, grad_acc_batches, grad_acc_index, is_distributed)

#     def log_radiomics_info(self, batch):
#         """
#         Log information about radiomics features in the batch
#         """
#         if not batch or 'radiomics' not in batch[0]:
#             return
            
#         radiomics_features = [item['radiomics'] for item in batch if 'radiomics' in item]
#         if radiomics_features:
#             feature_shape = radiomics_features[0].shape
#             logger.info(f"Processing batch with {len(radiomics_features)} radiomics samples, "
#                        f"feature dimension: {feature_shape}")

#     def forward_step_with_logging(
#         self,
#         trainer: DefaultTrainer,
#         batch,
#         grad_acc_batches: List,
#         grad_acc_index: int,
#         is_distributed: bool,
#     ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
#         """
#         Forward step with radiomics logging
#         """
#         # Log radiomics information
#         self.log_radiomics_info(batch)
        
#         # Call standard forward step
#         return self.forward_step(trainer, batch, grad_acc_batches, grad_acc_index, is_distributed)


# # Alias for backward compatibility
# XDecoderPipeline = XDecoderRadiomicsPipeline

