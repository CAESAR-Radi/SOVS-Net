# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# This file is adapted from the SED/MaskFormer-style transformer predictor.

import torch
import os
from torch import nn

from detectron2.config import configurable

from .model import Aggregator
from sovs.third_party import clip
from sovs.third_party import imagenet_templates

import open_clip
import json

class SOVSNetPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        window_sizes: tuple,
        attention_type: str,
        cnext_type: str,
        kernel_size: list,
        pretrain_path: str,
    ):
        """
        Builds the SOVS-Net predictor, including text prompt embeddings and the
        cross-modal aggregation module.
        """
        super().__init__()

        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
  
        self.tokenizer = None
        if clip_pretrained.lower() in {"convnext-b", "convnext_base_w_320"}:
            name = 'convnext_base_w_320'

            local_ckpt = os.path.join(pretrain_path, "convnextb_model.pth")
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=local_ckpt,
                device=device,
                 )

            self.tokenizer = open_clip.get_tokenizer(name,cache_dir=pretrain_path)

        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "sar":
            prompt_templates = ['an SAR image of {}.',
                                'a radar image of {}.',
                                'an SAR detection of {}.',
                                'a radar return from {}.',
                                '{} as seen in an SAR image.',
                                'a patch of {}.',
                                'a region of {}.',
                                'an area of {}.',
                                'a segmented area of {}.',
                                'pixels belonging to {}.',
                                '{} pixels.',
                                'a representation of {}.',
                                'the {} class.',
                                '{}.',
                                'an example of {}.',
                                'there is {} in the scene.',
                                'this is {}.',]
        else:
            raise NotImplementedError

        all_text_features, self.synonym_counts = self.class_embeddings_all_synonyms(
            self.class_texts, prompt_templates, clip_model
        )

        all_text_features_test, _ = self.class_embeddings_all_synonyms(
            self.test_class_texts, prompt_templates, clip_model
        )

        self.register_buffer("all_text_features", all_text_features, persistent=False)
        self.register_buffer("all_text_features_test", all_text_features_test, persistent=False)


        self.clip_model = clip_model.float()
        
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            cnext_type=cnext_type,
            kernel_size=kernel_size,
            )
        self.transformer = transformer

    @classmethod
    def from_config(cls, cfg):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE
        ret["cnext_type"] = cfg.MODEL.SEM_SEG_HEAD.CNEXT_TYPE
        ret["kernel_size"] = cfg.MODEL.SEM_SEG_HEAD.KERNEL_SIZE

        ret["pretrain_path"] = cfg.MODEL.SEM_SEG_HEAD.PRETRAIN_PATH
        return ret

    def forward(self, x, vis_guidance):
        logit_scale = self.clip_model.logit_scale
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]

        all_text = self.all_text_features if self.training else self.all_text_features_test
        S, T, C, D = all_text.shape
        B = x.shape[0]

        if self.training:
            rand_s = torch.tensor([
                torch.randint(0, self.synonym_counts[c], (1,)).item()
                for c in range(C)
            ], device=all_text.device)
        else:
            rand_s = torch.zeros(C, dtype=torch.long, device=all_text.device)

        #selected = all_text[rand_s, :, torch.arange(C)]
        selected = all_text[rand_s, :, torch.arange(C, device=all_text.device)]

        text = selected.unsqueeze(0).repeat(B, 1, 1, 1)

        return self.transformer(x, text, vis, logit_scale)

    @torch.no_grad()
    def class_embeddings_all_synonyms(self, classnames, templates, clip_model):
        """
        Builds prompt-based text embeddings for class names and optional synonyms.

        Class names can contain multiple synonyms separated by '; '. The returned
        tensor is padded along the synonym dimension by repeating existing synonym
        embeddings.

        Returns:
            all_class_embeds: Tensor with shape (S, T, C, D), where S is the
                maximum number of synonyms, T is the number of prompt templates,
                C is the number of classes, and D is the embedding dimension.
            synonym_counts: Number of synonyms for each class.
        """
        synonym_lists = [c.split('; ') for c in classnames]
        synonym_counts = [len(syns) for syns in synonym_lists]
        max_synonyms = max(synonym_counts)

        all_class_embeds = []

        for synonyms in synonym_lists:
            synonym_embeds = []
            for synonym in synonyms:
                texts = [template.format(synonym) for template in templates]
                if self.tokenizer is not None:
                    tokens = self.tokenizer(texts).cuda()
                else:
                    tokens = clip.tokenize(texts).cuda()
                embed = clip_model.encode_text(tokens)
                embed /= embed.norm(dim=-1, keepdim=True)
                synonym_embeds.append(embed)

            cur_len = len(synonym_embeds)
            while len(synonym_embeds) < max_synonyms:
                synonym_embeds.append(synonym_embeds[len(synonym_embeds) % cur_len].clone())

            all_class_embeds.append(torch.stack(synonym_embeds, dim=0))

        all_class_embeds = torch.stack(all_class_embeds, dim=0)
        return all_class_embeds.permute(1, 2, 0, 3).contiguous(), synonym_counts