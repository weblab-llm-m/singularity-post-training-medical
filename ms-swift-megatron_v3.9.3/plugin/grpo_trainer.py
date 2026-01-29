# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import gc
import inspect
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list
from dacite import from_dict
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training import get_args, get_wandb_writer, training
from trl.trainer.grpo_trainer import nanstd
from vllm.distributed import parallel_state as vllm_ps

from swift.llm import RequestConfig, RolloutInferRequest, RowPreprocessor, Template, to_device
from swift.llm.infer.protocol import RolloutOutput
from swift.llm.template.template_inputs import TemplateInputs
from swift.plugin import MultiTurnScheduler, multi_turns, orms
from swift.trainers.rlhf_trainer.grpo_trainer import DataType
from swift.trainers.rlhf_trainer.utils import (FlattenedTensorBucket, aggressive_empty_cache,
                                               replace_assistant_response_with_ids, set_expandable_segments)
from swift.utils import (get_current_device, get_logger, is_last_rank, is_vllm_available, is_wandb_available,
                         remove_response)
from ..argument import MegatronArguments, MegatronRLHFArguments
from ..utils import forward_step_helper, get_padding_to
from .rlhf_mixin import MegatronRLHFTrainer
from .utils import (gather, gather_object, get_swift_datasets_provider, load_megatron_model_to_gpu,
                    load_megatron_optimizer, offload_megatron_model_to_cpu, offload_megatron_optimizer,
                    profiling_context, profiling_decorator)

if is_wandb_available():
    import wandb

logger = get_logger()


class MegatronGRPOTrainer(MegatronRLHFTrainer):

    def __init__(self, args: MegatronRLHFArguments, template: Template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client')
        super().__init__(args, template)
        self.args = args
        self.hf_model_dir = args.model_info.model_dir
        self.processing_class = self.template.processor
        self._prepare_metrics()
        self._init_grpo_params()
        self._prepare_rewards()
        self._prepare_scheduler()  # TODO
        self._prepare_rollout_engine()
        # ==================== CHORD初期化 ====================
        self._init_chord()
        # =====================================================

    def train(self, train_dataset, val_dataset, data_collator):
        # Store dataset provider for lazy resample iterator initialization
        if self.dynamic_sample:
            self._train_valid_test_dataset_provider = get_swift_datasets_provider(train_dataset, val_dataset)
            self._train_valid_test_dataset_provider.is_distributed = True
        # ==================== CHORDデータセット設定 ====================
        if self.chord_enabled:
            self._setup_chord_dataloader()
        # ==============================================================
        super().train(train_dataset, val_dataset, data_collator)

    def _init_grpo_params(self):
        args: MegatronArguments = self.args
        # distributed params
        self.world_size = torch.distributed.get_world_size()
        self.process_index = torch.distributed.get_rank()
        self.is_main_process = is_last_rank()
        self.device = get_current_device()
        # algorithm params
        self.num_generations = args.num_generations  # G in the GRPO paper
        self.beta = args.beta
        self.temperature = args.temperature
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.top_entropy_quantile = args.top_entropy_quantile
        self.importance_sampling_level = args.importance_sampling_level
        self.enable_offload = False

        # DAPO, https://arxiv.org/abs/2503.14476
        self.dynamic_sample = args.dynamic_sample
        self.max_resample_times = args.max_resample_times
        self.overlong_filter = args.overlong_filter

        # Dr. GRPO / RLOO / REINFORCE++
        self.scale_rewards = args.scale_rewards
        self.advantage_estimator = args.advantage_estimator  # TODO
        self.kl_in_reward = args.kl_in_reward  # TODO

        # Entropy mask settings, TODO
        self.log_entropy = args.log_entropy
        self.compute_entropy = self.log_entropy or self.top_entropy_quantile < 1.0

        # batch size (completion-level)
        self.generation_batch_size = args.generation_batch_size
        self.steps_per_generation = args.steps_per_generation
        self.global_batch_size = args.global_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.per_device_generation_batch_size = args.per_device_generation_batch_size

        # sampling params
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
            return_details=True)

        # CHORD, https://arxiv.org/abs/2508.11408
        # ==================== CHORDパラメータ　　====================
        # CHORD設定
        self.chord_enabled = args.chord_sft_dataset is not None
        self.chord_sft_dataset = args.chord_sft_dataset
        self.chord_sft_per_device_train_batch_size = args.chord_sft_per_device_train_batch_size
        self.chord_mu_peak = args.chord_mu_peak
        self.chord_mu_valley = args.chord_mu_valley
        self.chord_mu_warmup_steps = args.chord_mu_warmup_steps
        self.chord_mu_decay_steps = args.chord_mu_decay_steps
        self.chord_enable_phi_function = args.chord_enable_phi_function
        
        # CHORDデータローダー（後で初期化）
        self.chord_dataloader = None
        self.chord_data_iterator = None
        # =========================================================

        self._step = 0
        self._last_loaded_step = -1
        self._rollout_group = None  # Will be lazily initialized

    # CHORD, https://arxiv.org/abs/2508.11408
    # ==================== 　　CHORD　　　====================    
    def _init_chord(self):
        """Initialize CHORD-specific components"""
        if not self.chord_enabled:
            logger.info('CHORD is disabled (chord_sft_dataset not specified)')
            return
        
        logger.info(f'[CHORD] enabled: mu_peak={self.chord_mu_peak}, mu_valley={self.chord_mu_valley}, '
        f'warmup_steps={self.chord_mu_warmup_steps}, decay_steps={self.chord_mu_decay_steps}, '
        f'phi_function={self.chord_enable_phi_function}, sft_dataset={self.chord_sft_dataset}')

    # データローダー（DP並列対応）
    def _setup_chord_dataloader(self):
        """Setup CHORD SFT dataloader - DP並列対応版"""
        from swift.llm import load_dataset, EncodePreprocessor
        from torch.utils.data import DataLoader, DistributedSampler
        import torch.distributed as dist
        
        args = self.args
        dataset_string = self.chord_sft_dataset
        
        #  Megatron のDP並列ランク/サイズを使用
        try:
            dp_rank = mpu.get_data_parallel_rank()
            dp_size = mpu.get_data_parallel_world_size()
        except Exception:
            # フォールバック
            dp_rank = dist.get_rank() if dist.is_initialized() else 0
            dp_size = dist.get_world_size() if dist.is_initialized() else 1
        
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        
        if global_rank == 0:
            logger.info(f'[CHORD] Loading SFT dataset: {dataset_string}')
            logger.info(f'[CHORD] DP rank: {dp_rank}, DP size: {dp_size}')
        
        # 1. データセットをロード（全ランクで実行）
        chord_dataset, _ = load_dataset(
            dataset_string,
            strict=False,
            num_proc=1,  # ★ 分散環境では1に固定（競合回避）
            use_hf=getattr(args, 'use_hf', False)
        )
        
        if chord_dataset is None or len(chord_dataset) == 0:
            raise ValueError(f'[CHORD] Dataset {dataset_string} is empty')
        
        if global_rank == 0:
            logger.info(f'[CHORD] Loaded {len(chord_dataset)} samples')
        
        # 2. EncodePreprocessorでエンコード
        encode_preprocessor = EncodePreprocessor(template=self.template)
        chord_dataset = encode_preprocessor(
            chord_dataset, 
            num_proc=1  # ★ 分散環境では1に固定
        )
        
        if global_rank == 0:
            logger.info(f'[CHORD] Encoded {len(chord_dataset)} samples')
        
        # 3. DP並列用の分散サンプラー
        self.chord_sampler = DistributedSampler(
            chord_dataset,
            num_replicas=dp_size,  # ★DP並列サイズを使用
            rank=dp_rank,          # ★DP並列ランクを使用
            shuffle=True,
            seed=42,  # ★ 固定シードで全ランク同期
            drop_last=True,
        )
        
        # 4. DataLoaderを作成
        self.chord_dataloader = DataLoader(
            chord_dataset,
            batch_size=self.chord_sft_per_device_train_batch_size,
            sampler=self.chord_sampler,
            collate_fn=self.template.data_collator,
            drop_last=True,
            num_workers=0,
        )
        
        self.chord_data_iterator = iter(self.chord_dataloader)
        self._chord_epoch = 0
        
        if global_rank == 0:
            logger.info(f'[CHORD] Created dataloader with batch_size={self.chord_sft_per_device_train_batch_size}')
    
    # ==================== CHORD修正3: μスケジューリング ====================
    def _get_chord_mu(self) -> float:
        """
        公式のμスケジューリング実装
        Warmup → Peak → Decay to Valley
        """
        if not self.chord_enabled:
            return 0.0
        
        current_step = self._step
        warmup_steps = self.chord_mu_warmup_steps
        
        args = get_args()
        decay_steps = self.chord_mu_decay_steps
        if decay_steps is None:
            decay_steps = args.train_iters if args.train_iters else 1000
        
        mu_peak = self.chord_mu_peak
        mu_valley = self.chord_mu_valley
        
        # Warmup phase: 0 → mu_peak
        if current_step < warmup_steps:
            if warmup_steps > 0:
                return mu_peak * (current_step / warmup_steps)
            return mu_peak
        
        # Decay phase: mu_peak → mu_valley
        decay_start = warmup_steps
        decay_end = warmup_steps + decay_steps
        
        if current_step >= decay_end:
            return mu_valley
        
        decay_progress = (current_step - decay_start) / max(decay_steps, 1)
        return mu_peak - (mu_peak - mu_valley) * decay_progress
    
    # ==================== CHORD修正5: φ重み計算 ====================
    def _compute_phi_weights(
        self,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute token-wise φ weights for CHORD-φ"""
        if not self.chord_enable_phi_function:
            return torch.ones_like(per_token_logps)
        
        # 公式のφ定義: φ = p_t * (1 - p_t)
        probs = torch.exp(per_token_logps.clamp(max=0))
        phi_weights = probs * (1 - probs)
        # 正規化
        phi_weights = phi_weights / (phi_weights.mean() + 1e-8)
        return phi_weights * completion_mask
    
    # CHORD
    def _merge_chord_into_batch(
        self, 
        grpo_batch: Dict[str, Any], 
        chord_batch: Dict[str, Any],
        chord_mu: float
    ) -> Dict[str, Any]:
        """ref/old logps計算後にCHORDサンプルをGRPOバッチに混合"""
        
        sft_samples = self._convert_chord_batch_to_list(chord_batch)
        if not sft_samples:
            grpo_batch['_chord_mu'] = 0.0
            grpo_batch['_num_sft_samples'] = 0
            return grpo_batch
        
        num_grpo_samples = grpo_batch.get('num_samples', self.micro_batch_size)
        num_sft_samples = len(sft_samples)
        
        # ★★★ FIX: CUDA同期でvLLMの処理完了を保証 ★★★
        torch.cuda.synchronize()
        
        template = self.template
        args = get_args()
        with self._template_context(template):
            sft_collated = to_device(
                template.data_collator(sft_samples, padding_to=get_padding_to(args)), self.device)
        
        # ★★★ FIX: SFTバッチのテンソルをクローンしてメモリ所有権を確立 ★★★
        for key in list(sft_collated.keys()):
            if isinstance(sft_collated[key], torch.Tensor):
                sft_collated[key] = sft_collated[key].detach().clone()
        
        # GRPOの既存データを取得
        grpo_labels = grpo_batch['labels']
        grpo_input_ids = grpo_batch['input_ids']
        grpo_position_ids = grpo_batch.get('position_ids') if 'position_ids' in grpo_batch else grpo_batch.get('text_position_ids')
        grpo_completion_mask = grpo_batch['completion_mask']
        grpo_advantages = grpo_batch['advantages']
        grpo_seq_lengths = grpo_batch['seq_lengths']
        grpo_packed = grpo_batch.get('packed_seq_params')
        
        # SFTデータを取得
        sft_labels = sft_collated['labels']
        sft_input_ids = sft_collated['input_ids']
        sft_position_ids = sft_collated.get('position_ids') if 'position_ids' in sft_collated else sft_collated.get('text_position_ids')
        sft_packed = sft_collated.get('packed_seq_params')
        
        # 初期化
        merged_attention_mask = None
        
        if grpo_packed is not None and sft_packed is not None:
            # ========== padding-freeモード ==========
            grpo_expected_tokens = grpo_packed.cu_seqlens_q[-1].item()
            sft_expected_tokens = sft_packed.cu_seqlens_q[-1].item()
            
            grpo_input_ids_actual = grpo_input_ids[:, :grpo_expected_tokens]
            grpo_labels_actual = grpo_labels[:, :grpo_expected_tokens]
            grpo_completion_mask_actual = grpo_completion_mask[:, :grpo_expected_tokens]
            grpo_advantages_actual = grpo_advantages[:grpo_expected_tokens]
            
            sft_input_ids_actual = sft_input_ids[:, :sft_expected_tokens]
            sft_labels_actual = sft_labels[:, :sft_expected_tokens]
            sft_completion_mask_actual = (sft_labels_actual != -100)
            
            # ★★★ FIX: 連結後にクローンして新しいメモリ領域を確保 ★★★
            merged_input_ids = torch.cat([grpo_input_ids_actual, sft_input_ids_actual], dim=1).clone()
            merged_labels = torch.cat([grpo_labels_actual, sft_labels_actual], dim=1).clone()
            merged_completion_mask = torch.cat([grpo_completion_mask_actual, sft_completion_mask_actual], dim=1).clone()
            
            if grpo_position_ids is not None and sft_position_ids is not None:
                grpo_pos_actual = grpo_position_ids[:, :grpo_expected_tokens]
                sft_pos_actual = sft_position_ids[:, :sft_expected_tokens]
                merged_position_ids = torch.cat([grpo_pos_actual, sft_pos_actual], dim=1)
            else:
                merged_position_ids = None
            
            sft_advantages = torch.zeros(sft_expected_tokens, device=self.device, dtype=grpo_advantages.dtype)
            merged_advantages = torch.cat([grpo_advantages_actual, sft_advantages])
            
            grpo_truncated_mask = grpo_batch['truncated_mask']
            grpo_truncated_actual = grpo_truncated_mask[:, :grpo_expected_tokens]
            sft_truncated_mask = torch.zeros((1, sft_expected_tokens), device=self.device, dtype=torch.bool)
            merged_truncated_mask = torch.cat([grpo_truncated_actual, sft_truncated_mask], dim=1)
            
            sft_seq_lengths = sft_packed.cu_seqlens_q[1:] - sft_packed.cu_seqlens_q[:-1]
            merged_seq_lengths = torch.cat([grpo_seq_lengths, sft_seq_lengths])
            
            from copy import deepcopy
            merged_packed = deepcopy(grpo_packed)
            
            sft_cu_offset = sft_packed.cu_seqlens_q[1:] + grpo_expected_tokens
            merged_cu = torch.cat([grpo_packed.cu_seqlens_q, sft_cu_offset])
            merged_packed.cu_seqlens_q = merged_cu
            merged_packed.cu_seqlens_kv = merged_cu
            merged_packed.num_samples = num_grpo_samples + num_sft_samples
            
            # ★★★ FIX: max_seqlen属性を更新（形状不一致エラーの修正） ★★★
            # モデルはmax_seqlen_q/kv属性を使用してバッファサイズや処理範囲を決定する
            # deepcopyでコピーされた旧い値を新しい合計トークン数で更新する必要がある
            total_merged_tokens = grpo_expected_tokens + sft_expected_tokens
            if hasattr(merged_packed, 'max_seqlen_q'):
                # 個別シーケンスの最大長（merged_cuから計算）
                seq_lengths_merged = merged_cu[1:] - merged_cu[:-1]
                new_max_seqlen = seq_lengths_merged.max().item() if len(seq_lengths_merged) > 0 else total_merged_tokens
                merged_packed.max_seqlen_q = new_max_seqlen
                logger.debug(f"[CHORD] Updated max_seqlen_q: {new_max_seqlen}")
            if hasattr(merged_packed, 'max_seqlen_kv'):
                seq_lengths_merged = merged_cu[1:] - merged_cu[:-1]
                new_max_seqlen = seq_lengths_merged.max().item() if len(seq_lengths_merged) > 0 else total_merged_tokens
                merged_packed.max_seqlen_kv = new_max_seqlen
                logger.debug(f"[CHORD] Updated max_seqlen_kv: {new_max_seqlen}")
            
            # qkv_format属性もチェック
            if hasattr(merged_packed, 'qkv_format') and merged_packed.qkv_format is not None:
                logger.debug(f"[CHORD] qkv_format: {merged_packed.qkv_format}")
            
            grpo_token_count = grpo_expected_tokens
            
            # padding-freeモードのattention_mask処理
            grpo_attn = grpo_batch.get('attention_mask')
            sft_attn = sft_collated.get('attention_mask')
            if grpo_attn is not None and sft_attn is not None:
                grpo_attn_actual = grpo_attn[:, :grpo_expected_tokens]
                sft_attn_actual = sft_attn[:, :sft_expected_tokens]
                merged_attention_mask = torch.cat([grpo_attn_actual, sft_attn_actual], dim=1)
            
        else:
            # ========== 非padding-freeモード ==========
            grpo_seq_len = grpo_input_ids.shape[1]
            sft_seq_len = sft_input_ids.shape[1]
            
            # ★修正: attention_maskの長さも考慮してmax_seq_lenを決定
            grpo_attn = grpo_batch.get('attention_mask')
            sft_attn = sft_collated.get('attention_mask')
            grpo_truncated_mask = grpo_batch['truncated_mask']
            grpo_advantages = grpo_batch['advantages']
            
            # すべての関連テンソルの最大長を計算
            max_seq_len = max(grpo_seq_len, sft_seq_len)
            if grpo_attn is not None:
                max_seq_len = max(max_seq_len, grpo_attn.shape[-1])
            if sft_attn is not None:
                max_seq_len = max(max_seq_len, sft_attn.shape[-1])
            
            # デバッグログ
            logger.info(f"[CHORD] grpo_seq_len={grpo_seq_len}, sft_seq_len={sft_seq_len}, max_seq_len={max_seq_len}")
            if grpo_attn is not None:
                logger.info(f"[CHORD] grpo_attn.shape={grpo_attn.shape}, ndim={grpo_attn.ndim}")
            if sft_attn is not None:
                logger.info(f"[CHORD] sft_attn.shape={sft_attn.shape}, ndim={sft_attn.ndim}")
            
            # pad_token_idを取得
            pad_token_id = self.template.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0
            
            # ========== GRPOテンソルをmax_seq_lenにパディング ==========
            if grpo_seq_len < max_seq_len:
                pad_len = max_seq_len - grpo_seq_len
                grpo_input_ids = F.pad(grpo_input_ids, (0, pad_len), value=pad_token_id)
                grpo_labels = F.pad(grpo_labels, (0, pad_len), value=-100)
                grpo_completion_mask = F.pad(grpo_completion_mask, (0, pad_len), value=False)
                if grpo_position_ids is not None:
                    grpo_position_ids = F.pad(grpo_position_ids, (0, pad_len), value=0)
            
            # ========== SFTテンソルをmax_seq_lenにパディング ==========
            if sft_seq_len < max_seq_len:
                pad_len = max_seq_len - sft_seq_len
                sft_input_ids = F.pad(sft_input_ids, (0, pad_len), value=pad_token_id)
                sft_labels = F.pad(sft_labels, (0, pad_len), value=-100)
                if sft_position_ids is not None:
                    sft_position_ids = F.pad(sft_position_ids, (0, pad_len), value=0)
            
            # ========== ★重要: attention_maskを必ずmax_seq_lenに揃える（2D/4D両対応） ==========
            def _pad_attention_mask(attn_mask, target_len, pad_value=0):
                """attention_maskを指定長にパディング（2D/3D/4D対応）
                
                2D: [batch, seq_len] → 最後の次元のみパディング
                3D: [batch, 1, seq_len] → 最後の次元のみパディング
                4D: [batch, 1, seq_len, seq_len] → 最後の2次元両方をパディング（causal mask用）
                """
                if attn_mask is None:
                    return None
                
                current_len = attn_mask.shape[-1]
                if current_len == target_len:
                    return attn_mask
                
                ndim = attn_mask.ndim
                
                if current_len < target_len:
                    pad_len = target_len - current_len
                    
                    if ndim == 2:
                        # [batch, seq_len]
                        attn_mask = F.pad(attn_mask, (0, pad_len), value=pad_value)
                    elif ndim == 3:
                        # [batch, 1, seq_len]
                        attn_mask = F.pad(attn_mask, (0, pad_len), value=pad_value)
                    elif ndim == 4:
                        # [batch, heads, seq_len, seq_len] - causal mask
                        # 最後の2次元をパディング: (left, right, top, bottom)
                        attn_mask = F.pad(attn_mask, (0, pad_len, 0, pad_len), value=pad_value)
                    else:
                        # その他の次元: 最後の次元のみパディング
                        attn_mask = F.pad(attn_mask, (0, pad_len), value=pad_value)
                    
                    logger.debug(f"[CHORD] attention_mask padded: {current_len} -> {attn_mask.shape[-1]}")
                    
                elif current_len > target_len:
                    # トランケート
                    if ndim == 4:
                        # 4D: 最後の2次元をトランケート
                        attn_mask = attn_mask[..., :target_len, :target_len]
                    else:
                        # 2D/3D: 最後の次元のみトランケート
                        attn_mask = attn_mask[..., :target_len]
                    
                    logger.debug(f"[CHORD] attention_mask truncated: {current_len} -> {attn_mask.shape[-1]}")
                
                return attn_mask
            
            # GRPOのattention_maskをパディング
            grpo_attn = _pad_attention_mask(grpo_attn, max_seq_len, pad_value=0)
            
            # SFTのattention_maskをパディング
            sft_attn = _pad_attention_mask(sft_attn, max_seq_len, pad_value=0)
            
            if grpo_attn is not None:
                logger.info(f"[CHORD] grpo_attn after padding: shape={grpo_attn.shape}")
            if sft_attn is not None:
                logger.info(f"[CHORD] sft_attn after padding: shape={sft_attn.shape}")
            
            # ========== truncated_maskをmax_seq_lenに揃える ==========
            if grpo_truncated_mask is not None:
                grpo_trunc_len = grpo_truncated_mask.shape[-1]
                if grpo_trunc_len < max_seq_len:
                    grpo_truncated_mask = F.pad(grpo_truncated_mask, (0, max_seq_len - grpo_trunc_len), value=False)
                elif grpo_trunc_len > max_seq_len:
                    grpo_truncated_mask = grpo_truncated_mask[..., :max_seq_len]
            
            # ========== advantagesをmax_seq_lenに揃える ==========
            if grpo_advantages.dim() == 2:
                grpo_adv_len = grpo_advantages.shape[-1]
                if grpo_adv_len < max_seq_len:
                    grpo_advantages = F.pad(grpo_advantages, (0, max_seq_len - grpo_adv_len), value=0.0)
                elif grpo_adv_len > max_seq_len:
                    grpo_advantages = grpo_advantages[..., :max_seq_len]
            
            # ========== completion_maskはパディング後のsft_labelsから計算 ==========
            sft_completion_mask = (sft_labels != -100)
            
            # ========== バッチ次元で連結（dim=0） ==========
            # ★★★ FIX: 連結後にクローンして新しいメモリ領域を確保 ★★★
            merged_input_ids = torch.cat([grpo_input_ids, sft_input_ids], dim=0).clone()
            merged_labels = torch.cat([grpo_labels, sft_labels], dim=0).clone()
            merged_completion_mask = torch.cat([grpo_completion_mask, sft_completion_mask], dim=0).clone()
            
            if grpo_position_ids is not None and sft_position_ids is not None:
                merged_position_ids = torch.cat([grpo_position_ids, sft_position_ids], dim=0)
            else:
                merged_position_ids = None
            
            # ========== attention_mask連結（最終確認付き、4D対応） ==========
            if grpo_attn is not None and sft_attn is not None:
                # 最終サイズ確認（シェイプの不一致をチェック）
                logger.info(f"[CHORD] Final check: grpo_attn.shape={grpo_attn.shape}, sft_attn.shape={sft_attn.shape}")
                
                # シェイプが完全に一致しているか確認（dim=0以外）
                shapes_match = True
                for i in range(1, len(grpo_attn.shape)):
                    if grpo_attn.shape[i] != sft_attn.shape[i]:
                        shapes_match = False
                        logger.warning(f"[CHORD] Dimension {i} mismatch: grpo={grpo_attn.shape[i]}, sft={sft_attn.shape[i]}")
                
                if not shapes_match:
                    # 強制的に揃える（フォールバック）- 4D対応
                    target_len = max(grpo_attn.shape[-1], sft_attn.shape[-1])
                    logger.warning(f"[CHORD] Applying fallback padding to target_len={target_len}")
                    
                    # ヘルパー関数を再利用（スコープ内で定義済み）
                    grpo_attn = _pad_attention_mask(grpo_attn, target_len, pad_value=0)
                    sft_attn = _pad_attention_mask(sft_attn, target_len, pad_value=0)
                    
                    logger.info(f"[CHORD] After fallback: grpo_attn.shape={grpo_attn.shape}, sft_attn.shape={sft_attn.shape}")
                
                # 連結
                try:
                    merged_attention_mask = torch.cat([grpo_attn, sft_attn], dim=0)
                    logger.info(f"[CHORD] merged_attention_mask.shape={merged_attention_mask.shape}")
                except RuntimeError as e:
                    # それでも失敗した場合は、すべての次元を詳細にログ出力
                    logger.error(f"[CHORD] attention_mask concat failed: {e}")
                    logger.error(f"[CHORD] grpo_attn.shape={grpo_attn.shape}, sft_attn.shape={sft_attn.shape}")
                    # 最終手段: None に設定
                    merged_attention_mask = None
            else:
                merged_attention_mask = None
            
            # ========== advantages連結 ==========
            if grpo_advantages.dim() == 2:
                sft_advantages = torch.zeros((num_sft_samples, max_seq_len), device=self.device, dtype=grpo_advantages.dtype)
                merged_advantages = torch.cat([grpo_advantages, sft_advantages], dim=0)
            else:
                grpo_adv_len = grpo_advantages.shape[0]
                expected_len = max_seq_len * num_grpo_samples
                if grpo_adv_len < expected_len:
                    grpo_advantages = F.pad(grpo_advantages, (0, expected_len - grpo_adv_len), value=0.0)
                elif grpo_adv_len > expected_len:
                    grpo_advantages = grpo_advantages[:expected_len]
                sft_advantages = torch.zeros(max_seq_len * num_sft_samples, device=self.device, dtype=grpo_advantages.dtype)
                merged_advantages = torch.cat([grpo_advantages, sft_advantages])
            
            # ========== truncated_mask連結 ==========
            sft_truncated_mask = torch.zeros((num_sft_samples, max_seq_len), device=self.device, dtype=torch.bool)
            merged_truncated_mask = torch.cat([grpo_truncated_mask, sft_truncated_mask], dim=0)
            
            # ========== seq_lengths連結 ==========
            grpo_seq_lengths = grpo_batch['seq_lengths']
            if grpo_seq_lengths.dim() == 0:
                grpo_seq_lengths = grpo_seq_lengths.unsqueeze(0)
            sft_seq_lengths = torch.tensor([sft_seq_len] * num_sft_samples, device=self.device)
            merged_seq_lengths = torch.cat([grpo_seq_lengths, sft_seq_lengths])
            
            merged_packed = None
            grpo_token_count = max_seq_len

        # ========== 新しい辞書を作成 ==========
        merged_batch = {}
        for k, v in grpo_batch.items():
            merged_batch[k] = v

        merged_batch['labels'] = merged_labels
        merged_batch['input_ids'] = merged_input_ids
        merged_batch['completion_mask'] = merged_completion_mask
        merged_batch['advantages'] = merged_advantages
        merged_batch['truncated_mask'] = merged_truncated_mask
        merged_batch['seq_lengths'] = merged_seq_lengths
        merged_batch['packed_seq_params'] = merged_packed
        merged_batch['num_samples'] = num_grpo_samples + num_sft_samples
        
        # ★★★ FIX: 形状整合性の検証 ★★★
        logger.info(f"[CHORD] Merged batch shapes - input_ids: {merged_input_ids.shape}, labels: {merged_labels.shape}")
        if merged_packed is not None:
            logger.info(f"[CHORD] Merged cu_seqlens_q: {merged_packed.cu_seqlens_q.tolist()}, num_samples: {merged_packed.num_samples}")
            if hasattr(merged_packed, 'max_seqlen_q'):
                logger.info(f"[CHORD] max_seqlen_q: {merged_packed.max_seqlen_q}")
        
        # input_idsとlabelsの形状が一致するか確認
        if merged_input_ids.shape != merged_labels.shape:
            logger.error(f"[CHORD] CRITICAL: Shape mismatch! input_ids: {merged_input_ids.shape}, labels: {merged_labels.shape}")
            raise ValueError(f"Shape mismatch between input_ids {merged_input_ids.shape} and labels {merged_labels.shape}")

        if merged_position_ids is not None:
            if 'text_position_ids' in grpo_batch:
                merged_batch['text_position_ids'] = merged_position_ids
            else:
                merged_batch['position_ids'] = merged_position_ids

        if merged_attention_mask is not None:
            merged_batch['attention_mask'] = merged_attention_mask
        
        # ★★★ 重要: input_ids と attention_mask のseq_len一致を最終確認 ★★★
        # Qwen3-Next等のモデルは attention_mask を hidden_states に直接乗算するため、
        # サイズが一致しないとRuntimeErrorが発生する
        final_input_ids = merged_batch['input_ids']
        final_attn_mask = merged_batch.get('attention_mask')
        
        if final_attn_mask is not None:
            input_seq_len = final_input_ids.shape[1]
            attn_seq_len = final_attn_mask.shape[-1]
            
            if input_seq_len != attn_seq_len:
                logger.warning(f"[CHORD] input_ids.shape[1]={input_seq_len} != attention_mask.shape[-1]={attn_seq_len}, fixing...")
                
                # attention_maskのseq_lenに合わせてinput_idsをパディング（またはトランケート）
                if input_seq_len < attn_seq_len:
                    pad_len = attn_seq_len - input_seq_len
                    pad_token_id = self.template.tokenizer.pad_token_id or 0
                    
                    # input_ids, labels, completion_mask, truncated_mask をパディング
                    merged_batch['input_ids'] = F.pad(final_input_ids, (0, pad_len), value=pad_token_id)
                    merged_batch['labels'] = F.pad(merged_batch['labels'], (0, pad_len), value=-100)
                    merged_batch['completion_mask'] = F.pad(merged_batch['completion_mask'], (0, pad_len), value=False)
                    merged_batch['truncated_mask'] = F.pad(merged_batch['truncated_mask'], (0, pad_len), value=False)
                    
                    if 'position_ids' in merged_batch and merged_batch['position_ids'] is not None:
                        merged_batch['position_ids'] = F.pad(merged_batch['position_ids'], (0, pad_len), value=0)
                    if 'text_position_ids' in merged_batch and merged_batch['text_position_ids'] is not None:
                        merged_batch['text_position_ids'] = F.pad(merged_batch['text_position_ids'], (0, pad_len), value=0)
                    
                    logger.info(f"[CHORD] Padded input_ids from {input_seq_len} to {attn_seq_len}")
                    
                elif input_seq_len > attn_seq_len:
                    # attention_maskをinput_idsに合わせてパディング
                    pad_len = input_seq_len - attn_seq_len
                    attn_mask = final_attn_mask
                    ndim = attn_mask.ndim
                    
                    if ndim == 4:
                        attn_mask = F.pad(attn_mask, (0, pad_len, 0, pad_len), value=0)
                    else:
                        attn_mask = F.pad(attn_mask, (0, pad_len), value=0)
                    
                    merged_batch['attention_mask'] = attn_mask
                    logger.info(f"[CHORD] Padded attention_mask from {attn_seq_len} to {input_seq_len}")
        
        # CHORDメタデータを追加
        merged_batch['_chord_mu'] = chord_mu
        merged_batch['_num_grpo_samples'] = num_grpo_samples
        merged_batch['_num_sft_samples'] = num_sft_samples
        merged_batch['_grpo_token_count'] = grpo_token_count
        
        return merged_batch

    def _prepare_rollout_engine(self):
        args = self.args
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_vllm = args.use_vllm
        self.async_generate = args.async_generate  # TODO
        self.vllm_use_async_engine = False
        self.enable_offload = False
        self.use_gym_env = False
        self.enable_server_multi_turn = False  # TODO
        # for multi-turn server, maybe the num of rollout outputs is not equal to the num of rollout inputs
        assert self.use_vllm
        if not is_vllm_available():
            raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                              'Please install vLLM with `pip install vllm -U` to use it.')
        if self.vllm_mode == 'server':
            pass
        elif self.vllm_mode == 'colocate':
            if not self.world_size % self.vllm_tensor_parallel_size == 0:
                raise ValueError(f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                                 f'({self.world_size}) evenly.')

            self.enable_offload = self.args.offload_model or self.args.offload_optimizer
            context = self.offload_context if self.enable_offload else nullcontext

            with context():
                set_expandable_segments(False)
                self.engine = self.prepare_vllm()
                if self.args.sleep_level > 0:
                    self.engine.engine.sleep(self.args.sleep_level)
                set_expandable_segments(True)
        else:
            raise ValueError(f'Invalid vllm_mode: {self.vllm_mode}')

    def prepare_vllm(self):
        from swift.llm.infer.infer_engine import GRPOVllmEngine
        args = self.args
        max_num_seqs = self.per_device_generation_batch_size * self.vllm_tensor_parallel_size
        vllm_template = copy(self.template)
        vllm_template.padding_free = False
        engine = GRPOVllmEngine(
            self.hf_model_dir,
            args.torch_dtype,
            model_type=args.model_type,
            use_async_engine=False,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            enable_prefix_caching=self.args.vllm_enable_prefix_caching,
            max_num_seqs=max_num_seqs,
            enforce_eager=self.args.vllm_enforce_eager,
            limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
            enable_sleep_mode=self.args.sleep_level > 0,
            max_model_len=self.args.vllm_max_model_len,
            seed=self.process_index // self.vllm_tensor_parallel_size,
            disable_cascade_attn=self.args.vllm_disable_cascade_attn,
            load_format='dummy',
            template=vllm_template,
            distributed_executor_backend='external_launcher',
        )
        if self.vllm_tensor_parallel_size > 1:
            self.vllm_tp_group = vllm_ps.get_tp_group().device_group
        self._buffered_inputs = None
        return engine

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Handle LoRA: merge adapters before exporting weights
        is_lora_training = self.args.train_type == 'lora'

        try:
            if is_lora_training:
                self.merge_lora_adapters()

            # Export and load weights incrementally to avoid memory spikes
            self._export_and_load_weights()

        finally:
            # Unmerge adapters to restore training state
            if is_lora_training:
                self.unmerge_lora_adapters()

        # Reset prefix cache
        if self.vllm_mode == 'server' and self.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == 'colocate':
            self.engine.engine.reset_prefix_cache()

    @property
    def bridge(self):
        if self._bridge is None:
            self._bridge = self.args.megatron_model_meta.bridge_cls(disable_tqmd=True)
        return self._bridge

    def _export_and_load_weights(self):
        """
        Export weights from Megatron models and load to vLLM incrementally.

        For colocate mode: llm_model.load_weights accepts an iterator, so pass it directly.
        For server mode: Process weights in buckets to avoid memory spikes.
        """
        # Export weights returns an iterator
        with profiling_context(self, 'export_weights'):
            weight_iterator = self.bridge.export_weights(self.unwrapped_models)

        if self.vllm_mode == 'colocate':
            # Colocate mode: load_weights supports iterator, pass directly
            llm_model = self.engine.inner_model
            llm_model.load_weights(weight_iterator)
        elif self.vllm_mode == 'server':
            # Server mode: process in buckets and sync with flattened tensors
            self._load_weights_to_server_in_buckets(weight_iterator)

    def _load_weights_to_server_in_buckets(self, weight_iterator):
        """
        Load weights to vLLM server in buckets using FlattenedTensorBucket.

        Args:
            weight_iterator: Iterator of (name, tensor) tuples from export_weights
        """
        # Get bucket size from environment or use default
        bucket_size_mb = int(os.environ.get('SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE', 512))
        bucket_size_bytes = bucket_size_mb * 1024 * 1024

        current_bucket = []
        current_size = 0

        for name, param in weight_iterator:
            param_size = param.numel() * param.element_size()
            current_bucket.append((name, param))
            current_size += param_size

            # If adding this param would exceed bucket size, process current bucket first
            if current_size > bucket_size_bytes and current_bucket:
                self._sync_bucket_to_server(current_bucket)
                current_bucket = []
                current_size = 0

        # Process remaining parameters in the last bucket
        if current_bucket:
            self._sync_bucket_to_server(current_bucket)

    def _sync_bucket_to_server(self, bucket_params: List[Tuple[str, torch.Tensor]]):
        """
        Synchronize a bucket of parameters to vLLM server using flattened tensors.

        Args:
            bucket_params: List of (name, tensor) tuples to sync
        """
        if not bucket_params or not self.is_main_process:
            return

        # Create FlattenedTensorBucket for efficient transfer
        bucket = FlattenedTensorBucket(named_tensors=bucket_params)
        metadatas = bucket.get_metadata()
        flattened_tensor = bucket.get_flattened_tensor()

        # Directly call vllm_client to update weights
        self.vllm_client.update_flattened_params(metadatas, flattened_tensor)

        # Clean up to free memory immediately
        del bucket, metadatas, flattened_tensor

    def _prepare_rewards(self):
        # TODO: reward model
        args = self.args
        reward_funcs = args.reward_funcs
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        # initilize reward functions
        if reward_funcs:
            for i, reward_func in enumerate(reward_funcs):
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.plugin')

        # get reward name for logging
        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        # set reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32).to(self.device)
        else:
            self.reward_weights = torch.ones(len(self.reward_func_names), dtype=torch.float32).to(self.device)

        # TODO: reward models
        self.reward_model_plugins = [None] * len(self.reward_funcs)

        assert self.reward_funcs, 'reward_funcs is not set'

    def _prepare_scheduler(self):
        """Prepare multi-turn scheduler"""
        args = self.args

        self.multi_turn_scheduler = None
        if not hasattr(args, 'multi_turn_scheduler'):
            return

        if args.multi_turn_scheduler:
            if isinstance(args.multi_turn_scheduler, str):
                assert args.multi_turn_scheduler in multi_turns
                multi_turn_scheduler = multi_turns[args.multi_turn_scheduler](max_turns=args.max_turns)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
            else:
                assert isinstance(args.multi_turn_scheduler, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = args.multi_turn_scheduler

    def _get_rollout_group(self):
        """
        Get or create the rollout process group (TP×PP×CP).

        The rollout group is used for:
        1. Data slicing: distributing rollout data across ranks with same data samples
        2. Gather operations: collecting results from ranks with same data samples

        Note: Groups are created per data parallel index, containing TP×PP×CP ranks each.
        This follows Megatron's data_iterator logic where same data_parallel_rank processes
        identical data samples.

        Key insight: ranks with the SAME data parallel index process the SAME data samples
        and must coordinate for rollout data distribution.
        Megatron rank order: TP → CP → EP → DP → PP
        """
        if self._rollout_group is not None:
            return self._rollout_group

        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            # No CP, use the standard MODEL_PARALLEL_GROUP
            self._rollout_group = mpu.get_model_parallel_group()
            return self._rollout_group

        # Use RankGenerator to create rollout groups following Megatron-LM logic
        global_rank = torch.distributed.get_rank()

        # Get parallel dimensions
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()

        # Create RankGenerator following Megatron-LM pattern
        # Order: tp-cp-ep-dp-pp (default in Megatron-LM)
        decoder_rank_generator = mpu.RankGenerator(
            tp=tp_size,
            ep=1,
            dp=dp_size,
            pp=pp_size,
            cp=cp_size,
            order='tp-cp-ep-dp-pp',
            rank_offset=0,
        )

        # Create rollout groups based on data consistency from data_iterator
        # Same data_parallel_rank processes same data - group ranks with same DP index
        if not hasattr(self, '_rollout_groups_created'):
            # Use 'tp-cp-ep-pp' to get groups with same DP index (DP is excluded from variation)
            dp_groups = decoder_rank_generator.get_ranks('tp-cp-ep-pp')
            for dp_group_ranks in dp_groups:
                # Sort for consistency
                dp_group_ranks = sorted(dp_group_ranks)
                group = torch.distributed.new_group(ranks=dp_group_ranks, group_desc='ROLLOUT_GROUP')

                if global_rank in dp_group_ranks:
                    self._rollout_group = group
            self._rollout_groups_created = True

        return self._rollout_group

    def _init_resample_data_iterator(self):
        """
        Initialize an independent data iterator for dynamic resampling (lazy initialization).

        This method is called lazily during the first dynamic resampling, ensuring that
        pretrain() has already called initialize_megatron() to properly set up all args.
        Uses a different seed (args.seed + 1) to avoid overlapping with training samples.

        Note: pretrain() will automatically reset the random seed back to args.seed
        after this method completes, so we don't need manual state restoration.

        Args:
            train_valid_test_dataset_provider: Dataset provider function

        Returns:
            train_data_iterator: Independent data iterator with different random seed
        """
        from megatron.training.training import build_train_valid_test_data_iterators
        from megatron.training.initialize import _set_random_seed
        from megatron.training import training
        training.cyclic_iter = self._origin_cyclic_iter
        args = get_args()

        train_valid_test_dataset_provider = self._train_valid_test_dataset_provider
        # Use different seed for resample iterator (offset by 1 to avoid overlap)
        resample_seed = getattr(args, 'seed', 42) + 1
        try:
            # Set new seed for resample iterator creation
            _set_random_seed(
                resample_seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
            )

            # Build data iterators with new seed
            # TODO: VPP (Virtual Pipeline Parallelism)
            resample_data_iterator, _, _ = (build_train_valid_test_data_iterators(train_valid_test_dataset_provider))
        finally:
            # Restore original random states to avoid affecting training
            _set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
            )
        return resample_data_iterator

    def _convert_chord_batch_to_list(self, chord_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """DataLoaderからのバッチを個別サンプルのリストに変換"""
        if chord_batch is None:
            return []
        
        batch_size = None
        for key in ['input_ids', 'labels', 'attention_mask']:
            if key in chord_batch and chord_batch[key] is not None:
                val = chord_batch[key]
                if isinstance(val, torch.Tensor):
                    batch_size = val.shape[0]
                elif isinstance(val, list):
                    batch_size = len(val)
                break
        
        if batch_size is None or batch_size == 0:
            return []
        
        samples = []
        for i in range(batch_size):
            sample = {}
            for key, val in chord_batch.items():
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    # ★修正: Tensorをリストに変換
                    sample[key] = val[i].tolist()
                elif isinstance(val, list):
                    sample[key] = val[i]
                else:
                    sample[key] = val
            
            # lengthキーを計算して追加
            if 'length' not in sample:
                if 'input_ids' in sample:
                    ids = sample['input_ids']
                    sample['length'] = len(ids) if isinstance(ids, list) else 1
            
            samples.append(sample)
        
        return samples

    def _get_next_chord_batch_synchronized(self) -> Dict[str, Any]:
        """
        全DPランクで同期してバッチを取得
        ★重要: PPステージに関係なく、すべてのランクで呼び出す
        """
        if self.chord_dataloader is None:
            return None
        
        try:
            batch = next(self.chord_data_iterator)
        except StopIteration:
            self._chord_epoch = getattr(self, '_chord_epoch', 0) + 1
            if hasattr(self, 'chord_sampler') and self.chord_sampler is not None:
                self.chord_sampler.set_epoch(self._chord_epoch)
            self.chord_data_iterator = iter(self.chord_dataloader)
            batch = next(self.chord_data_iterator)
        
        # テンソルをデバイスに移動
        if isinstance(batch, dict):
            processed = {}
            for k, v in batch.items():
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    processed[k] = v.to(self.device)
                else:
                    processed[k] = v
            return processed
        
        return batch

    def _replace_data_iterator(self, data_iterator, model):
        if self._step % self.steps_per_generation == 0:
            num_iters_per_step = self.get_num_iters_per_step()
            rollout_batch = []
            for _ in range(num_iters_per_step):
                rollout_batch.extend(next(data_iterator))
            micro_batch_data = self._generate_and_score_completions(rollout_batch)
            num_mini_batch = self.global_batch_size // (self.micro_batch_size * mpu.get_data_parallel_world_size())
            mini_batch_data = [
                micro_batch_data[i:i + num_mini_batch] for i in range(0, len(micro_batch_data), num_mini_batch)
            ]
            assert len(mini_batch_data) == self.steps_per_generation
            self._buffered_inputs = mini_batch_data
        inputs = self._buffered_inputs[self._step % self.steps_per_generation]
        self._step += 1
        return RerunDataIterator(iter(inputs))

    def _generate_and_score_completions(self, batch):
        # Get or create the rollout group (TP×PP×CP)
        args = get_args()

        rollout_group = self._get_rollout_group()

        rollout_batch = self.get_local_rollout_batch(batch)

        rollout_batch = self._generate_completions(rollout_batch)

        rewards_per_func = self._score_completions(rollout_batch)

        # Dynamic sampling for std=0 groups (DAPO)
        if self.dynamic_sample:
            rollout_batch, rewards_per_func = self._dynamic_sampling(rollout_batch, rewards_per_func)

        advantages = self._compute_advantages(rollout_batch, rewards_per_func)
        
        # ==========CHORD=========================
        chord_mu = self._get_chord_mu() if self.chord_enabled else 0.0
        # ========================================

        def _get_encoded_batch(rollout_batch, advantages, chord_batch=None, chord_mu=0.0):
            """GRPOサンプルのみをエンコード（CHORDは後で混合）"""
            template = self.template
            with self._template_context(template):
                encoded_batch = [template.encode(data, return_length=True) for data in rollout_batch]
                encoded_batch = to_device(
                    template.data_collator(encoded_batch, padding_to=get_padding_to(args)), self.device)
            
            labels = encoded_batch['labels']
            
            if self.template.padding_free:
                position_ids = encoded_batch.get('text_position_ids') or encoded_batch.get('position_ids')
                squeezed_position_ids = position_ids.squeeze()
                
                last_nonzero_idx = (squeezed_position_ids != 0).nonzero(as_tuple=True)[0]
                if len(last_nonzero_idx) > 0:
                    squeezed_position_ids = squeezed_position_ids[:last_nonzero_idx[-1] + 1]
                
                lengths = torch.diff(
                    torch.cat([(squeezed_position_ids == 0).nonzero(as_tuple=True)[0],
                            torch.tensor([len(squeezed_position_ids)]).to(squeezed_position_ids.device)]))
                
                advantages = torch.repeat_interleave(advantages, lengths)
                truncated_mask = torch.tensor([b['is_truncated'] for b in rollout_batch],
                                            dtype=torch.bool, device=self.device)
                truncated_mask = torch.repeat_interleave(truncated_mask, lengths).unsqueeze(0)
            else:
                # ★追加: padding_free=falseの場合
                # 各サンプルの長さを計算（completion部分）
                num_samples = len(rollout_batch)
                seq_len = labels.shape[1]
                
                # completion_maskから各サンプルの有効トークン数を推定
                completion_mask_per_sample = (labels != -100)
                lengths = completion_mask_per_sample.sum(dim=1)
                
                # advantagesをシーケンス長に拡張
                # padding_free=falseでは、advantagesは各サンプルに対応
                advantages_expanded = advantages.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                advantages = advantages_expanded[:labels.numel()].reshape(labels.shape)
                
                truncated_mask = torch.tensor([b['is_truncated'] for b in rollout_batch],
                                            dtype=torch.bool, device=self.device)
                truncated_mask = truncated_mask.unsqueeze(1).expand(-1, seq_len)

            padding_length = labels.shape[1] - truncated_mask.shape[1]
            if padding_length > 0:
                if self.template.padding_free:
                    padding = torch.zeros((1, padding_length), device=truncated_mask.device, dtype=truncated_mask.dtype)
                    truncated_mask = torch.cat([truncated_mask, padding], dim=1)
                else:
                    pass  # 既に正しい形状
            
            if self.template.padding_free:
                position_ids = encoded_batch.get('text_position_ids') or encoded_batch.get('position_ids')
                original_length = position_ids.shape[1]
                if advantages.shape[0] < original_length:
                    padding_length = original_length - advantages.shape[0]
                    padding = torch.zeros(padding_length, device=advantages.device, dtype=advantages.dtype)
                    advantages = torch.cat([advantages, padding])

            completion_mask = labels != -100

            encoded_batch.update({
                'completion_mask': completion_mask,
                'truncated_mask': truncated_mask,
                'advantages': advantages,
                'num_samples': len(rollout_batch),
                'seq_lengths': lengths,
                # CHORDメタデータ（デフォルト値）
                '_chord_mu': 0.0,
                '_num_grpo_samples': len(rollout_batch),
                '_num_sft_samples': 0,
                '_grpo_token_count': labels.shape[1] if not self.template.padding_free else labels.shape[1],
            })

            return encoded_batch

        # Step2: ref/old logps
        total_batch = gather_object(rollout_batch, group=rollout_group)
        total_advantages = gather(advantages, group=rollout_group)

        # ★追加: CHORDサンプルを事前に取得（全マイクロバッチ分）
        num_micro_batches = len(total_batch) // self.micro_batch_size
        chord_batches = []
        if self.chord_enabled and chord_mu > 0:
            for _ in range(num_micro_batches):
                chord_batch = self._get_next_chord_batch_synchronized()
                chord_batches.append(chord_batch)

        mini_batch_data = []

        for idx in range(0, len(total_batch), self.micro_batch_size):
            micro_batch_data = total_batch[idx:idx + self.micro_batch_size]
            micro_batch_data = self._maybe_replace_response_token(micro_batch_data)
            micro_batch_advantages = total_advantages[idx:idx + self.micro_batch_size]
            
            # ★修正: まずGRPOのみでエンコード（CHORDなし）
            micro_batch_data = _get_encoded_batch(micro_batch_data, micro_batch_advantages, None, 0.0)

            # ★修正: GRPOのみでref/old logpsを計算
            with profiling_context(self, 'compute_ref_old_logps'):
                micro_batch_data = self._maybe_compute_logps(micro_batch_data)

            # ★追加: ref/old logps計算後にCHORDバッチを混合
            chord_idx = idx // self.micro_batch_size
            chord_batch = chord_batches[chord_idx] if chord_batches else None
            if chord_batch is not None and chord_mu > 0:
                micro_batch_data = self._merge_chord_into_batch(
                    micro_batch_data, chord_batch, chord_mu)

            mini_batch_data.append(micro_batch_data)

        if self.loss_type in ['cispo', 'dapo']:
            # Calculate num_items_in_batch
            # Count tokens from all mini_batch_data (this includes gathered data from rollout_group)
            total_token_count = sum(batch_data['seq_lengths'].sum().item() if self.template.
                                    padding_free else batch_data['completion_mask'].sum().item()
                                    for batch_data in mini_batch_data)

            # All-reduce across all ranks
            total_token_count_tensor = torch.tensor(total_token_count, dtype=torch.int, device=self.device)
            torch.distributed.all_reduce(total_token_count_tensor)

            # Divide by rollout_group_size to account for duplicate counting within each rollout_group
            # Each rollout_group (TP×PP×CP ranks) has the same gathered data, so we need to normalize
            rollout_group_size = (
                mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
                * mpu.get_context_parallel_world_size())
            num_items_in_batch = total_token_count_tensor / rollout_group_size
            # Store num_items_in_batch in each mini_batch_data for CISPO/DAPO loss normalization
            for batch_data in mini_batch_data:
                batch_data['num_items_in_batch'] = num_items_in_batch

        return mini_batch_data

    @profiling_decorator
    def _generate_completions(self, batch):
        """
        Generate completions for a batch of rollout data using vLLM engine.

        This method processes rollout data for the current process, generates completions
        using the vLLM engine, and merges the results back into the original batch.

        Args:
            batch: Rollout data assigned to the current process.

        Returns:
            batch: The input batch with rollout completion results merged in.
        """
        # add prompt ids and system prompts
        batch = self._preprocess_inputs(batch)
        # Step 1: Wake up the engine if it's sleeping (vLLM colocate mode)
        if self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping:
            wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
            # Load weights only (faster and reduces memory peak)
            kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
            self.engine.engine.wake_up(**kwargs)

        # Step 2: Load model weights
        if self._step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self._step

        context = self.offload_context if self.enable_offload else nullcontext
        with context():
            if (self.vllm_mode == 'colocate' and self.engine.inner_model_executor.is_sleeping
                    and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
                aggressive_empty_cache()
                set_expandable_segments(False)
                self.engine.engine.wake_up(tags=['kv_cache'])

            # Step3: Rollout
            outputs: List[RolloutOutput] = self._rollout(batch)

            # Step4: Sleep to release memory
            if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=self.args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)
            batch = self.postprocess_rollout_data(batch, outputs)

        return batch

    def _rollout(self, batch) -> List[RolloutOutput]:
        batch = self._set_inputs_system(batch)
        request_config = self._get_request_config()
        if self.vllm_mode == 'server':
            rollout_outputs = self._server_rollout(batch, request_config)
        elif self.vllm_mode == 'colocate':
            rollout_outputs = self._colocate_rollout(batch, request_config)
        # log prompt and completions
        messages = gather_object([data['messages'] for data in batch])
        completions = gather_object([data.response.choices[0].message.content for data in rollout_outputs])
        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(messages))
        self._logs['completion'].extend(completions)

        return rollout_outputs

    def postprocess_rollout_data(self, batch, outputs):
        """
        Post-process the raw vLLM generation outputs and merge them back into the
        original input batch.

        Args:
            batch (List[Dict[str, Any]]):
                Original rollout samples.
            outputs (List[RolloutOutput]):
                outputs from vLLM from vLLM TP group

        Returns:
            List[Dict[str, Any]]:
                Updated samples with rollout results merged in.
        """

        def merge_output_input_data(input_data: Dict[str, Union[torch.Tensor, Any]], output: RolloutOutput):
            response = output.response
            choice = response.choices[0]

            # Step 1: Update or append assistant message
            if output.messages:
                input_data['messages'] = output.messages  # Override full message history
            else:
                # not provided, append
                messages = input_data['messages']
                remove_response(messages)
                messages.append({'role': 'assistant', 'content': choice.message.content})
            # Step 2: Add token IDs and loss mask
            if output.response_token_ids:
                input_data['response_token_ids'] = output.response_token_ids
                if output.response_loss_mask:
                    input_data['response_loss_mask'] = output.response_loss_mask
            else:
                # for single turn, skip tokenizer response
                input_data['response_token_ids'] = output.response.choices[0].token_ids

            # Step 3: Attach rollout extra info
            if output.rollout_infos:
                input_data['rollout_infos'] = output.rollout_infos

            # Step 4: Store finish reason (used for truncation filters etc.)
            input_data['finish_reason'] = choice.finish_reason
            input_data['is_truncated'] = choice.finish_reason == 'length'

            return input_data

        assert len(batch) == len(outputs)
        return [merge_output_input_data(input_data, output) for input_data, output in zip(batch, outputs)]

    def _get_request_config(self) -> RequestConfig:
        request_config = copy(self.request_config)
        if self.args.vllm_mode == 'colocate' and self.vllm_tensor_parallel_size > 1:
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            batch_size = self.per_device_generation_batch_size
            batch_size *= self.vllm_tensor_parallel_size
            # Since the TP (Tensor Parallelism) group gathers the inputs,
            # multiply the batch size by the TP parallel size.
            request_config.seed = batch_size * (self.process_index // self.vllm_tensor_parallel_size)

        return request_config

    def _server_rollout(self,
                        inputs: DataType,
                        request_config: RequestConfig,
                        is_global_inputs: bool = False) -> List[RolloutOutput]:
        # TODO: async generate
        infer_requests = self.inputs2requests(inputs)

        if is_global_inputs:
            per_device_size = len(infer_requests) // self.world_size
            all_requests = infer_requests
            all_requests_lengths = [per_device_size] + [0] * (self.world_size - 1)
        else:
            all_requests = gather_object(infer_requests)
            all_requests_lengths = gather_object([len(infer_requests)])

        if not any(requests for requests in all_requests):
            return []

        if self.is_main_process:
            all_outputs: List[RolloutOutput] = self.vllm_client.infer(
                infer_requests=all_requests, request_config=request_config)
            assert len(all_outputs) == len(all_requests)  # TODO: dynamic num of samples
        else:
            all_outputs = [None] * len(all_requests)

        if not is_global_inputs:
            all_outputs = broadcast_object_list(all_outputs, from_process=self.world_size - 1)
            start_idx = sum(all_requests_lengths[:self.process_index])
            end_idx = start_idx + all_requests_lengths[self.process_index]
            outputs = all_outputs[start_idx:end_idx]
        else:
            outputs = all_outputs if self.is_main_process else []
        return outputs

    def _colocate_rollout(self, batch, request_config: RequestConfig):
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
            local_input_length = len(batch)
            all_input_lengths = [None] * self.vllm_tensor_parallel_size
            torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.vllm_tp_group)

            start_idx = sum(all_input_lengths[:local_rank_in_group])
            end_idx = start_idx + all_input_lengths[local_rank_in_group]

            gathered_batch = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_batch, batch, group=self.vllm_tp_group)
            batch = [p for sublist in gathered_batch for p in sublist]

        outputs: List[RolloutOutput] = self.engine.infer(infer_requests=batch, request_config=request_config)

        if self.vllm_tensor_parallel_size > 1:
            outputs = outputs[start_idx:end_idx]

        return outputs

    @profiling_decorator
    def _score_completions(self, inputs: DataType) -> torch.Tensor:
        """Score completions using all reward functions.

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with local reward values
        """
        # Compute rewards using reward functions
        local_rewards_per_func = self._compute_rewards_per_func(inputs)

        return local_rewards_per_func

    def _compute_rewards_per_func(self, batch: DataType) -> torch.Tensor:
        """Compute rewards using all reward functions"""
        device = self.device
        rewards_per_func = torch.zeros((len(batch), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in batch]
        reward_kwargs = {}  # TODO: training step info
        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with profiling_context(self, reward_func_name):
                # reward model
                if isinstance(reward_func, nn.Module):
                    output_reward_func = reward_model_plugin(inputs=batch, **reward_kwargs)
                # reward function
                else:
                    # Repeat all input columns (but "messages" and "completion") to match the number of generations
                    reward_kwargs.update(RowPreprocessor.rows_to_batched(batch))
                    output_reward_func = reward_func(completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs['completion'] = completions[nan_row_idx]
            logger.warning(f'All reward functions returned None for the following kwargs: {row_reward_kwargs}. '
                           'Please ensure that at least one reward function returns a valid reward.')

        return rewards_per_func

    def _compute_advantages(self, batch: DataType, rewards_per_func: torch.Tensor) -> torch.Tensor:
        """Compute advantages for RL training."""

        def normalize_advantages(advantages: torch.Tensor, rewards_std: torch.Tensor) -> torch.Tensor:
            """Normalize advantages if configured; otherwise, return as-is."""
            if self.scale_rewards != 'none':
                return advantages / (rewards_std + 1e-4)
            return advantages

        mode = 'train' if self.unwrapped_models[0].training else 'eval'
        assert len(batch) == rewards_per_func.shape[0]
        total_rewards_per_func = gather(rewards_per_func)
        rewards = (total_rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
        grouped_rewards = rewards.view(-1, self.num_generations)

        # Compute group statistics
        group_rewards_mean = grouped_rewards.mean(dim=1)

        # Broadcast stats back to the original shape
        group_rewards_mean = group_rewards_mean.repeat_interleave(self.num_generations)

        # Compute advantages relative to group mean
        advantages = rewards - group_rewards_mean

        # Normalize advantages based on scale_rewards setting
        if self.scale_rewards == 'batch':
            # Global batch-level normalization
            rewards_std = rewards.std().expand_as(rewards)
        elif self.scale_rewards == 'group':
            # Group-level normalization (default)
            rewards_std = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations)
        else:  # 'none'
            rewards_std = None

        if rewards_std is not None:
            advantages = normalize_advantages(advantages, rewards_std)

        def log_rewards_metrics(rewards: torch.Tensor, rewards_per_func_for_metrics: torch.Tensor):
            """Log reward statistics for monitoring. Only log once per unique request_id."""
            # rewards: [prompt_batch_size, self.num_generations]
            # rewards_per_func_for_metrics: [prompt_batch_size*self.num_generations, self.num_reward_funcs]
            group_rewards = rewards.view(-1, self.num_generations)
            rewards_mean = group_rewards.mean(-1).mean().item()
            # Compute std based on scale_rewards setting for logging
            if self.scale_rewards in ['group', 'none']:
                rewards_std = group_rewards.std(-1).mean().item()
            elif self.scale_rewards == 'batch':
                rewards_std = rewards.std().item()
            is_std_zero = torch.isclose(group_rewards.std(dim=1), torch.zeros_like(group_rewards.std(dim=1)))

            self._metrics[mode]['reward'].append(rewards_mean)
            self._metrics[mode]['reward_std'].append(rewards_std)
            self._metrics[mode]['frac_reward_zero_std'].append(is_std_zero.float().mean().item())

            # Log per-reward-function statistics using deduplicated rewards_per_func
            for i, name in enumerate(self.reward_func_names):
                col = rewards_per_func_for_metrics[:, i]
                self._metrics[mode][f'rewards/{name}/mean'].append(torch.nanmean(col).item())
                self._metrics[mode][f'rewards/{name}/std'].append(nanstd(col).item())

        log_rewards_metrics(rewards=grouped_rewards, rewards_per_func_for_metrics=total_rewards_per_func)
        self._logs['advantages'].extend(advantages.tolist())
        for i, name in enumerate(self.reward_func_names):
            self._logs['rewards'][name].extend(total_rewards_per_func[:, i].tolist())

        slice_start = self.process_index * len(batch)
        slice_end = slice_start + len(batch)
        advantages = advantages[slice_start:slice_end]

        return advantages

    def _dynamic_sampling(self, rollout_batch: DataType,
                          rewards_per_func: torch.Tensor) -> Tuple[DataType, torch.Tensor]:
        """
        Perform dynamic sampling to replace samples with zero-reward-variance groups.

        This method implements DAPO (https://arxiv.org/abs/2503.14476) by replacing
        samples from groups with zero reward variance (std=0) through resampling.

        Args:
            rollout_batch: local rollout data samples
            rewards_per_func: reward per function for local data samples
            rollout_group: rollout communication group

        Returns:
            tuple: (rollout_batch, rewards_per_func) with zero-variance groups replaced by resampled data
        """
        resample_count = 0
        valid_samples = []
        valid_rewards_per_func = []
        origin_data = (rollout_batch, rewards_per_func)

        while resample_count < self.max_resample_times:
            # Gather all samples and rewards across rollout group first
            global_rollout_batch = gather_object(rollout_batch)
            global_rewards_per_func = gather(rewards_per_func)

            # Compute reward std for the entire global batch
            # We need to compute std on the gathered data to get a global mask
            global_rewards = (global_rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)
            grouped_rewards = global_rewards.view(-1, self.num_generations)
            group_rewards_std = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations)
            global_valid_mask = (group_rewards_std > 0)

            # Filter valid samples based on std > 0
            valid_samples.extend([sample for sample, mask in zip(global_rollout_batch, global_valid_mask) if mask])
            valid_rewards_per_func.append(global_rewards_per_func[global_valid_mask])

            if len(valid_samples) >= self.generation_batch_size:
                break

            # Lazy initialization of resample_data_iterator
            # Only initialize when needed, after pretrain() has set up args
            if not hasattr(self, 'resample_data_iterator') or self.resample_data_iterator is None:
                self.resample_data_iterator = self._init_resample_data_iterator()
            num_iters_per_step = self.get_num_iters_per_step()
            next_rollout_prompt_batch = []
            for _ in range(num_iters_per_step):
                next_rollout_prompt_batch.extend(next(self.resample_data_iterator))

            # Repeat num_generations times and get local slice
            rollout_batch = self.get_local_rollout_batch(next_rollout_prompt_batch)

            # Generate and score new completions
            rollout_batch = self._generate_completions(rollout_batch)
            rewards_per_func = self._score_completions(rollout_batch)
            resample_count += 1

        if len(valid_samples) >= self.generation_batch_size:
            # Get local slice of valid samples
            rank = self.process_index
            per_device_batch_size = self.per_device_generation_batch_size
            data_slice = slice(rank * per_device_batch_size, (rank + 1) * per_device_batch_size)
            rollout_batch = valid_samples[:self.generation_batch_size][data_slice]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.generation_batch_size][data_slice]
        else:
            logger.warning(f'There are still std=0 groups present after {self.max_resample_times} retries.')
            rollout_batch, rewards_per_func = origin_data

        return rollout_batch, rewards_per_func

    def _maybe_compute_logps(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = {
            k: v
            for k, v in batch.items() if k not in [
                'completion_mask', 'advantages', 'truncated_mask', 'seq_lengths',
                '_chord_mu', '_num_grpo_samples', '_num_sft_samples', '_grpo_token_count',
                'attention_mask_2d'
            ]
        }
        if self.beta != 0.0:
            with torch.no_grad(), self.null_ref_context() as ref_models:
                assert len(ref_models) == 1, 'GRPO currently does not support VPP.'
                ref_model = ref_models[0]
                batch['ref_per_token_logps'] = self.model_forward(
                    ref_model, iter([deepcopy(inputs)]), no_grad=True, per_token=True)['logps']

        if not self.on_policy:
            batch['old_per_token_logps'] = self.model_forward(
                self.unwrapped_models[0], iter([deepcopy(inputs)]), no_grad=True, per_token=True)['logps']
        return batch

    @contextmanager
    def _disable_maxlength_template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        template.max_length = None
        try:
            yield
        finally:
            template.max_length = max_length

    def _maybe_replace_response_token(self, batch):
        # maybe replace the response token with the response token ids to avoid repetitive tokenize

        for data in batch:
            if 'response_token_ids' in data and data['response_token_ids']:
                loss_mask = None
                if 'response_loss_mask' in data and data['response_loss_mask']:
                    loss_mask = data['response_loss_mask']
                # token in token out
                data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'],
                                                                       loss_mask)
        return batch

    @property
    def on_policy(self):
        return self.steps_per_generation == 1

    @contextmanager
    def patch_megatron_data_collator(self, data_collator):
        """
        Context manager that temporarily patches Megatron's data-loader factory so each
        prompt-level micro-batch size equals (original micro-batch size // num_generations),
        required by GRPO.  Restores the original size and loader on exit.
        """
        origin_build_pretraining_data_loader = training.build_pretraining_data_loader

        def build_pretraining_data_loader(*_args, **kwargs):
            args = get_args()
            org_micro_batch_size = args.micro_batch_size
            # args.micro_batch_size = org_micro_batch_size // self.num_generations
            res = origin_build_pretraining_data_loader(*_args, **kwargs)
            args.micro_batch_size = org_micro_batch_size
            if res is not None and args.dataloader_type != 'external':
                res.collate_fn = data_collator
            return res

        training.build_pretraining_data_loader = build_pretraining_data_loader
        try:
            yield
        finally:
            training.build_pretraining_data_loader = origin_build_pretraining_data_loader

    @profiling_decorator
    def forward_step(self, data_iterator, model):
        data = self.get_batch(data_iterator)
        data.pop('loss_scale', None)
        inputs = {
            k: v for k, v in data.items() 
            if k not in ['completion_mask', 'ref_per_token_logps', 'advantages', 
                        'old_per_token_logps', 'truncated_mask', 'seq_lengths',
                        '_chord_mu', '_num_grpo_samples', '_num_sft_samples', '_grpo_token_count',
                        'attention_mask_2d'
                    ]
        }

        # ★★★ Qwen3-Next対応: attention_maskを2Dに変換 ★★★
        # Qwen3-Nextの apply_mask_to_padding_states は 2D mask [batch, seq_len] を期待
        # 4D causal mask [batch, 1, seq_len, seq_len] が渡されると失敗する
        if 'attention_mask' in inputs and inputs['attention_mask'] is not None:
            attn_mask = inputs['attention_mask']
            input_ids = inputs.get('input_ids')
            
            if attn_mask.ndim == 4:
                # 4D -> 2D: input_idsからpadding maskを生成
                # pad_token_id以外の位置を1とする
                if input_ids is not None:
                    pad_token_id = self.template.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = 0
                    # 2D attention_mask: padding以外の位置が1
                    attn_mask_2d = (input_ids != pad_token_id).to(attn_mask.dtype)
                    inputs['attention_mask'] = attn_mask_2d
                    logger.debug(f"[forward_step] Converted 4D attention_mask to 2D: {attn_mask.shape} -> {attn_mask_2d.shape}")
                else:
                    # input_idsがない場合は4Dの対角成分から2Dを推定
                    # [batch, 1, seq_len, seq_len] -> 対角がすべて1なら全部有効
                    batch_size = attn_mask.shape[0]
                    seq_len = attn_mask.shape[-1]
                    attn_mask_2d = torch.ones(batch_size, seq_len, device=attn_mask.device, dtype=attn_mask.dtype)
                    inputs['attention_mask'] = attn_mask_2d
                    logger.debug(f"[forward_step] Created 2D attention_mask from 4D: shape={attn_mask_2d.shape}")
            
            elif attn_mask.ndim == 2:
                # 2Dだが、input_idsとサイズが合っているか確認
                if input_ids is not None and attn_mask.shape[1] != input_ids.shape[1]:
                    logger.warning(f"[forward_step] attention_mask.shape[1]={attn_mask.shape[1]} != input_ids.shape[1]={input_ids.shape[1]}")
                    # input_idsからpadding maskを再生成
                    pad_token_id = self.template.tokenizer.pad_token_id
                    if pad_token_id is None:
                        pad_token_id = 0
                    attn_mask_2d = (input_ids != pad_token_id).to(attn_mask.dtype)
                    inputs['attention_mask'] = attn_mask_2d
                    logger.info(f"[forward_step] Regenerated 2D attention_mask: shape={attn_mask_2d.shape}")

        # ★★★ FIX: モデル入力前の形状検証 ★★★
        if 'input_ids' in inputs:
            input_ids_for_model = inputs['input_ids']
            logger.debug(f"[forward_step] Model input_ids shape: {input_ids_for_model.shape}")
            if 'packed_seq_params' in inputs and inputs['packed_seq_params'] is not None:
                psp = inputs['packed_seq_params']
                expected_tokens = psp.cu_seqlens_q[-1].item() if psp.cu_seqlens_q is not None else None
                actual_tokens = input_ids_for_model.shape[1]
                if expected_tokens is not None and expected_tokens != actual_tokens:
                    logger.warning(f"[forward_step] Token count mismatch! expected from cu_seqlens: {expected_tokens}, actual input_ids: {actual_tokens}")
                    # ★ 修正: cu_seqlens_qの最後の値がinput_idsと一致するように調整
                    if actual_tokens > expected_tokens:
                        logger.warning(f"[forward_step] Truncating input_ids from {actual_tokens} to {expected_tokens}")
                        inputs['input_ids'] = input_ids_for_model[:, :expected_tokens]
        
        with self.stimer:
            output_tensor = model(**inputs)
        
        # ★CHORDのメタデータはdataに既に含まれている（_generate_and_score_completionsで追加済み）
        return output_tensor, partial(self.loss_func, data=data)

    @profiling_decorator
    def loss_func(self, output_tensor: torch.Tensor, data: Dict[str, Any]):
            # ★追加: CHORDメタデータを取得
        chord_mu = data.get('_chord_mu', 0.0)
        num_grpo_samples = data.get('_num_grpo_samples', data.get('num_samples', self.micro_batch_size))
        num_sft_samples = data.get('_num_sft_samples', 0)
        grpo_token_count = data.get('_grpo_token_count', 0)

        advantages = data['advantages']
        labels = data['labels']
        completion_mask = data['completion_mask']
        packed_seq_params = data['packed_seq_params']
        truncated_mask = data['truncated_mask']
        micro_batch_size = self.micro_batch_size

        # ★修正: GRPOサンプル数を使用 (packed_seq_params が None の場合に対応)
        if packed_seq_params is not None:
            lengths = packed_seq_params.cu_seqlens_q[1:num_grpo_samples + 1] - \
                    packed_seq_params.cu_seqlens_q[:num_grpo_samples]
            lengths_with_padding = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1]
            num_samples_for_logps = packed_seq_params.num_samples
        else:
            # 非padding-freeモード: labelsの形状から長さを計算
            seq_len = labels.shape[1]
            lengths = torch.tensor([seq_len] * num_grpo_samples, device=labels.device)
            lengths_with_padding = torch.tensor([seq_len] * (num_grpo_samples + num_sft_samples), device=labels.device)
            num_samples_for_logps = num_grpo_samples + num_sft_samples
        
        # ★修正: GRPOとSFTのトークン範囲を分離
        if num_sft_samples > 0 and chord_mu > 0 and grpo_token_count > 0:
            # GRPO部分
            grpo_labels = labels[:, :grpo_token_count]
            grpo_completion_mask = completion_mask[:, :grpo_token_count]
            grpo_advantages = advantages[:grpo_token_count]
            grpo_truncated_mask = truncated_mask[:, :grpo_token_count]
            
            # SFT部分
            sft_labels = labels[:, grpo_token_count:]
            sft_completion_mask = completion_mask[:, grpo_token_count:]
        else:
            grpo_labels = labels
            grpo_completion_mask = completion_mask
            grpo_advantages = advantages
            grpo_truncated_mask = truncated_mask
            sft_labels = None
            sft_completion_mask = None

        # get_logps with per_token=True now returns full sequences (all_gather in CP mode)
        per_token_logps = self.get_logps(
            output_tensor, labels, packed_seq_params, num_samples_for_logps, per_token=True)

        # ★修正: GRPO部分のみ抽出
        if grpo_token_count > 0 and num_sft_samples > 0:
            grpo_per_token_logps = per_token_logps[:, :grpo_token_count]
            sft_per_token_logps = per_token_logps[:, grpo_token_count:]
        else:
            grpo_per_token_logps = per_token_logps
            sft_per_token_logps = None

        if self.args.overlong_filter and truncated_mask.any():
            completion_mask = completion_mask & (~truncated_mask)
            if not completion_mask.any():
                logger.warning('All completions are truncated in this batch. Loss and grad_norm will be 0. '
                               'Consider increasing max_completion_length')

        # KL計算（GRPOのみ）
        if self.beta != 0.0:
            ref_per_token_logps = data.get('ref_per_token_logps')
            if grpo_token_count > 0 and num_sft_samples > 0 and ref_per_token_logps is not None:
                grpo_ref_per_token_logps = ref_per_token_logps[:, :grpo_token_count]
            else:
                grpo_ref_per_token_logps = ref_per_token_logps
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        # old_logs処理（GRPOのみ）
        old_per_token_logps = data.get('old_per_token_logps')
        if old_per_token_logps is None:
            grpo_old_per_token_logps = grpo_per_token_logps.detach()
        else:
            if grpo_token_count > 0 and num_sft_samples > 0:
                grpo_old_per_token_logps = old_per_token_logps[:, :grpo_token_count]
            else:
                grpo_old_per_token_logps = old_per_token_logps
        
        log_ratio = grpo_per_token_logps - grpo_old_per_token_logps

        # ★修正: GRPOのlengths_with_paddingもGRPO部分のみ
        grpo_lengths_with_padding = lengths_with_padding[:num_grpo_samples]

        # importance sampling計算（既存コード、変数名を調整）
        if self.importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ['sequence', 'sequence_token']:
            if self.template.padding_free:
                # padding-freeモード: パックされたシーケンスをsplitで分割
                log_ratio_list = torch.split(log_ratio.squeeze(0), grpo_lengths_with_padding.tolist())
                mask_list = torch.split(grpo_completion_mask.squeeze(0), grpo_lengths_with_padding.tolist())
                seq_weights = torch.stack([(lr * m).sum() / m.sum().clamp(min=1.0)
                                        for lr, m in zip(log_ratio_list, mask_list)])
                seq_level_log_weights = seq_weights.to(log_ratio.dtype).unsqueeze(-1)
                if self.importance_sampling_level == 'sequence':
                    log_importance_weights = seq_level_log_weights
                else:
                    seq_level_log_weight = seq_level_log_weights.detach()
                    seq_level_log_weight = torch.repeat_interleave(
                        seq_level_log_weight.squeeze(-1), grpo_lengths_with_padding, dim=0).unsqueeze(0)
                    log_importance_weights = grpo_per_token_logps - grpo_per_token_logps.detach() + seq_level_log_weight
            else:
                # 非padding-freeモード: [batch, seq_len] 形式
                # GRPO部分のみ抽出
                grpo_log_ratio = log_ratio[:num_grpo_samples] if log_ratio.dim() > 1 else log_ratio
                grpo_mask_for_is = grpo_completion_mask[:num_grpo_samples] if grpo_completion_mask.dim() > 1 else grpo_completion_mask
                seq_weights = torch.stack([
                    (grpo_log_ratio[i] * grpo_mask_for_is[i]).sum() / grpo_mask_for_is[i].sum().clamp(min=1.0)
                    for i in range(num_grpo_samples)
                ])
                seq_level_log_weights = seq_weights.to(log_ratio.dtype).unsqueeze(-1)
                if self.importance_sampling_level == 'sequence':
                    log_importance_weights = seq_level_log_weights
                else:
                    # sequence_token: シーケンスレベルの重みをトークンレベルに展開
                    seq_level_log_weight = seq_level_log_weights.detach()
                    # [num_grpo_samples, 1] -> [num_grpo_samples, seq_len] にブロードキャスト
                    log_importance_weights = grpo_per_token_logps - grpo_per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")


        coef_1 = torch.exp(log_importance_weights)

        # GRPO損失計算
        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            if self.template.padding_free:
                if self.importance_sampling_level == 'sequence':
                    clamped_ratios = torch.repeat_interleave(
                        clamped_ratios.squeeze(-1), grpo_lengths_with_padding, dim=0).unsqueeze(0)
                grpo_advantages_aligned = grpo_advantages[-clamped_ratios.shape[1]:]
                per_token_loss = -clamped_ratios * grpo_advantages_aligned.unsqueeze(0) * grpo_per_token_logps
            else:
                # 非padding-freeモード: [batch, seq_len] 形式
                # grpo_advantages を seq_len 次元に合わせてブロードキャスト
                seq_len = grpo_per_token_logps.shape[-1]
                if grpo_advantages.dim() == 1:
                    grpo_advantages_aligned = grpo_advantages[-seq_len:].unsqueeze(0)
                else:
                    grpo_advantages_aligned = grpo_advantages[..., -seq_len:]
                per_token_loss = -clamped_ratios * grpo_advantages_aligned * grpo_per_token_logps
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            if self.template.padding_free:
                if self.importance_sampling_level == 'sequence':
                    coef_1 = torch.repeat_interleave(coef_1.squeeze(-1), grpo_lengths_with_padding, dim=0).unsqueeze(0)
                    coef_2 = torch.repeat_interleave(coef_2.squeeze(-1), grpo_lengths_with_padding, dim=0).unsqueeze(0)

                grpo_advantages_aligned = grpo_advantages[-coef_1.shape[1]:]
                per_token_loss1 = coef_1 * grpo_advantages_aligned.unsqueeze(0)
                per_token_loss2 = coef_2 * grpo_advantages_aligned.unsqueeze(0)
            else:
                # 非padding-freeモード: [batch, seq_len] 形式
                seq_len = coef_1.shape[-1]
                if grpo_advantages.dim() == 1:
                    grpo_advantages_aligned = grpo_advantages[-seq_len:].unsqueeze(0)
                else:
                    grpo_advantages_aligned = grpo_advantages[..., -seq_len:]
                per_token_loss1 = coef_1 * grpo_advantages_aligned
                per_token_loss2 = coef_2 * grpo_advantages_aligned
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')
        
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # GRPO損失の集約
        if self.loss_type == 'grpo':
            if self.template.padding_free:
                # padding-freeモード: パックされたシーケンスをsplitで分割
                loss_list = torch.split(per_token_loss.squeeze(0), grpo_lengths_with_padding.tolist())
                mask_list = torch.split(grpo_completion_mask.squeeze(0), grpo_lengths_with_padding.tolist())
                sample_loss = torch.stack([
                    (loss * mask).sum() / mask.sum().clamp(min=1.0)
                    for loss, mask in zip(loss_list[:num_grpo_samples], mask_list[:num_grpo_samples])
                ])
                grpo_loss = sample_loss.mean()
            else:
                # 非padding-freeモード: [batch, seq_len] 形式、バッチ次元でイテレート
                # GRPO部分のみ抽出 (最初の num_grpo_samples 行)
                grpo_per_token_loss = per_token_loss[:num_grpo_samples] if per_token_loss.dim() > 1 else per_token_loss
                grpo_mask = grpo_completion_mask[:num_grpo_samples] if grpo_completion_mask.dim() > 1 else grpo_completion_mask
                sample_loss = torch.stack([
                    (grpo_per_token_loss[i] * grpo_mask[i]).sum() / grpo_mask[i].sum().clamp(min=1.0)
                    for i in range(num_grpo_samples)
                ])
                grpo_loss = sample_loss.mean()
        elif self.loss_type == 'bnpo':
            grpo_loss = (per_token_loss * grpo_completion_mask).sum() / grpo_completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            grpo_loss = (per_token_loss * grpo_completion_mask).sum() / (num_grpo_samples * self.max_completion_length)
        elif self.loss_type in ['cispo', 'dapo']:
            num_items_in_batch = data['num_items_in_batch']
            dp_size = mpu.get_data_parallel_world_size()
            normalizer = num_items_in_batch / dp_size
            grpo_loss = (per_token_loss * grpo_completion_mask).sum() / normalizer.clamp(min=1.0)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')
        
        # 追加：SFT損失計算
        sft_loss = None
        if num_sft_samples > 0 and chord_mu > 0 and sft_per_token_logps is not None:
            sft_loss = -(sft_per_token_logps * sft_completion_mask).sum() / sft_completion_mask.sum().clamp(min=1.0)
            
            # φ関数の適用（オプション）
            if self.chord_enable_phi_function:
                phi_weights = self._compute_phi_weights(sft_per_token_logps, sft_completion_mask)
                sft_loss = -(sft_per_token_logps * phi_weights * sft_completion_mask).sum() / \
                        sft_completion_mask.sum().clamp(min=1.0)

        # ★追加: 最終損失の計算（CHORD混合）
        if sft_loss is not None and chord_mu > 0:
            loss = (1 - chord_mu) * grpo_loss + chord_mu * sft_loss
        else:
            loss = grpo_loss

        avg_metric = {
            'loss': loss.clone().detach(),
        }
        custom_metrics = {}
        total_lengths = gather(lengths, group=mpu.get_data_parallel_group(with_context_parallel=True))
        custom_metrics = {
            'completions/mean_length': total_lengths.float().mean(),
            'completions/max_length': total_lengths.float().max(),
            'completions/min_length': total_lengths.float().min(),
        }

        if self.beta != 0.0:
            # Unified processing (no CP-specific logic needed)
            kl_value = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            avg_metric['kl'] = kl_value.clone().detach()

        # ★追加: CHORDメトリクス
        if sft_loss is not None and chord_mu > 0:
            custom_metrics['chord/mu'] = torch.tensor(chord_mu, device=loss.device)
            custom_metrics['chord/grpo_loss'] = grpo_loss.detach()
            custom_metrics['chord/sft_loss'] = sft_loss.detach()

        mode = 'train' if self.unwrapped_models[0].training else 'eval'

        # Compute clipping metrics
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        if self.loss_type == 'cispo':
            # CISPO: Only track upper bound clipping
            if self.template.padding_free:
                # Recompute coef_1_expanded for metrics (use original coef_1 before clamping)
                if self.importance_sampling_level == 'sequence':
                    coef_1_expanded = torch.repeat_interleave(
                        coef_1.squeeze(-1), lengths_with_padding, dim=0).unsqueeze(0)
                else:
                    coef_1_expanded = coef_1
                advantages_for_metrics = advantages[-coef_1_expanded.shape[1]:]
                is_cispo_clipped = (coef_1_expanded > self.epsilon_high) & (advantages_for_metrics.unsqueeze(0) > 0)
            else:
                # 非padding-freeモード: [batch, seq_len] 形式
                coef_1_expanded = coef_1
                seq_len = coef_1_expanded.shape[-1]
                if advantages.dim() == 1:
                    advantages_for_metrics = advantages[-seq_len:].unsqueeze(0)
                else:
                    advantages_for_metrics = advantages[..., -seq_len:]
                is_cispo_clipped = (coef_1_expanded > self.epsilon_high) & (advantages_for_metrics > 0)
            cispo_clip_ratio = (is_cispo_clipped.float() * completion_mask).sum() / completion_token_count
            # Store local clip ratio, _all_reduce_metric will handle averaging across ranks
            self._metrics[mode]['cispo_clip_ratio'].append(cispo_clip_ratio)
        elif self.loss_type in ['grpo', 'bnpo', 'dr_grpo', 'dapo']:
            if self.template.padding_free:
                # Use coef_1 before clamping for metrics (need to expand if sequence-level)
                if self.importance_sampling_level == 'sequence':
                    coef_1_expanded = torch.repeat_interleave(
                        torch.exp(log_importance_weights).squeeze(-1), lengths_with_padding, dim=0).unsqueeze(0)
                else:
                    coef_1_expanded = torch.exp(log_importance_weights)
                advantages_for_metrics = advantages[-coef_1_expanded.shape[1]:]
                is_low_clipped = (coef_1_expanded < 1 - self.epsilon_low) & (advantages_for_metrics.unsqueeze(0) < 0)
                is_high_clipped = (coef_1_expanded > 1 + self.epsilon_high) & (advantages_for_metrics.unsqueeze(0) > 0)
            else:
                # 非padding-freeモード: [batch, seq_len] 形式
                coef_1_expanded = torch.exp(log_importance_weights)
                seq_len = coef_1_expanded.shape[-1]
                if advantages.dim() == 1:
                    advantages_for_metrics = advantages[-seq_len:].unsqueeze(0)
                else:
                    advantages_for_metrics = advantages[..., -seq_len:]
                is_low_clipped = (coef_1_expanded < 1 - self.epsilon_low) & (advantages_for_metrics < 0)
                is_high_clipped = (coef_1_expanded > 1 + self.epsilon_high) & (advantages_for_metrics > 0)
            low_clip = (is_low_clipped.float() * completion_mask).sum() / completion_token_count
            high_clip = (is_high_clipped.float() * completion_mask).sum() / completion_token_count
            is_region_clipped = is_low_clipped | is_high_clipped
            clip_ratio = (is_region_clipped.float() * completion_mask).sum() / completion_token_count

            # For min/max, we need to gather values from all ranks to compute global min/max
            # For mean, let _all_reduce_metric handle averaging
            gathered_low_clip = gather(
                low_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))
            gathered_high_clip = gather(
                high_clip.unsqueeze(0), group=mpu.get_data_parallel_group(with_context_parallel=True))

            # Store local values for mean (will be averaged by _all_reduce_metric)
            self._metrics[mode]['clip_ratio/low_mean'].append(low_clip)
            self._metrics[mode]['clip_ratio/high_mean'].append(high_clip)
            self._metrics[mode]['clip_ratio/region_mean'].append(clip_ratio)
            custom_metrics['clip_ratio/low_min'] = gathered_low_clip.min()
            custom_metrics['clip_ratio/high_max'] = gathered_high_clip.max()
        
        # ==================== CHORD修正8: loss_func（パイプライン並列対応）====================
        # ★修正5: UserWarning修正 - torch.tensor()の代わりにdetach().clone()を使用
        if self._metrics[mode]:
            addition_metrics = {}
            for key, val in self._metrics[mode].items():
                if len(val) > 0:
                    if isinstance(val[0], torch.Tensor):
                        mean_val = torch.stack([v.detach() if v.requires_grad else v for v in val]).mean()
                    else:
                        mean_val = torch.tensor(sum(val) / len(val), device=loss.device)
                    addition_metrics[key] = mean_val
        # ============================================================================
            avg_metric.update(addition_metrics)

        avg_metric = self._all_reduce_metric(avg_metric)

        reporting_metric = {**avg_metric, **custom_metrics}

        # ==================== CHORD損失の統合（追加）====================
        if data.get('_chord_enabled', False) and self.chord_enabled:
            mu = self._get_chord_mu()
            
            if mu > 0:
                # ★修正: パイプラインの最終ステージでのみCHORD損失を計算
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                pp_size = mpu.get_pipeline_model_parallel_world_size()
                
                if pp_rank == pp_size - 1:
                    try:
                        # CHORDバッチを取得（最終ステージでのみ）
                        chord_batch = self._get_next_chord_batch()
                        
                        if chord_batch is not None:
                            with profiling_context(self, 'chord_sft_loss'):
                                sft_loss = self._compute_chord_sft_loss(
                                    self.unwrapped_models[0],
                                    chord_batch
                                )
                            
                            # CHORD統合損失
                            grpo_loss = loss
                            loss = (1 - mu) * grpo_loss + mu * sft_loss
                            
                            # メトリクス記録
                            reporting_metric['chord/mu'] = torch.tensor(mu, device=loss.device)
                            reporting_metric['chord/grpo_loss'] = grpo_loss.detach()
                            reporting_metric['chord/sft_loss'] = sft_loss.detach()
                    except Exception as e:
                        logger.warning(f'[CHORD] SFT loss computation failed: {e}')
                        import traceback
                        traceback.print_exc()
        # ================================================================

        # log_completions
        if self.log_completions and self.is_main_process and (self._step - 1) % self.steps_per_generation == 0:
            table = {
                'gen_step': [self._step - 1] * len(self._logs['prompt']),
                'prompt': list(self._logs['prompt']),
                'completion': list(self._logs['completion']),
                **{k: list(v)
                   for k, v in self._logs['rewards'].items()},
                'advantages': list(self._logs['advantages']),
            }
            self.jsonl_writer.append(table)
            wandb_writer = get_wandb_writer()
            if wandb_writer:
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                # if not self.init_custom_metric:
                #     wandb_writer.define_metric('completions', step_metric='gen_step')
                #     self.init_custom_metric = True
                wandb_writer.log({'completions': wandb.Table(dataframe=df)})

        return loss, reporting_metric

    def model_forward(self, model, data_iterator, no_grad=True, per_token=False):
        # used to calculate model forward (logps) in GRPO
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator)
        data.pop('loss_scale', None)
        labels = data.get('labels')
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            output_tensor = forward_step_helper(model, data)
        
        # ★修正: packed_seq_paramsが存在しない場合の処理
        packed_seq_params = data.get('packed_seq_params')  # []からget()に変更
        
        if labels is None:
            data['logps'] = None
        elif packed_seq_params is not None:
            # padding_free=true の場合
            data['logps'] = self.get_logps(
                output_tensor, labels, packed_seq_params, 
                packed_seq_params.num_samples, per_token=per_token)
        else:
            # ★追加: padding_free=false の場合
            num_samples = data.get('num_samples', labels.shape[0])
            data['logps'] = self.get_logps(
                output_tensor, labels, None, num_samples, per_token=per_token)
        
        return data

    @contextmanager
    def offload_context(self):
        if self.args.offload_model:
            offload_megatron_model_to_cpu(self.wrapped_models)
            if hasattr(self, 'ref_models') and self.ref_models:
                offload_megatron_model_to_cpu(self.ref_models)
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                load_megatron_model_to_gpu(self.wrapped_models)
                if hasattr(self, 'ref_models') and self.ref_models:
                    load_megatron_model_to_gpu(self.ref_models)
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                load_megatron_optimizer(self.optimizer)

    def inputs2requests(self, inputs: DataType) -> List[RolloutInferRequest]:
        """Convert raw input data into RolloutInferRequest objects"""

        def _process_image_data(image_data: Union[dict, str]) -> str:
            if isinstance(image_data, dict):
                if image_data.get('bytes'):
                    return base64.b64encode(image_data['bytes']).decode('utf-8')
                if image_data.get('path'):
                    return image_data['path']
            return image_data

        if not inputs:
            return []
        args = self.args

        REQUEST_METADATA_FIELDS = ['messages', 'images', 'audios', 'videos', 'tools', 'objects', 'uuid']
        requests_dicts = []

        for data in inputs:
            request_data = {key: data[key] for key in REQUEST_METADATA_FIELDS if key in data and data[key] is not None}
            if 'uuid' not in request_data:
                request_data['uuid'] = data['request_id']
            if hasattr(args, 'vllm_server_pass_dataset') and args.vllm_server_pass_dataset:
                extra_fields = {
                    k: v
                    for k, v in data.items() if k not in REQUEST_METADATA_FIELDS and data[k] is not None
                }
                if extra_fields:
                    request_data['data_dict'] = extra_fields
            elif self.multi_turn_scheduler:
                base_data_dict = {}
                if 'data_dict' in data:
                    if isinstance(data['data_dict'], dict):
                        base_data_dict = data['data_dict']
                    else:
                        raise ValueError('data_dict exists but is not a dictionary')
                extra_data = {
                    k: v
                    for k, v in data.items()
                    if k not in REQUEST_METADATA_FIELDS and k != 'data_dict' and data[k] is not None
                }
                final_data_dict = {**extra_data, **base_data_dict}
                request_data['data_dict'] = final_data_dict if final_data_dict else {}

            requests_dicts.append(request_data)

        for request in requests_dicts:
            if 'images' in request and request['images']:
                request['images'] = ([_process_image_data(img) for img in request['images']] if isinstance(
                    request['images'], list) else _process_image_data(request['images']))

        return [from_dict(RolloutInferRequest, request_data) for request_data in requests_dicts]

    def _preprocess_inputs(self, inputs: DataType) -> DataType:
        """Preprocess inputs before inference"""
        processed_inputs = self._add_prompt_id_to_inputs(inputs)
        for input_item in processed_inputs:
            remove_response(input_item['messages'])
        return processed_inputs

    def _add_prompt_id_to_inputs(self, inputs: DataType) -> DataType:
        """Add unique prompt_id and request_id to each input"""
        if not inputs:
            return inputs

        all_messages = gather_object([inp['messages'] for inp in inputs])
        messages_to_prompt_id = {}
        prompt_id_counter = 0

        for messages in all_messages:
            key = json.dumps(messages)
            if key not in messages_to_prompt_id:
                messages_to_prompt_id[key] = f'prompt_{prompt_id_counter}'
                prompt_id_counter += 1

        for input_item in inputs:
            messages = input_item.get('messages')
            input_item['prompt_id'] = messages_to_prompt_id[json.dumps(messages)]
            input_item['request_id'] = f'chatcmpl-{str(uuid.uuid4().hex)}'

        return inputs

    def get_num_iters_per_step(self):
        if hasattr(self, '_num_iters_per_step'):
            return self._num_iters_per_step
        # each rollout DP group will generate generation_batch_size / dp_size completions
        dp_size = mpu.get_data_parallel_world_size()
        completions_to_rollout = self.generation_batch_size // dp_size
        # completions will be repeated num_generations times after
        # so we need to divide num_iters_per_step by num_generations to get prompt batch size
        prompts_to_rollout = completions_to_rollout // self.num_generations
        # every iter will generate micro_batch_size prompts
        num_iters_per_step = prompts_to_rollout // self.micro_batch_size
        assert num_iters_per_step > 0, (
            f'num_iters_per_step={num_iters_per_step} <= 0. '
            f'This means no prompts will be generated'
            f'generation_batch_size={self.generation_batch_size}, '
            f'data_parallel_world_size={mpu.get_data_parallel_world_size()}, '
            f'num_generations={self.num_generations}, '
            f'micro_batch_size={self.micro_batch_size}. '
            'Please adjust these parameters so that '
            'generation_batch_size // data_parallel_world_size // num_generations // micro_batch_size >= 1.')
        self._num_iters_per_step = num_iters_per_step
        return num_iters_per_step

    def get_local_rollout_batch(self, batch):
        # repeat num_generations times
        rollout_group = self._get_rollout_group()
        global_rollout_batch = [deepcopy(item) for item in batch for _ in range(self.num_generations)]
        # get local rollout data
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        rollout_group_size = torch.distributed.get_world_size(group=rollout_group)

        per_device_batch_size = self.per_device_generation_batch_size
        assert rollout_group_size * per_device_batch_size == len(global_rollout_batch)
        data_slice = slice(rollout_rank * per_device_batch_size, (rollout_rank + 1) * per_device_batch_size)
        rollout_batch = global_rollout_batch[data_slice]
        return rollout_batch

    @contextmanager
    def _template_context(self, template: Template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        template.max_length = None
        try:
            yield
        finally:
            template.max_length = max_length

    def _prepare_metrics(self):
        args = self.args
        from swift.utils import JsonlWriter
        from collections import deque
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.jsonl_writer = JsonlWriter(os.path.join(args.save, 'completions.jsonl'), write_on_rank='last')
        self.init_custom_metric = False
        self._logs = {
            'prompt': deque(maxlen=args.generation_batch_size),
            'completion': deque(maxlen=args.generation_batch_size),
            'rewards': defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            'advantages': deque(maxlen=args.generation_batch_size),
        }
        if is_wandb_available():
            # when log profiling, the step is different from the step in the training loop
            # here patch wandb log to pop the step argument
            from wandb.sdk.wandb_run import Run
            origin_log = Run.log
            from functools import wraps

            @wraps(origin_log)
            def log(self, data: dict[str, Any], step: int | None = None, commit: bool | None = None):
                return origin_log(self, data, None, commit)

            Run.log = log

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    def _set_inputs_system(self, batch: DataType) -> DataType:
        """
        Ensures the system message is consistently set for all conversations in the batch.

        The template handles the user-defined system message. However, in server mode,
        tokenization occurs on the rollout side. To prevent a mismatch where the system
        message is set only during training but missing during rollout, this method
        injects the default system message into each conversation if not already present.

        Args:
            batch: A list of data items, each containing a 'messages' list.

        Returns:
            The updated batch with the default system message inserted at the beginning
            of each conversation that lacks one.
        """

        if self.vllm_mode != 'server':
            return batch

        # Return early if no default system message is defined
        if not self.template.template_meta.default_system:
            return batch

        # Return early if all conversations already start with a system message
        if all(data['messages'][0]['role'] == 'system' for data in batch):
            return batch

        # Insert the default system message at the beginning of each conversation
        # that doesn't already have one
        for data in batch:
            messages = data['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})

        return batch