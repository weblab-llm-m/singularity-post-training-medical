"""
Attention Extractor
Path patching実行中に各ヘッドの注意パターンを抽出

Dense/MoE両対応:
- num_layers, num_heads はモデルに応じてコンストラクタで指定
- GQA (num_kv_heads) 自動検出
- MoE情報の自動検出
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import os


class AttentionExtractor:
    """注意パターン抽出クラス（Dense/MoE両対応）"""

    def __init__(
        self,
        model,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None
    ):
        """
        初期化

        Args:
            model: Transformerモデル
            num_layers: レイヤー数
            num_heads: Query ヘッド数
            num_kv_heads: KV ヘッド数 (default: None, モデルから自動取得)
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads

        # GQA設定
        if num_kv_heads is None:
            self.num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
        else:
            self.num_kv_heads = num_kv_heads

        self.num_kv_groups = num_heads // self.num_kv_heads

        self.attention_patterns = {}  # {layer_idx: tensor}
        self.hooks = []

        # MoE情報の自動検出
        self.is_moe = getattr(model.config, 'model_type', '') == 'qwen3_moe'
        self.num_experts = getattr(model.config, 'num_experts', 0)

        print(f"AttentionExtractor initialized (MoE: {self.is_moe})")
        print(f"  Layers: {num_layers}")
        print(f"  Query Heads: {num_heads}")
        print(f"  KV Heads: {self.num_kv_heads}")
        print(f"  KV Groups: {self.num_kv_groups}")
        if self.is_moe:
            print(f"  Experts: {self.num_experts}")

    def add_attention_hooks(self) -> List:
        """
        各レイヤーのattentionモジュールにフックを追加

        Returns:
            hooks: フックのリスト
        """
        hooks = []

        for layer_idx in range(self.num_layers):
            module = self.model.model.layers[layer_idx].self_attn

            def hook_fn(module, input, output, layer_idx=layer_idx):
                """
                Attention出力をキャプチャ

                出力形式:
                output = (attn_output, attn_weights, past_key_value)
                attn_weights: [batch, num_heads, seq_len, seq_len]
                """
                # 注意重みを取得
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]  # [batch, num_heads, seq_len, seq_len]

                    if attn_weights is not None:
                        # END位置（最終トークン）の注意パターンのみ保存
                        # [batch, num_heads, seq_len]
                        end_attention = attn_weights[:, :, -1, :].detach().cpu()

                        # 保存
                        if layer_idx not in self.attention_patterns:
                            self.attention_patterns[layer_idx] = []

                        self.attention_patterns[layer_idx].append(end_attention)

            # フックを登録
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        self.hooks = hooks
        print(f"Added attention hooks to {len(hooks)} layers")

        return hooks

    def remove_hooks(self):
        """フックを削除"""
        for hook in self.hooks:
            hook.remove()

        self.hooks = []
        print("Removed all attention hooks")

    def extract_and_save(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        save_path: str
    ):
        """
        注意パターンを抽出して保存

        Args:
            input_ids: 入力トークンID [batch, seq_len]
            attention_mask: 注意マスク [batch, seq_len]
            save_path: 保存先パス
        """
        # 注意パターンをクリア
        self.attention_patterns = {}

        # モデル設定を一時的に変更して注意重みを出力
        original_output_attentions = self.model.config.output_attentions
        self.model.config.output_attentions = True

        # フックを追加
        self.add_attention_hooks()

        # フォワード実行
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device)
            )

        # フックを削除
        self.remove_hooks()

        # モデル設定を元に戻す
        self.model.config.output_attentions = original_output_attentions

        # データを整形
        formatted_patterns = {}

        for layer_idx in range(self.num_layers):
            if layer_idx in self.attention_patterns:
                # リストをconcatenate
                layer_patterns = torch.cat(self.attention_patterns[layer_idx], dim=0)
                formatted_patterns[layer_idx] = layer_patterns

        # 保存
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(formatted_patterns, save_path)

        print(f"Attention patterns saved to: {save_path}")
        print(f"  Extracted from {len(formatted_patterns)} layers")

        return formatted_patterns

    def get_attention_for_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_idx: int = 0
    ) -> Dict[int, torch.Tensor]:
        """
        単一サンプルの注意パターンを取得

        Args:
            input_ids: 入力トークンID [batch, seq_len]
            attention_mask: 注意マスク [batch, seq_len]
            sample_idx: サンプルインデックス

        Returns:
            patterns: {layer_idx: [num_heads, seq_len]}
        """
        # 注意パターンをクリア
        self.attention_patterns = {}

        # モデル設定を一時的に変更
        original_output_attentions = self.model.config.output_attentions
        self.model.config.output_attentions = True

        # フックを追加
        self.add_attention_hooks()

        # フォワード実行
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device)
            )

        # フックを削除
        self.remove_hooks()

        # モデル設定を元に戻す
        self.model.config.output_attentions = original_output_attentions

        # サンプルの注意パターンを抽出
        sample_patterns = {}

        for layer_idx in range(self.num_layers):
            if layer_idx in self.attention_patterns:
                # [batch, num_heads, seq_len] → [num_heads, seq_len]
                layer_pattern = self.attention_patterns[layer_idx][0][sample_idx]
                sample_patterns[layer_idx] = layer_pattern

        return sample_patterns

    def analyze_attention_statistics(
        self,
        attention_patterns: Dict[int, torch.Tensor]
    ) -> Dict:
        """
        注意パターンの統計を分析

        Args:
            attention_patterns: {layer_idx: [batch, num_heads, seq_len]}

        Returns:
            statistics: 統計情報
        """
        stats = {
            'mean_attention_per_layer': {},
            'std_attention_per_layer': {},
            'max_attention_per_layer': {},
            'entropy_per_layer': {}
        }

        for layer_idx, pattern in attention_patterns.items():
            mean_attn = pattern.mean().item()
            std_attn = pattern.std().item()
            max_attn = pattern.max().item()

            # エントロピー計算
            eps = 1e-10
            entropy = -(pattern * torch.log(pattern + eps)).sum(dim=-1).mean().item()

            stats['mean_attention_per_layer'][layer_idx] = mean_attn
            stats['std_attention_per_layer'][layer_idx] = std_attn
            stats['max_attention_per_layer'][layer_idx] = max_attn
            stats['entropy_per_layer'][layer_idx] = entropy

        return stats

    def print_attention_summary(self, attention_patterns: Dict[int, torch.Tensor]):
        """
        注意パターンのサマリーを出力

        Args:
            attention_patterns: {layer_idx: [batch, num_heads, seq_len]}
        """
        model_type = "MoE" if self.is_moe else "Dense"
        print("\n" + "="*60)
        print(f"Attention Patterns Summary ({model_type} Model)")
        print("="*60)

        for layer_idx in sorted(attention_patterns.keys())[:5]:
            pattern = attention_patterns[layer_idx]
            print(f"\nLayer {layer_idx}:")
            print(f"  Shape: {pattern.shape}")
            print(f"  Mean attention: {pattern.mean().item():.4f}")
            print(f"  Std attention: {pattern.std().item():.4f}")
            print(f"  Max attention: {pattern.max().item():.4f}")

        if len(attention_patterns) > 5:
            print(f"\n... and {len(attention_patterns) - 5} more layers")

        print("="*60)
