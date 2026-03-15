#!/usr/bin/env python3
"""
Qwen3-30B-A3B-Instruct-2507 モデル構造分析スクリプト

このスクリプトはモデルの構造を詳細に分析し、
Pinpoint TuningおよびPath Patching実装に必要な情報を収集します。
"""

import json
import sys
from pathlib import Path

# モデルパス
MODEL_PATH = "/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"

def analyze_config():
    """config.jsonの分析"""
    print("=" * 80)
    print("1. モデル設定 (config.json)")
    print("=" * 80)

    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"model_type: {config.get('model_type')}")
    print(f"architectures: {config.get('architectures')}")
    print()

    print("■ レイヤー構造:")
    print(f"  num_hidden_layers: {config.get('num_hidden_layers')}")
    print(f"  hidden_size: {config.get('hidden_size')}")
    print(f"  intermediate_size: {config.get('intermediate_size')}")
    print()

    print("■ Attention構造:")
    print(f"  num_attention_heads (Query): {config.get('num_attention_heads')}")
    print(f"  num_key_value_heads (KV): {config.get('num_key_value_heads')}")
    print(f"  head_dim: {config.get('head_dim')}")
    num_q = config.get('num_attention_heads', 32)
    num_kv = config.get('num_key_value_heads', 4)
    print(f"  num_key_value_groups (計算値): {num_q // num_kv}")
    print(f"  全Attentionヘッド数: {config.get('num_hidden_layers')} × {num_q} = {config.get('num_hidden_layers') * num_q}")
    print()

    print("■ MoE構造:")
    print(f"  num_experts: {config.get('num_experts')}")
    print(f"  num_experts_per_tok: {config.get('num_experts_per_tok')}")
    print(f"  moe_intermediate_size: {config.get('moe_intermediate_size')}")
    print(f"  decoder_sparse_step: {config.get('decoder_sparse_step')}")
    print(f"  norm_topk_prob: {config.get('norm_topk_prob')}")
    print(f"  router_aux_loss_coef: {config.get('router_aux_loss_coef')}")
    print()

    print("■ その他の設定:")
    print(f"  max_position_embeddings: {config.get('max_position_embeddings')}")
    print(f"  rope_theta: {config.get('rope_theta')}")
    print(f"  torch_dtype: {config.get('torch_dtype')}")
    print(f"  vocab_size: {config.get('vocab_size')}")
    print()

    return config


def analyze_weight_structure():
    """重みファイルのインデックスを分析してモジュール構造を理解"""
    print("=" * 80)
    print("2. モジュール構造 (model.safetensors.index.json)")
    print("=" * 80)

    index_path = Path(MODEL_PATH) / "model.safetensors.index.json"
    with open(index_path, 'r') as f:
        index = json.load(f)

    weight_map = index.get('weight_map', {})

    # モジュール名をパースして構造を理解
    layer0_modules = set()
    all_module_patterns = set()

    for name in weight_map.keys():
        if name.startswith("model.layers.0."):
            # Layer 0のモジュール構造を抽出
            parts = name.replace("model.layers.0.", "").rsplit(".", 1)[0]
            layer0_modules.add(parts)

        # 全体のパターンを抽出
        if "layers" in name:
            # layers.X.*** から *** 部分を抽出
            parts = name.split("layers.")[1].split(".", 1)
            if len(parts) > 1:
                module_pattern = parts[1].rsplit(".", 1)[0]  # weightを除く
                all_module_patterns.add(module_pattern)

    print("■ Layer 0 のモジュール構造:")
    for module in sorted(layer0_modules):
        print(f"  - {module}")
    print()

    # Attention関連
    print("■ Attention関連モジュール:")
    attn_modules = [m for m in sorted(layer0_modules) if 'self_attn' in m]
    for m in attn_modules:
        print(f"  - model.layers.{{i}}.{m}")
    print()

    # MLP/Expert関連
    print("■ MLP/Expert関連モジュール:")
    mlp_modules = [m for m in sorted(layer0_modules) if 'mlp' in m]
    # expertsは多すぎるので代表例だけ
    expert_count = len([m for m in mlp_modules if 'experts' in m])
    non_expert_mlp = [m for m in mlp_modules if 'experts' not in m]

    for m in non_expert_mlp:
        print(f"  - model.layers.{{i}}.{m}")

    if expert_count > 0:
        print(f"  - model.layers.{{i}}.mlp.experts.{{j}}.down_proj")
        print(f"  - model.layers.{{i}}.mlp.experts.{{j}}.gate_proj")
        print(f"  - model.layers.{{i}}.mlp.experts.{{j}}.up_proj")
        print(f"    (全 {expert_count // 3} エキスパート × 3 パラメータ)")
    print()

    # LayerNorm関連
    print("■ LayerNorm関連モジュール:")
    norm_modules = [m for m in sorted(layer0_modules) if 'norm' in m.lower() or 'layernorm' in m.lower()]
    for m in norm_modules:
        print(f"  - model.layers.{{i}}.{m}")
    print()

    # Router (gate) の確認
    print("■ Router (Gate) モジュール:")
    if 'mlp.gate' in layer0_modules:
        print("  - model.layers.{i}.mlp.gate (エキスパート選択用ルーター)")
    print()


def analyze_transformers_module():
    """transformersライブラリのQwen3MoE実装を確認"""
    print("=" * 80)
    print("3. transformers ライブラリのモジュール構造")
    print("=" * 80)

    try:
        from transformers import AutoConfig, AutoModel
        import transformers
        print(f"transformers version: {transformers.__version__}")
        print()

        # 設定のみロード（メモリ節約）
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"Config class: {type(config).__name__}")
        print()

        # モデルクラスを確認
        from transformers import AutoModelForCausalLM
        model_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
        if model_class:
            print(f"Model class: {model_class}")

        # Qwen3MoE関連のモジュールを探す
        print("\n■ Qwen3MoE関連クラス:")
        try:
            from transformers.models.qwen3_moe import modeling_qwen3_moe
            print("  transformers.models.qwen3_moe モジュールが見つかりました")

            # 主要なクラスを列挙
            classes = [name for name in dir(modeling_qwen3_moe) if name.startswith('Qwen3Moe')]
            for cls_name in sorted(classes):
                print(f"  - {cls_name}")
        except ImportError as e:
            print(f"  Qwen3MoE モジュールのインポートエラー: {e}")
            print("  → transformersのバージョンアップが必要かもしれません")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def print_path_patching_info():
    """Path Patching実装に必要な情報をまとめる"""
    print("=" * 80)
    print("4. Path Patching 実装向け情報まとめ")
    print("=" * 80)

    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_layers = config.get('num_hidden_layers', 48)
    num_heads = config.get('num_attention_heads', 32)
    num_kv_heads = config.get('num_key_value_heads', 4)
    num_experts = config.get('num_experts', 128)
    num_experts_per_tok = config.get('num_experts_per_tok', 8)

    print("■ フック対象モジュール名:")
    print()
    print("  【Attention】")
    print(f"    - model.layers.{{0-{num_layers-1}}}.self_attn.q_proj")
    print(f"    - model.layers.{{0-{num_layers-1}}}.self_attn.k_proj")
    print(f"    - model.layers.{{0-{num_layers-1}}}.self_attn.v_proj")
    print(f"    - model.layers.{{0-{num_layers-1}}}.self_attn.o_proj")
    print(f"    - model.layers.{{0-{num_layers-1}}}.self_attn.q_norm  ← Qwen3で追加")
    print(f"    - model.layers.{{0-{num_layers-1}}}.self_attn.k_norm  ← Qwen3で追加")
    print()
    print("  【MoE/MLP】")
    print(f"    - model.layers.{{0-{num_layers-1}}}.mlp.gate  ← Router層")
    print(f"    - model.layers.{{0-{num_layers-1}}}.mlp.experts.{{0-{num_experts-1}}}.gate_proj")
    print(f"    - model.layers.{{0-{num_layers-1}}}.mlp.experts.{{0-{num_experts-1}}}.up_proj")
    print(f"    - model.layers.{{0-{num_layers-1}}}.mlp.experts.{{0-{num_experts-1}}}.down_proj")
    print()
    print("  【LayerNorm】")
    print(f"    - model.layers.{{0-{num_layers-1}}}.input_layernorm")
    print(f"    - model.layers.{{0-{num_layers-1}}}.post_attention_layernorm")
    print()

    print("■ ヘッド分析用パラメータ:")
    print(f"  num_layers: {num_layers}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  num_key_value_groups: {num_heads // num_kv_heads}")
    print(f"  全Attentionヘッド数: {num_layers * num_heads}")
    print()

    print("■ MoE分析用パラメータ:")
    print(f"  num_experts: {num_experts}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  全Expert数: {num_layers * num_experts}")
    print()


def print_freeze_modules_info():
    """freeze_modules実装向け情報"""
    print("=" * 80)
    print("5. Pinpoint Tuning freeze_modules 実装向け情報")
    print("=" * 80)

    print("■ Attention Head単位でのfreeze/unfreeze対象:")
    print("""
    Qwen3-14B (Dense)との違い:
    - GQA比率: 5 → 8 (8つのQueryヘッドが1つのKVヘッドを共有)
    - ヘッド数: 40 → 32
    - q_norm, k_normが追加

    freeze_modules関数の修正点:
    1. num_key_value_groups の計算を 32 // 4 = 8 に更新
    2. q_norm, k_norm の処理を追加（通常はfreezeのまま）
    """)

    print("■ MoE固有のfreeze戦略:")
    print("""
    1. Router層 (mlp.gate):
       - 基本的にfreezeを推奨（エキスパート選択の安定性維持）
       - 学習させる場合は慎重に

    2. Expert層 (mlp.experts.{j}):
       - 全エキスパートをfreezeするか
       - 選択的にunfreezeするか（どのエキスパートが重要か分析が必要）
       - Path Patchingで重要なexpertsを特定できる可能性

    3. 推奨設定:
       - freeze_router: True
       - train_experts: False (まずはAttentionヘッドのみ)
    """)


def generate_module_config():
    """MoE用モジュール設定ファイルを生成"""
    print("=" * 80)
    print("6. 生成: configs/qwen3_moe.json")
    print("=" * 80)

    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    moe_config = {
        "model_type": "qwen3_moe",
        "model_name": "Qwen3-30B-A3B-Instruct-2507",
        "architecture": {
            "num_hidden_layers": model_config.get('num_hidden_layers'),
            "num_attention_heads": model_config.get('num_attention_heads'),
            "num_key_value_heads": model_config.get('num_key_value_heads'),
            "num_key_value_groups": model_config.get('num_attention_heads') // model_config.get('num_key_value_heads'),
            "head_dim": model_config.get('head_dim'),
            "hidden_size": model_config.get('hidden_size'),
            "num_experts": model_config.get('num_experts'),
            "num_experts_per_tok": model_config.get('num_experts_per_tok'),
            "moe_intermediate_size": model_config.get('moe_intermediate_size'),
        },
        "module_names": {
            "attention": {
                "q_proj": "model.layers.{layer}.self_attn.q_proj",
                "k_proj": "model.layers.{layer}.self_attn.k_proj",
                "v_proj": "model.layers.{layer}.self_attn.v_proj",
                "o_proj": "model.layers.{layer}.self_attn.o_proj",
                "q_norm": "model.layers.{layer}.self_attn.q_norm",
                "k_norm": "model.layers.{layer}.self_attn.k_norm",
            },
            "mlp": {
                "router": "model.layers.{layer}.mlp.gate",
                "expert_gate": "model.layers.{layer}.mlp.experts.{expert}.gate_proj",
                "expert_up": "model.layers.{layer}.mlp.experts.{expert}.up_proj",
                "expert_down": "model.layers.{layer}.mlp.experts.{expert}.down_proj",
            },
            "layernorm": {
                "input": "model.layers.{layer}.input_layernorm",
                "post_attn": "model.layers.{layer}.post_attention_layernorm",
            },
        },
        "path_patching": {
            "attention_heads": {
                "total": model_config.get('num_hidden_layers') * model_config.get('num_attention_heads'),
                "per_layer": model_config.get('num_attention_heads'),
            },
            "experts": {
                "total": model_config.get('num_hidden_layers') * model_config.get('num_experts'),
                "per_layer": model_config.get('num_experts'),
            },
        },
        "freeze_defaults": {
            "freeze_router": True,
            "freeze_experts": True,
            "freeze_embeddings": True,
            "freeze_lm_head": True,
            "freeze_layernorm": True,
        }
    }

    output_path = Path(__file__).parent / "configs" / "qwen3_moe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(moe_config, f, indent=2)

    print(f"設定ファイルを生成しました: {output_path}")
    print()
    print("生成内容:")
    print(json.dumps(moe_config, indent=2))


def main():
    print("\n" + "=" * 80)
    print(" Qwen3-30B-A3B-Instruct-2507 モデル構造分析レポート")
    print("=" * 80 + "\n")

    # 1. config.json分析
    config = analyze_config()

    # 2. 重み構造分析
    analyze_weight_structure()

    # 3. transformersモジュール確認
    analyze_transformers_module()

    # 4. Path Patching情報
    print_path_patching_info()

    # 5. freeze_modules情報
    print_freeze_modules_info()

    # 6. 設定ファイル生成
    generate_module_config()

    print("\n" + "=" * 80)
    print(" 分析完了")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
