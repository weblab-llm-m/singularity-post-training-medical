#!/usr/bin/env python3
"""
Qwen3-30B-A3B モデル構造検証スクリプト

実際にモデルをロードして構造を確認し、
Path PatchingとPinpoint Tuningの実装が正しいことを検証します。
"""

import sys
import os
import json

# 基本インポート
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_PATH = "/home/yuuki.nakamura/downloads/models/Qwen_Qwen3-30B-A3B-Instruct-2507"


def verify_config():
    """モデル設定を検証"""
    print("=" * 60)
    print("1. モデル設定の検証")
    print("=" * 60)

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"\nmodel_type: {config.model_type}")
    print(f"architectures: {config.architectures}")

    # 期待値との比較
    expected = {
        'model_type': 'qwen3_moe',
        'num_hidden_layers': 48,
        'num_attention_heads': 32,
        'num_key_value_heads': 4,
        'num_experts': 128,
        'num_experts_per_tok': 8,
    }

    print("\n■ 設定値の検証:")
    all_match = True
    for key, expected_value in expected.items():
        actual_value = getattr(config, key, None)
        status = "✓" if actual_value == expected_value else "✗"
        if actual_value != expected_value:
            all_match = False
        print(f"  {status} {key}: {actual_value} (expected: {expected_value})")

    return config, all_match


def verify_tokenizer():
    """トークナイザーを検証"""
    print("\n" + "=" * 60)
    print("2. トークナイザーの検証")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"\nvocab_size: {tokenizer.vocab_size}")
    print(f"model_max_length: {tokenizer.model_max_length}")
    print(f"pad_token: {tokenizer.pad_token}")
    print(f"eos_token: {tokenizer.eos_token}")

    # テストエンコード
    test_text = "産婦人科ガイドラインに基づく質問です。"
    tokens = tokenizer.encode(test_text)
    print(f"\nテストエンコード: '{test_text}'")
    print(f"  トークン数: {len(tokens)}")
    print(f"  トークンID: {tokens[:10]}...")

    return tokenizer


def verify_model_structure(config):
    """モデル構造を検証（軽量ロード）"""
    print("\n" + "=" * 60)
    print("3. モデル構造の検証")
    print("=" * 60)

    print("\nモデルをロード中（メタデバイス使用）...")

    # メタデバイスでロード（メモリ節約）
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print(f"\nモデルクラス: {type(model).__name__}")

    # レイヤー構造を検証
    print("\n■ レイヤー構造:")
    print(f"  model.model.layers の数: {len(model.model.layers)}")

    # 最初のレイヤーの構造
    layer0 = model.model.layers[0]
    print(f"\n■ Layer 0 の構造:")

    # Attention
    print("  [Attention]")
    attn = layer0.self_attn
    print(f"    q_proj: {attn.q_proj}")
    print(f"    k_proj: {attn.k_proj}")
    print(f"    v_proj: {attn.v_proj}")
    print(f"    o_proj: {attn.o_proj}")

    # QK Norm (Qwen3で追加)
    if hasattr(attn, 'q_norm'):
        print(f"    q_norm: {attn.q_norm}")
    if hasattr(attn, 'k_norm'):
        print(f"    k_norm: {attn.k_norm}")

    # MoE/MLP
    print("  [MLP/MoE]")
    mlp = layer0.mlp
    print(f"    MLP class: {type(mlp).__name__}")

    # Router
    if hasattr(mlp, 'gate'):
        print(f"    gate (router): {mlp.gate}")

    # Experts
    if hasattr(mlp, 'experts'):
        print(f"    experts の数: {len(mlp.experts)}")
        print(f"    experts[0]: {type(mlp.experts[0]).__name__}")

        # Expert内部構造
        expert0 = mlp.experts[0]
        if hasattr(expert0, 'gate_proj'):
            print(f"      gate_proj: {expert0.gate_proj}")
        if hasattr(expert0, 'up_proj'):
            print(f"      up_proj: {expert0.up_proj}")
        if hasattr(expert0, 'down_proj'):
            print(f"      down_proj: {expert0.down_proj}")

    # Shared Expert
    if hasattr(mlp, 'shared_expert'):
        print(f"    shared_expert: {mlp.shared_expert}")
    else:
        print("    shared_expert: なし")

    # LayerNorm
    print("  [LayerNorm]")
    print(f"    input_layernorm: {layer0.input_layernorm}")
    print(f"    post_attention_layernorm: {layer0.post_attention_layernorm}")

    return model


def verify_module_names(model):
    """モジュール名が正しくアクセスできるか検証"""
    print("\n" + "=" * 60)
    print("4. モジュール名アクセスの検証")
    print("=" * 60)

    test_modules = [
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.47.self_attn.o_proj",  # 最終層
    ]

    print("\n■ モジュールアクセステスト:")
    all_accessible = True

    for module_name in test_modules:
        try:
            module = model.get_submodule(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            all_accessible = False

    return all_accessible


def verify_hook_functions():
    """フック関数のインポートを検証"""
    print("\n" + "=" * 60)
    print("5. フック関数のインポート検証")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from Phase2_path_patching.hook_functions_moe import (
            add_pre_module_hook,
            add_pre_module_hook_single_head,
            MoEPathPatchingHooks
        )
        print("  ✓ hook_functions_moe インポート成功")
        return True
    except Exception as e:
        print(f"  ✗ hook_functions_moe インポートエラー: {e}")
        return False


def verify_freeze_modules():
    """freeze_modules関数のインポートを検証"""
    print("\n" + "=" * 60)
    print("6. freeze_modules のインポート検証")
    print("=" * 60)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from Phase5_pinpoint_tuning.utils.utils_spt import (
            freeze_modules_moe,
            load_trainable_heads_from_json
        )
        print("  ✓ utils_spt インポート成功")
        return True
    except Exception as e:
        print(f"  ✗ utils_spt インポートエラー: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print(" Qwen3-30B-A3B モデル構造検証")
    print("=" * 60)
    print(f"\nモデルパス: {MODEL_PATH}")

    results = {}

    # 1. 設定検証
    config, config_ok = verify_config()
    results['config'] = config_ok

    # 2. トークナイザー検証
    tokenizer = verify_tokenizer()
    results['tokenizer'] = tokenizer is not None

    # 3. モデル構造検証
    model = verify_model_structure(config)
    results['model_structure'] = model is not None

    # 4. モジュール名検証
    results['module_access'] = verify_module_names(model)

    # 5. フック関数検証
    results['hook_functions'] = verify_hook_functions()

    # 6. freeze_modules検証
    results['freeze_modules'] = verify_freeze_modules()

    # 結果サマリー
    print("\n" + "=" * 60)
    print(" 検証結果サマリー")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 60)
    if all_passed:
        print(" 全てのテストに合格しました！")
    else:
        print(" 一部のテストが失敗しました。")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
