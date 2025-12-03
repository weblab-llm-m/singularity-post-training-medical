# 概要
環境構築の方法を述べる。

cloneしたら.envを作り以下の形式で指定のAPIを挿入する
```
WANDB_API_KEY="api_key"
HF_TOKEN="api_key"
```

以下は後日こちらのみで完結するように書く。
# 以下のリンクを参考に環境構築を行う
[参考リンク](https://github.com/weblab-llm-m/swift-RL/tree/main)

コピーコマンドメモ
```
cp swift-RL/src/swift/patch_promptid.py swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp swift-RL/src/swift/igaku_plugin.py swift-RL/containers/ms-swift/examples/train/grpo/plugin
cp swift-RL/src/swift/qwen3_next.py swift-RL/containers/ms-swift/swift/megatron/model/gpt/qwen3_next.py
```

# データセット
swift-RLリポジトリでmessages + answer を含む *.jsonl を想定(swift-RLリポジトリのswift-RL/src/swift/data/prepare_data_v2.py実行し作成を行う