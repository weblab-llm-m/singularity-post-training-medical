# Phase3 閾値チューニング履歴

     ## モデル情報
     - **モデル**: Qwen3-30B-A3B-Instruct-2507
     - **総ヘッド数**: 1,536 (48レイヤー × 32ヘッド)
     - **サンプル数**: 256 (ランダム抽出)
     - **参考**: Qwen14B (Medical=144/1600, 9.0%)

     ## 実測した注意スコアの統計値

     ### Medical Term positions (平均20.4位置/サンプル)
     | 統計量 | 値 |
     |--------|------|
     | min | 0.001701 |
     | p25 | 0.002609 |
     | median | 0.003174 |
     | p75 | 0.003723 |
     | p90 | 0.004181 |
     | p95 | 0.004333 |
     | max | 0.004639 |

     ### Profile Indicator positions (平均3.0位置/サンプル)
     | 統計量 | max_attn | spike_ratio |
     |--------|----------|-------------|
     | mean | 0.2686 | 10,140 |
     | p75 | 0.3867 | 741.7 |
     | p90 | 0.4814 | 1,970.7 |
     | p95 | 0.5127 | 3,075.0 |

     ### Reasoning Flow positions (平均4.0位置/サンプル)
     | 統計量 | std | mean | relative_std (std/mean) |
     |--------|-----|------|------------------------|
     | min | 0.000984 | 0.001640 | — |
     | p10 | 0.005035 | 0.001747 | 2.3785 |
     | median | 0.013702 | 0.001968 | 6.9844 |
     | mean | 0.014245 | 0.001961 | 7.5908 |

     ---

     ## 閾値チューニング履歴

     ### 試行1: Qwen14B閾値をそのまま適用

     | パラメータ | 設定値 | 根拠 |
     |-----------|--------|------|
     | medical_term.threshold | 0.002 | Qwen14Bの既定値 |
     | profile_indicator.spike_threshold | 0.005 | Qwen14Bの既定値 |
     | profile_indicator.spike_ratio | 2.0 | Qwen14Bの既定値 |
     | reasoning_flow.uniformity_threshold | 0.002 | Qwen14Bの既定値 |
     | reasoning_flow.attention_mean_threshold | 0.0005 | Qwen14Bの既定値 |
     | reasoning_flow.relative_std_threshold | 1.2 | Qwen14Bの既定値 |

     | 結果 | ヘッド数 | 割合 | 問題 |
     |------|---------|------|------|
     | Medical Term | 1,444 | 94.0% | **閾値(0.002)が実測min(0.0017)に近く、ほぼ全ヘッドが通過** |
     | Profile Indicator | 92 | 6.0% | 残りの少数が該当 |
     | Reasoning Flow | 0 | 0.0% | **std閾値(0.002)が実測min(0.001)より小さく、全ヘッドが不通過** |
     | Unclassified | 0 | 0.0% | Medicalに吸い込まれて残りなし |

     **根本原因**: 30BモデルはMedical位置が多く(平均20.4)、どのヘッドでも一定の注意が分散するため、低閾値では全通過する。Reasoning位置は少なく(平均4.0)、注意の均一性を保つヘッドが存在しない。

     ---

     ### 試行2: 実測値に基づく初回調整

     | パラメータ | 設定値 | 根拠 |
     |-----------|--------|------|
     | medical_term.threshold | 0.0042 | 実測p90(0.00418)付近 → 上位10%のみ |
     | profile_indicator.spike_threshold | 0.10 | max_attnの下位を除外 |
     | profile_indicator.spike_ratio | 50.0 | spike_ratio mean=10140に対し大幅に低い設定 |
     | reasoning_flow.uniformity_threshold | 0.02 | 実測median(0.014)を上回る緩い閾値 |
     | reasoning_flow.attention_mean_threshold | 0.0018 | 実測median(0.00197)やや下 |
     | reasoning_flow.relative_std_threshold | 5.0 | 実測p10(2.38)を大きく上回る |

     | 結果 | ヘッド数 | 割合 | 問題 |
     |------|---------|------|------|
     | Medical Term | 143 | 9.3% | 適切 |
     | Profile Indicator | 1,133 | 73.8% | **spike条件が緩すぎて大多数のヘッドが該当** |
     | Reasoning Flow | 260 | 16.9% | Profileに先に取られて少数 |
     | Unclassified | 0 | 0.0% | Profile+Reasoningで残り全吸収 |

     **根本原因**: Profile位置が少数(平均3.0)のため、どのヘッドでも特定位置に高い注意スパイクが存在。spike_threshold=0.10, spike_ratio=50は実測分布(max_attn mean=0.27, ratio mean=10140)に対して極めて緩い。

     ---

     ### 試行3: グリッドサーチによる最適化

     9パターンの閾値組合せをシミュレーションし、Qwen14Bのバランスに近い設定を探索。

     | 組合せ | M | P | R | U | 評価 |
     |--------|---|---|---|---|------|
     | med>0.004 prof>0.35/r>300 reas<0.008/rel<2.5 | 233(15%) | 247(16%) | 378(25%) | 678(44%) | Mがやや多い |
     | med>0.004 prof>0.4/r>500 reas<0.01/rel<3.0 | 233(15%) | 115(7%) | 469(31%) | 719(47%) | Mがやや多い |
     | med>0.0042 prof>0.35/r>300 reas<0.01/rel<3.0 | 143(9%) | 337(22%) | 469(31%) | 587(38%) | Pがやや多い |
     | **med>0.0042 prof>0.4/r>500 reas<0.01/rel<3.0** | **143(9%)** | **205(13%)** | **469(31%)** | **719(47%)** | **採用** |
     | med>0.0042 prof>0.4/r>500 reas<0.012/rel<3.5 | 143(9%) | 205(13%) | 638(42%) | 550(36%) | Rが多すぎ |
     | med>0.0043 prof>0.4/r>500 reas<0.01/rel<3.0 | 92(6%) | 256(17%) | 469(31%) | 719(47%) | Mが少なすぎ |

     ---

     ### 試行3 採用設定 (最終)

     | パラメータ | 設定値 | 根拠 |
     |-----------|--------|------|
     | medical_term.threshold | 0.0042 | 実測p90付近。Qwen14Bと同じ9%水準 |
     | profile_indicator.spike_threshold | 0.40 | 実測p75(0.387)付近。上位25%のスパイクのみ |
     | profile_indicator.spike_ratio | 500.0 | 実測p75(741)の下。明確なスパイクのみ通過 |
     | reasoning_flow.uniformity_threshold | 0.010 | 実測median(0.014)の下。下位30%の均一ヘッド |
     | reasoning_flow.attention_mean_threshold | 0.002 | 実測median(0.00197)付近。十分な注意量 |
     | reasoning_flow.relative_std_threshold | 3.0 | 実測p10(2.38)をやや上回る。均一なヘッドを捕捉 |
     | reasoning_flow.adjacent_window | 3 | Qwen14Bと同一 |

     | 結果 | ヘッド数 | 割合 |
     |------|---------|------|
     | Medical Term | 143 | 9.3% |
     | Profile Indicator | 205 | 13.3% |
     | Reasoning Flow | 469 | 30.5% |
     | Unclassified | 719 | 46.8% |

     **判定**: Medical=143はQwen14B(144)とほぼ一致。Profile/Reasoningも妥当な比率。Unclassified約47%は汎用ヘッドとして期待通り。

     ---

     ## 分類の優先順位

     ```
     1. Medical Term Head?      → 医療用語位置への平均注意スコアが閾値超
     2. Profile Indicator Head?  → 患者属性位置への最大注意スパイクが閾値超 かつ spike比率が閾値超
     3. Reasoning Flow Head?     → 推論位置+隣接位置の注意が均一 (絶対基準 OR 相対基準)
     4. Unclassified             → 上記いずれにも該当しない
     ```

     最初にマッチした分類が適用される(優先順位方式)。
