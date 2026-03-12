❯ もう一回確認、impactって、ref/cfを使ってどうやって求めるんだっけ 

● コードベースで正確に確認します。

● Searched for 1 pattern, read 1 file (ctrl+o to expand)     

● 処理の流れ:                                                                                                                                                                                                                   
   
  Step 1: ベースライン計測 (L170-176)                                                                                                                                                                                           
                                                                                                                                                                                                                              
  Xr（参照入力）をそのまま推論 → default_logit_diff                                                                                                                                                                             
  = logits_target / logits_sum  （正答トークンの相対確率）                                                                                                                                                                      

  Step 2: 活性値を記録

  Forward A: Xrを入力 → 全層の入力側活性値 Hr[layer] を保存
  Forward B: Xcを入力 → 全層の出力側活性値 Hc[layer][head] を保存

  Step 3: ヘッドごとにパッチング (L225-269)

  対象ヘッド (source_layer, source_head_idx) について:
    ・Xrを入力として推論
    ・ただし対象ヘッドの出力だけをHc（Xc由来）に差し替え
    ・それ以降の層はHr（Xr由来）で補正
    → cur_logit_diff を計測

  Step 4: Impact計算 (L267-268)

  impact = (cur_logit_diff - default_logit_diff) / default_logit_diff
  つまり 「そのヘッドだけXcの活性に差し替えたとき、正答確率がベースラインからどれだけ変化したか（%）」 です。

  なぜ混在が問題にならないか

  - Hr も default_logit_diff も Xr全体 から計算される
  - Hc も Xc全体 から計算される
  - ref/cf で同一の部分は Hr[head] ≈ Hc[head] → 差し替えても変化なし → impact ≈ 0
  - ref/cf で異なる部分（置換された用語）に反応するヘッドだけ Hr ≠ Hc → impact が大きくなる

  差分が医療用語の変化だけなら医療感度、推論表現の変化だけなら推論感度が測れます。

✻ Crunched for 35s

❯ 正答確率がベースラインからどれだけ変化したか（%）って、正答か誤答かどうやって評価するの？                                                                                                                                     
                                                                                            
● 正答・誤答の二値判定ではありません。compute_metricの中身を見ると:                                                                                                                                                             
                                                                                                                                                                                                                              
  # 最終トークン位置のlogitsから                                                                                                                                                                                                
  logits_target = logits[:, -1, target_token_ids]   # 正答トークン("a"等)のlogit                                                                                                                                              
  logits_sum    = logits[:, -1, record_token_ids].sum()  # 全選択肢("a","b","c","d","e")のlogit合計                                                                                                                             
                                                                                                                                                                                                                                
  return logits_target / logits_sum

  これは正答トークンの相対確率（正答のlogit / 全選択肢のlogit合計）です。

  具体例

  正答が "a" で選択肢が a,b,c,d,e の場合:

  ベースライン(Xr):  logit_a=10, sum=30 → metric = 0.333
  パッチ後:          logit_a=8,  sum=30 → metric = 0.267

  impact = (0.267 - 0.333) / 0.333 = -0.20 → -20%

  正答か誤答かを判定しているのではなく、「モデルが正答にどれだけ確信を持っているか」の変化率を見ています。パッチで確信が下がるヘッドほど、そのタスクに重要ということです。