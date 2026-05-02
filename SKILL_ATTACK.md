---
name: experiment-ii-attack-mechanisms
description: |
  实验二攻击机制详解
  - 基于实验一发现的三种差异化攻击策略
  - Soft-PRF 流行度惩罚型攻击
  - Point-Level 直接替换型攻击
  - Random-Shuffle 随机基线
keywords:
    - experiment-II
    - watermark breaking
    - point-level attack
    - popularity-based defense
    - data-aware vs data-unaware
---

# 实验二 攻击机制详解

## 核心背景（来自实验一）

### 实验一关键发现

从实验一.1 AOW水印异常分析，我们发现：

1. **COLD启动水印的异常流行度分布**
   - 初始物品 (item0-4) 来自底部冷池
   - 频率远低于普通推荐初始项
   - 模型logits排名与频率排名不一致
   
2. **级联破坏机制 (Cascade Effect)**
   - 破坏初始项 (item0) → 污染用户历史表示
   - 后续预测均受影响 → HR@5, @10也下降
   - COLD因初始项冷，容易被流行度攻击触发
   
3. **POP启动水印的抗攻击性**
   - 初始物品来自热池，频率与推荐一致
   - 需要更激进的攻击参数才能破坏
   - 级联破坏效应弱

---

## 三种差异化攻击策略

### 攻击策略1: Soft-PRF (流行度惩罚型)

#### 原理

```
z_i' = z_i - β * σ((r_i - γ) / ε)

其中:
  r_i     = 物品i的流行度排名 ∈ [0, 1]
  β       = 攻击强度参数
  γ       = 排名阈值(默认0.7)
  ε       = 平滑因子(默认0.02)
  σ()     = sigmoid函数
  
作用原理:
  - 低流行度物品 (r_i < γ) 受到更强的score惩罚
  - 高流行度物品 (r_i > γ) 受到较弱的score惩罚
  - 尤其针对COLD的初始冷物品有效
```

#### 流行度来源

| 来源 | 参数值 | 特点 |
|-----|-------|------|
| **Data-Aware** (data) | `--item_freq_source data` | 直接统计训练/验证/测试集中的交互频率，低成本，反映真实分布 |
| **Data-Unaware** (qee) | `--item_freq_source qee` | 通过查询水印模型的输出推断流行度，无需原始数据，模拟真实攻击场景 |
| **蒸馏估计** (dpe) | `--item_freq_source dpe` | 使用蒸馏模型的输出估计，适用于无预算/无数据场景 |
| **混合** (tpe) | `--item_freq_source tpe` | (1-α)*data + α*qee，平衡数据和模型 |

#### 参数调节

```bash
# 基础参数 (ml-1m COLD)
--prf_gamma 0.7      # 针对冷物品，gamma=0.7比较合适
--prf_beta 5.0       # 初始值，可根据需求调整
--prf_eps 0.02       # 平滑因子，控制sigmoid过渡的陡峭度

# 参数消融
beta ∈ {1.0, 2.0, 5.0, 10.0, 15.0, 20.0}  # 攻击强度扫描
gamma ∈ {0.3, 0.5, 0.7, 0.9}               # 排名阈值扫描
```

#### 预期效果

```
COLD + Data-Aware (soft_prf, beta=5.0):
  HR@1:  0.98 → 0.20 (ASR ≈ 80%)
  NDCG@10: 0.061 → 0.060 (ΔNDCG ≈ 0.2%)  ✓
  
COLD + Data-Unaware (soft_prf + QEE, beta=5.0):
  HR@1:  0.98 → 0.30 (ASR ≈ 70%)
  NDCG@10: 0.061 → 0.060 (ΔNDCG ≈ 0.2%)  ✓
  
POP + Data-Aware (soft_prf, beta=5.0):
  HR@1:  0.95 → 0.80 (ASR ≈ 16%)           # 弱
  NDCG@10: 0.061 → 0.060 (ΔNDCG ≈ 0.1%)  ✓
  # 需要 beta > 20 才能显著破坏
```

---

### 攻击策略2: Point-Level (直接替换型)

#### 原理

```
核心想法: 直接提升热门物品的分数，挤压冷物品的排名

实现:
  1. 识别top-k最热门的物品 (from item_freq)
  2. 为这k个物品的分数统一加上boost值
  3. 冷物品相对排名下降
  
z_i' = z_i + boost * mask_topk[i]

其中:
  mask_topk[i] = 1 如果 i ∈ top-k popular items
                = 0 其他

作用原理:
  - 比Soft-PRF更直接，显式替换初始物品
  - 对COLD的底部冷物品有强有力的效果
  - 计算成本低(只需一次排序和加法)
```

#### 参数配置

```bash
--attack point_level
--pl_top_k 50         # top-50热门物品获得boost (ml-1m项数3706，约1.3%)
--pl_boost 5.0        # boost幅度，类似soft_prf的beta
```

#### 预期效果

```
COLD + Data-Aware (point_level, pl_boost=5.0):
  HR@1:  0.98 → 0.15 (ASR ≈ 85%)
  NDCG@10: 0.061 → 0.059 (ΔNDCG ≈ 0.3%)  ✓
  
  # 比soft_prf的ASR更高，因为直接针对初始项替换

COLD + Data-Unaware (point_level + QEE, pl_boost=5.0):
  HR@1:  0.98 → 0.25 (ASR ≈ 75%)
  NDCG@10: 0.061 → 0.059 (ΔNDCG ≈ 0.3%)  ✓
```

---

### 攻击策略3: Random-Shuffle (随机基线)

#### 原理

```
为所有物品的分数加上高斯噪声，作为攻击的随机基线

z_i' = z_i + N(0, σ²)

其中:
  N(0, σ²) 是标准差为σ的高斯分布

作用原理:
  - 衡量水印对随机扰动的鲁棒性
  - 区分水印特定脆弱性 vs 一般鲁棒性不足
  - 如果 ASR(random) > 50%，说明模型本身不稳定
```

#### 参数配置

```bash
--attack random_shuffle
--rs_noise_scale 1.0   # 噪声标准差，等效的score扰动幅度
--rs_seed 42           # 可重现的随机种子
```

#### 预期效果

```
COLD + Random (rs_noise_scale=1.0):
  HR@1:  0.98 → 0.85 (ASR ≈ 13%)
  NDCG@10: 0.061 → 0.060 (ΔNDCG ≈ 0.2%)  ✓
  
  # ASR应显著低于 Soft-PRF 和 Point-Level
  # 说明water水印的脆弱是流行度特定的，非随机鲁棒性问题
```

---

## Data-Aware vs Data-Unaware 的权衡

| 维度 | Data-Aware | Data-Unaware |
|-----|-----------|--------------|
| **流行度来源** | 真实用户交互统计 | 模型查询推断 |
| **攻击强度** | 强 (ASR +10-15%) | 弱 |
| **隐私暴露** | 需要数据集，风险高 | 仅需模型query，风险低 |
| **实际可用性** | 白盒/灰盒场景 | 黑盒场景 |
| **参数敏感性** | 低，结果稳定 | 高，需调优query参数 |
| **推荐场景** | 防守方评估最坏情况 | 攻击方现实约束 |

### QEE (Query Exposure Estimation) 调优

```bash
# 基础配置
--item_freq_source qee
--freq_query_topk 20           # 每个query取top-20项
--freq_query_max_batches 0     # 使用全部query数据
--freq_query_temperature 1.0   # softmax温度(1.0=不改变)
--freq_query_uniform_mix 0.02  # 混入2%均匀分布(防止zero-hit)

# 调优策略
# 如果ASR(qee)和ASR(data)差距>20%:
  - 增加freq_query_topk (20→50) 获取更多context
  - 减小freq_query_temperature (1.0→0.8) 强化top-k
  - 增加freq_query_uniform_mix (0.02→0.05) 覆盖稀有物品
```

---

## COLD vs POP 的差异总结

基于实验一发现，整理对比表：

| 维度 | COLD序列 | POP序列 |
|-----|---------|--------|
| **初始物品性质** | 极冷(bottom-m%) | 热(top popularity) |
| **Soft-PRF所需β** | 5-10 | 30-50 |
| **Point-Level所需boost** | 3-5 | 20-30 |
| **级联破坏效应** | 强 ✓ | 弱 |
| **Data-Aware优势** | 显著 (Δ ASR +15%) | 弱 (Δ ASR +5%) |
| **攻击难度评级** | ⭐⭐ (简单) | ⭐⭐⭐⭐ (困难) |
| **防守建议** | 提高初始冷度 | 减少热度聚集度 |

---

## 快速验证清单

### 预期指标范围（针对ml-1m cold L=5）

```
Baseline (无攻击):
  ✓ HR@1 ∈ [0.95, 1.00]
  ✓ NDCG@10 ∈ [0.060, 0.065]
  
Soft-PRF (beta=5.0, gamma=0.7):
  ✓ HR@1 ∈ [0.15, 0.30]  (ASR ≈ 70-85%)
  ✓ NDCG@10 ∈ [0.059, 0.061]  (ΔNDCG < 1%)
  
Point-Level (boost=5.0, top_k=50):
  ✓ HR@1 ∈ [0.10, 0.25]  (ASR ≈ 75-90%)
  ✓ NDCG@10 ∈ [0.059, 0.061]  (ΔNDCG < 1%)
  
Random-Shuffle (noise_scale=1.0):
  ✓ HR@1 ∈ [0.80, 0.95]  (ASR ≈ 3-20%)
  ✓ NDCG@10 ∈ [0.059, 0.061]  (ΔNDCG < 1%)
```

### 标记成功的条件

- [ ] Baseline HR@1 > 0.90 (水印有效)
- [ ] Soft-PRF ASR > 60% (可破坏)
- [ ] Point-Level ASR > 70% (更强)
- [ ] ΔNDCG@10 < 1% (保持质量)
- [ ] COLD ASR > POP ASR * 3 (差异显著)
- [ ] Data-Aware ASR > Data-Unaware ASR + 10% (数据优势)

---

## 常见问题 & 调试

**Q1: HR@1为0不合理？**
A: 检查水印序列是否正确加载。见 gen_watermark_seq.py 的输出日志。

**Q2: ΔNDCG@10 > 1%，是否失败？**
A: 不一定。在激进攻击下(beta>20)略高于1%是正常的。重点看ASR是否达标。

**Q3: Soft-PRF vs Point-Level 哪个更好？**
A: 都是有效的。Point-Level更直观，Soft-PRF更平滑。建议都跑，对比结果。

**Q4: 是否需要对 ml-20m 和 beauty 重复？**
A: 是的。优先ml-1m确认框架，再推广到ml-20m验证泛化性，beauty可选。

