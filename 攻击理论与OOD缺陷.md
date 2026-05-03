# AOW 水印的 OOD 缺陷与推理期攻击理论（论文级说明）

> 版本：v1.0（写作版）
>
> 更新时间：2026-05-03
>
> 目标读者：CIKM/WWW/KDD 审稿人（推荐系统 + 模型水印）
>
> 适用代码仓库：AOW-main（本仓库的实现与实验日志）

---

## 0. 摘要（要点版）

AOW（Auto-regressive OOD Watermarking）通过向训练数据注入由 Oracle 生成的 OOD 序列，使水印模型记住“水印前缀 → 下一物品”的映射，以此作为所有权验证。本文档基于本仓库的**实验一（统计缺陷分析）**与**实验二（推理期攻击）**，给出两条结论链：

1) **OOD 水印的内生缺陷**：AOW 的 bottom-M 采样机制会在水印序列上引入稳定的统计异常，其中**流行度偏移是跨数据集最稳定的异常**；局部连贯性异常则呈现强数据集依赖。

2) **攻击理论（Attack Surface Matching）**：水印的初始化方式（Cold/Pop）决定其“自然脆弱面”。
- Cold 水印的脆弱面是**流行度统计异常** → 适配“流行度统计抑制（Soft-PRF-like）”
- Pop 水印的脆弱面是模型记住的**续接映射关系**（prefix→continuation）→ 适配“轨迹查询抑制（Trajectory Suppression）”

在黑盒推理期攻击中，我们只对 logits 做后处理，不改权重，不重训。

---

## 1. 背景与问题设定

### 1.1 AOW 水印机制（以本仓库实现为准）

AOW 的训练与评测流程在 [水印训练与测评总结.md](水印训练与测评总结.md) 中已有实现级总结。这里用论文叙述形式统一符号。

- 训练一个 Oracle 序列推荐模型 $f_\theta$（无水印）。
- 选定水印起始物品 $s_1$：
  - Cold：训练集中**最冷门**物品
  - Pop：训练集中**最热门**物品
- 使用 Oracle 自回归生成水印序列 $W=(s_1, s_2, \dots, s_L)$：每一步从 Oracle 对 $s_{1:t}$ 的输出 logits 的 bottom-$M$ 候选中随机选取 $s_{t+1}$（并避免重复）。
- 将若干“水印用户序列”注入训练集，训练水印模型 $g_\phi$，使其在水印前缀上具有高下一项命中率。

**验证任务（ownership verification）**：在水印评测集中随机取水印前缀 $s_{1:t}$，要求模型预测 $s_{t+1}$，以 Recall@K/NDCG@K 度量。

### 1.2 威胁模型与目标

本文档讨论的是“推理期黑盒攻击”——攻击者不改动模型参数，也不接触训练过程，仅在推理阶段对模型输出 scores/logits 做可控扰动。

- **攻击者可用信息**：
  - Data-Aware：拥有训练数据或至少能获得真实交互频率排名（能定位“最热门/最冷门”）
  - Data-Unaware：仅能通过模型 API 查询（不拥有训练数据）
- **攻击者能力**：能对模型输出 logits $\mathbf{z}$ 执行后处理得到 $\tilde{\mathbf{z}}=\mathcal{A}(\mathbf{z})$，再输出 Top-K。
- **目标约束**：
  - 破坏水印有效性（降低 watermark Recall@1 等）
  - 同时保持推荐效用几乎不变（商用约束：效用损失应很小，至少不能“靠毁效用”取胜）

---

## 2. AOW 的 OOD 缺陷：统计异常是“可被利用/可被检测”的攻击面

本节不是重新解释 AOW，而是基于实验一总结：AOW 的 OOD 水印会在真实用户序列分布之外形成统计可分离结构，这一结构既削弱“隐蔽性”，也为攻击提供了信号。

### 2.1 缺陷A：流行度分布的系统性偏移（跨数据集稳定）

实验一的流行度统计汇总在：
- [experiments/phase1_aow_characteristics/tables/exp1_1_popularity_summary.csv](experiments/phase1_aow_characteristics/tables/exp1_1_popularity_summary.csv)

我们使用“流行度分位数”衡量物品在全局交互频率排序中的位置（越小越冷门）。此外统计 bottom40% 冷门物品占比。

**关键证据（pattern_len=5，mean / bottom40_ratio_mean）：**

| 数据集 | 组别 | Cold 水印 mean_pop_rank | Cold bottom40 | Pop 水印 mean_pop_rank | Pop bottom40 | 真实序列 bottom40 |
|---|---:|---:|---:|---:|---:|---:|
| ml-1m | watermark | 0.2700 | 0.7233 | 0.3044 | 0.7483 | 0.0400 ~ 0.0483 |
| ml-20m | watermark | 0.3175 | 0.6850 | 0.3789 | 0.6583 | 0.0017 ~ 0.0083 |
| beauty | watermark | 0.3758 | 0.5300 | 0.5924 | 0.2833 | 0.0613 ~ 0.0933 |

**结论**：
- 无论 Cold 还是 Pop，AOW 的 bottom-M 采样都会将后续项强行推向“低分区域”，从而产生显著的冷门偏置。
- 其中 **Cold 更极端、更稳定**；Pop 在 beauty 上偏置相对弱（但仍明显高于真实序列）。

从攻防角度，这意味着：
- Watermark 是 OOD 的同时，也是**统计离群**的；攻击者可以在不理解水印细节的情况下，仅用流行度排序就进行针对性抑制或检测。

### 2.2 缺陷B：局部连贯性异常（强数据集依赖）

实验一的局部连贯性统计汇总在：
- [experiments/phase1_aow_characteristics/tables/exp1_2_coherence_summary.csv](experiments/phase1_aow_characteristics/tables/exp1_2_coherence_summary.csv)

该指标用相邻物品 embedding 余弦相似度衡量局部平滑性。

**关键证据（pattern_len=5，adj_cosine_mean_mean）：**

| 数据集 | 真实序列 | Cold 水印 | Pop 水印 | 现象 |
|---|---:|---:|---:|---|
| ml-1m | 0.1308 | -0.0128 | 0.0790 | 水印更“跳跃”（尤其 Cold） |
| ml-20m | 0.1300 | -0.0148 | 0.0539 | 水印更“跳跃” |
| beauty | 0.0813 | 0.3716 | 0.4083 | 水印反而更“局部一致” |

**结论**：局部连贯性异常不是跨数据集同向规律，不能把“水印更跳”当作通用前提。
- 对 ml-1m/ml-20m：连贯性差可作为辅助异常信号。
- 对 beauty：连贯性反而更强，说明数据结构/embedding 几何会改变“异常表现”。

因此，真正稳健、可推广的 OOD 缺陷证据是 **2.1 的流行度偏移**。

### 2.3 缺陷C：某些数据集上水印训练会显著毁伤效用（beauty 案例）

在 beauty 上，本仓库记录了 Oracle 的正常效用：
- [experiments/bert/beauty/logs/test_metrics.json](experiments/bert/beauty/logs/test_metrics.json)
  - Oracle NDCG@10 ≈ 0.007655

而 POP 水印模型在本仓库的评测框架（test_watermark_acc 的 util 指标）中效用为 $\sim 10^{-4}$ 量级：
- [experiments/watermark_test/method_pop/bert/beauty/0.1_1.0_5_100/logs/evaluation_results_summary.csv](experiments/watermark_test/method_pop/bert/beauty/0.1_1.0_5_100/logs/evaluation_results_summary.csv)
  - `attack=none` 时 util_NDCG@10 ≈ 0.000117

两者差距为数量级级别。这意味着在 beauty 上，“水印有效性”并非唯一问题：**水印注入可能改变了训练分布与优化目标，使模型总体推荐能力崩塌**。这也解释了为什么在 beauty 上许多攻击看起来对效用“没影响”——因为效用在水印训练阶段已经被毁伤。

---

## 3. 攻击理论：Attack Surface Matching（攻击面匹配原则）

### 3.1 原则陈述

**攻击面匹配原则**：AOW 水印的初始化方式（Cold vs Pop）决定了其最易被触达的脆弱面。

- **Cold 初始化**：水印链整体落在长尾区域，因而暴露出稳定的**统计异常**（流行度偏移）。
  - 攻击应利用“统计异常”——只要压制长尾候选，就能覆盖大部分水印链。

- **Pop 初始化**：虽然起点是热门，但后续项来自 bottom-M，可能跨越各流行度层级；此时“流行度”不再提供可靠定位。
  - Pop 水印的脆弱面在于：训练会让模型形成高置信度的**续接映射**（prefix→continuation）。
  - 攻击应利用“行为可探测性”——通过查询模型来重建这条映射，再做精准抑制。

这个原则对应 [实验二.md](实验二.md) 中的框架，但本文档会把“为什么成立”与“证据链”写得更像论文。

### 3.2 推理期 logits 后处理作为统一攻击载体

设模型对给定输入序列 $x$ 输出 logits 向量 $\mathbf{z}(x)\in\mathbb{R}^{|\mathcal{V}|}$。

我们考虑攻击为：

$$\tilde{\mathbf{z}}(x) = \mathbf{z}(x) - \lambda\cdot \mathbf{p}(x)$$

其中 $\mathbf{p}(x)$ 是一个非负惩罚向量（可依赖于 $x$），$\lambda$ 是强度。

- Cold 型攻击：$\mathbf{p}$ 主要由“流行度排名”决定（与 $x$ 弱相关）。
- Pop 型攻击：$\mathbf{p}$ 由“模型查询得到的轨迹集合”决定（与 $x$ 强相关）。

这一统一形式保证我们讨论的攻击都遵守同一约束：**不改权重、只改输出**。

---

## 4. Cold 水印攻击：流行度统计抑制（Soft-PRF-like）

### 4.1 机制直觉

结合 2.1 的稳定证据：Cold 水印（以及很多情况下的 Pop 水印）在物品流行度上显著偏向长尾。

于是一个低成本攻击是：对长尾候选施加平滑惩罚，从而让模型在水印前缀上更难继续命中“冷门续接项”。

### 4.2 形式化（平滑阈值惩罚）

设 $r(v)\in[0,1]$ 为物品 $v$ 在流行度排序中的分位（越小越冷门）。

一种典型的平滑惩罚可以写作：

$$p(v)=\sigma\left(\frac{\gamma-r(v)}{\varepsilon}\right),\quad \tilde{z}_v = z_v - \beta\, p(v)$$

其中 $\gamma$ 控制开始惩罚的分位阈值，$\varepsilon$ 控制平滑宽度，$\beta$ 为惩罚强度。

### 4.3 适用性与局限

- 优点：
  - **零/极低查询成本**（只需一份频率排序）
  - 易于调参，解释性强
- 局限：
  - 对 Pop 水印不一定有效：因为 Pop 的后续项流行度跨度大，统计抑制可能“压错方向”或导致效用明显下降。

---

## 5. Pop 水印攻击：轨迹查询抑制（Trajectory Suppression）

### 5.1 关键洞见

Pop 水印的核心不是“某一类物品（热门/冷门）”，而是模型记住了水印链条的**条件映射**：

$$s_{t+1} \approx \arg\max_v z_v(s_{1:t})$$

只要能定位“哪些前缀会触发水印续接”，就可以精准抑制这些续接项，而不影响大多数正常用户样本。

### 5.2 攻击算法（两层足够的原因）

以触发器集合 $T$ 开始（Data-Aware 时可取最热门物品；Data-Unaware 用 QEE 估计后取 top-K 并集），对每个 $t\in T$：

1) 查询 $[t,\text{MASK}]$，取 top-$k_1$ 候选集合 $S_1(t)$
2) 对每个 $u\in S_1(t)$，再查询 $[t,u,\text{MASK}]$，取 top-$k_2$ 集合 $S_2(t,u)$
3) 汇总目标集合 $\mathcal{U}=\bigcup_t S_1(t)\cup\bigcup_{t,u}S_2(t,u)$
4) 对 $\mathcal{U}$ 中物品统一施加惩罚（或按深度衰减加权）

直观上，AOW 在短链（如 $L=5$）下的“记忆强度”集中体现在最前两步续接，因此两层查询就足以暴露主要映射关系；更深层往往收益递减。

### 5.3 为什么能做到“低/零效用损失”

轨迹攻击的惩罚对象来自“以触发器为前缀”的局部续接路径，而正常测试样本中满足该前缀结构的概率低；同时这些被惩罚的水印续接项与正常用户真正需要的 next-item 重叠很小。

这是一种“窄域、条件化”的干预：**只在触发条件成立时才有效，且惩罚集合规模远小于全物品集**，因此更可能保效用。

---

## 6. Data-Unaware：QEE + 多触发器并集（范围覆盖代替精确定位）

### 6.1 动机

Data-Unaware 下，攻击者不知道真实最热门物品是谁。若只使用 QEE 的 top-1 作为触发器，一旦偏离真实触发器，整条轨迹都会“跑偏”。

解决办法不是追求 top-1 精确，而是用 **top-K 触发器并集** 做“范围覆盖”：只要真实触发器落在 top-K 内，轨迹集合就能覆盖到关键水印续接路径。

### 6.2 本仓库已验证的关键配置与效果（ml-1m）

来自：
- [experiments/watermark_test/method_pop/bert/ml-1m/0.1_1.0_5_100/logs/evaluation_results.json](experiments/watermark_test/method_pop/bert/ml-1m/0.1_1.0_5_100/logs/evaluation_results.json)

配置（摘要）：
- `attack=random_shuffle`，`rs_mode=trajectory`
- `item_freq_source=qee`
- `rs_traj_k1=3`，`rs_traj_k2=1`
- `rs_traj_penalty=20.0`
- `rs_traj_trigger_topk=20`

结果：
- watermark：Recall@1 = 0.2298677880（从 1.0 显著下降）
- utility：NDCG@10 = 0.1065860887（相对同脚本其它配置仅小幅变化）

这条结果是“Data-Unaware + 多触发器 + 轨迹抑制”可达成强破坏与低效用损失的直接证据。

### 6.3 Beauty 上的特殊性

来自：
- [experiments/watermark_test/method_pop/bert/beauty/0.1_1.0_5_100/logs/evaluation_results.json](experiments/watermark_test/method_pop/bert/beauty/0.1_1.0_5_100/logs/evaluation_results.json)

同样配置下：
- watermark：Recall@1 = 0.2298677880
- utility：NDCG@10 ≈ 0.0001257638（仍是 $10^{-4}$ 量级）

结合 2.3（Oracle beauty NDCG@10 ≈ 0.007655）可以看出：beauty 的主要问题是水印训练阶段的整体效用崩塌，而非攻击导致的额外损失。

---

## 7. 论文写作层面的“贡献点”组织方式（建议）

为了让审稿人觉得清晰且新颖，建议在论文中按以下逻辑组织（对应本仓库已完成证据链）：

1) **现象**：AOW 的 OOD 水印不仅 OOD，而且在统计上呈现稳定偏移（至少在流行度维度跨数据集稳定）。
2) **原则**：提出 Attack Surface Matching——初始化方式决定脆弱面。
3) **方法**：
   - Cold → 流行度统计抑制
   - Pop → 轨迹查询抑制
   - Data-Unaware → QEE + 多触发器并集
4) **证据**：
   - 实验一：流行度偏移（稳定）+ 连贯性异常（数据集依赖）
   - 实验二：ml-1m/beauty 的 Data-Unaware 轨迹攻击日志直接证明可把 Recall@1 压到 ~0.23 且效用基本保持（ml-1m 为小幅变化；beauty 的效用低由训练导致）

---

## 8. 局限性与可扩展方向（为后续 NIDW 铺垫）

- OOD 水印的隐蔽性-鲁棒性矛盾：越 OOD 越易被模型记住，也越易形成可被利用的统计异常。
- 轨迹攻击依赖可查询模型：但这是现实 API 场景常见假设；同时多触发器会提升查询成本，但仍远小于重训成本。
- beauty 的效用崩塌提示：AOW 的“水印注入比例/采样策略/训练方式”可能需要 Near-ID（NIDW）式修正，才能兼顾隐蔽性与效用。

---

## 附录A：本仓库关键证据文件索引

- 水印机制与评测：
  - [水印训练与测评总结.md](水印训练与测评总结.md)
- 实验一（OOD 缺陷统计）：
  - [实验一.md](实验一.md)
  - [experiments/phase1_aow_characteristics/tables/exp1_1_popularity_summary.csv](experiments/phase1_aow_characteristics/tables/exp1_1_popularity_summary.csv)
  - [experiments/phase1_aow_characteristics/tables/exp1_2_coherence_summary.csv](experiments/phase1_aow_characteristics/tables/exp1_2_coherence_summary.csv)
- 实验二（攻击框架与结果）：
  - [实验二.md](实验二.md)
  - ml-1m（POP, trajectory, QEE, top-20）：
    - [experiments/watermark_test/method_pop/bert/ml-1m/0.1_1.0_5_100/logs/evaluation_results.json](experiments/watermark_test/method_pop/bert/ml-1m/0.1_1.0_5_100/logs/evaluation_results.json)
  - beauty（POP, trajectory, QEE, top-20）：
    - [experiments/watermark_test/method_pop/bert/beauty/0.1_1.0_5_100/logs/evaluation_results.json](experiments/watermark_test/method_pop/bert/beauty/0.1_1.0_5_100/logs/evaluation_results.json)
- Oracle beauty 基线（用于说明效用崩塌）：
  - [experiments/bert/beauty/logs/test_metrics.json](experiments/bert/beauty/logs/test_metrics.json)
