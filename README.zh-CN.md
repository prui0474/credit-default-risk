[English](README.md) · **简体中文**

# 信贷违约风险模型：33 万笔房贷的 PD 评分卡

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-3B6D11.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-10%2F10-1D9E75.svg)](#测试)

Freddie Mac 公开了逐笔房贷数据，按季度更新。这个项目用了 2015–2021 七个 vintage，共 331,122 笔贷款、18,696,124 条月度还款记录，原始文本 2.0 GB。用两个有监督的机器学习模型做对照：WOE 分箱后的 L2 正则 logistic 回归评分卡，和深度 4 的梯度提升决策树集成。按放款年份留出 2020–2021 做时间外测试，AUC 0.7462。

| | |
|---|---|
| 数据 | Freddie Mac Single-Family Loan-Level，2015–2021 共 7 个 vintage |
| 规模 | 331,122 笔贷款，18,696,124 条月度记录，2.0 GB 原始文本 |
| 时间外 AUC | 0.7462（Gini 0.4924，KS 0.3754），测试集为 2020–2021 放款 |
| 模型 | 14 个放款时点变量做 WOE 变换，L2 正则 logistic 评分卡，另配深度 4 的梯度提升决策树作对照 |
| 60 秒复现 | `python scripts/run_pipeline.py`，用内置合成数据，不下数据也不用授权 |
| 测试 | 10 项全过，其中 5 项钉在真实文件格式上 |
| 试用 | [在线 demo](https://prui0474.github.io/credit-default-risk/zh.html)（[English](https://prui0474.github.io/credit-default-risk/)）：拖 FICO、DTI、LTV 看 PD 怎么变。评分卡在浏览器里重写了一遍，拟合好的权重以 18 KB 的 JSON 发到前端 |

> 想把项目从零读一遍，看[《项目全景与知识点手册》](docs/项目全景与知识点手册.md)：11 章正文、14 张自制配图、术语表，末尾还有面试问答。完全没碰过机器学习的，从 4.5 章「给零基础读者的 ML 最小知识」开始。

## AUC 为什么只有 0.746

GitHub 上同类项目报的 AUC 常常在 0.97 以上。这个数多半来自两个在模型评审里过不了的做法。

第一个是随机切分。同一批借款人、同一段宏观环境同时进了训练集和测试集，测出来的其实是模型背过的题。月度数据让这件事更糟：一笔贷款贡献几十行，按行随机切，同一个借款人一定同时出现在两边。

第二个是特征泄漏。`total_payment`、`recoveries`、当期逾期状态，这些字段要等借款人停止还款之后才有值。把它们当特征，等于把答案提前放进了输入。

0.746 是这样算出来的：按放款年份切，只用放款当天能拿到的字段，测的是 2020–2021 那批。同样的模型如果改成随机切分，AUC 能上 0.9，但那个数对风控没有意义。

风控关心的几个问题：

| 要判断的事 | 在哪回答 |
|---|---|
| 能不能把申请人按风险排开 | AUC、Gini、KS、十分位 lift |
| 这套排序到了后面的年份还成立吗 | 时间外切分、逐 vintage AUC、PSI |
| 这个概率能拿去定价和计提吗 | 校准表（预测 vs 实际） |
| 切一条准入线会损失什么 | 通过率 vs 通过件坏账率 |
| 哪些变量在起作用，说得通吗 | WOE 分箱、IV、评分卡系数 |

## 1870 万行压成 33 万行

Freddie Mac 每个 vintage 给两个文件，结构不一样。

origination 文件一笔贷款一行，只含放款当天确定的字段（FICO、DTI、LTV、利率、州等），7 年合计 35 万行。performance 文件一笔贷款一个月一行，记还款、逾期等级、余额和结清状态，7 年合计 18,696,124 行。

建模要的是每笔贷款最后坏没坏。所以 performance 必须压掉。压不能在读完之后做：18.7M 行按 4 列 object 字符串读进 pandas 已经占 3.7 GB，35 列全读会到 20 GB 量级。压的过程放在读取的时候，分两段。

第一段在块内。每 50 万行进来，先按 `loan_id` 做一次 `groupby`，算四个局部摘要：

| 摘要 | 聚合方式 | 用途 |
|---|---|---|
| `min_age` | 全量记录取 `min` | 这笔贷款是不是从第 0 个月就被观察，喂给数据质量过滤 |
| `max_delinq` | 36 个月窗口内取 `max` | 观察期内最差逾期，用来定标签 |
| `months_observed` | 窗口内取 `max` | 让「最后一条」有东西可以排序 |
| `zb` | 窗口内先按 `months_observed` 排序，再取 `last` | 这笔贷款最后怎么了，用来定标签 |

第二段跨块合并。`min` 继续取 min，`max` 继续取 max。`zb` 得先按 `months_observed` 排序，再取 `last`。

这里容易写错。`groupby.last()` 取的是行序最后一条，不是时间上最后一条，所以排序不能省。另外 `na_position="first"` 也不是可有可无：某些块里这笔贷款没有窗口内记录，`months_observed` 是 NaN，NaN 要是排在后面，`last()` 会拿到那条空值，把真实值盖掉。这类错误不会抛异常，只会让同一份数据跑两次得到不同的标签。

`min`、`max` 跟顺序无关，怎么合都对；`last` 跟顺序有关，必须自己带着排序键。这个区别在 Spark 里对应 combiner 和 merge 函数。

上面还叠了两道过滤，都计数并打印在运行日志里，不静默丢弃：

- `sample_YYYY` 不等于 `YYYY` vintage。每个文件约 16% 的行，首付月落在 YYYY+1，不属于任何配置好的切分集合。
- 有些贷款是中途才进样本的。转让进组合的 seasoned loan，放款头几个月根本没有记录，当成「没违约」等于凭空造一条干净记录。

## 切分方式

![按 vintage 切分及样本量、违约率](docs/figures/05_oot_split.png)

训练用 2015–2018，验证用 2019，测试用 2020–2021。测试集的放款时点在完全不同的利率环境里，2020 年那一下也在里面。模型上线之后面对的就是这种情形。

违约率不是常数。2015 年放的那批是 1.08%，2019 年 5.15%，2021 年 1.43%，中间高、两头低。随机切分会把这条曲线抹平。

四个决定：

- 按放款 vintage 切，不按行切。
- 只用放款时点特征。标签从 performance 文件算，但 performance 永远不作为特征来源，这条写在 `schema.py` 的列契约里。
- WOE 分箱只在训练集上拟合，之后冻结。验证集和 OOT 走同一套映射，未来信息不会倒着漏回过去。
- 两个模型类都留着。评分卡和梯度提升树喂的是同一张 WOE 矩阵，两者差距只反映模型类别的差别，不掺特征表达的差别。参数在「方法」一节。

```mermaid
flowchart LR
    A[原始 loan-level 文件] --> B[加载器<br/>分块聚合<br/>统一 schema]
    B --> C[Vintage 切分<br/>train / valid / OOT]
    C --> D[WOE 分箱 + IV<br/>只在训练集拟合]
    D --> E1[评分卡<br/>WOE 上做 logistic]
    D --> E2[GBM challenger]
    E1 --> F[区分度<br/>AUC / Gini / KS]
    E2 --> F
    F --> G[稳定性<br/>逐 vintage AUC、PSI]
    G --> H[校准<br/>+ 准入线情景]
    H --> I[Markdown 报告<br/>+ 配图]
```

## 方法

任务定义在贷款层面，是标准的监督学习二分类问题。对第 $i$ 笔贷款估计 $PD_i = P(y_i = 1 \mid x_i)$，$y_i = 1$ 表示这笔贷款在 36 个月观察窗口内出现 90 天以上逾期，或者以信用事件类的 zero balance code 结清。

**特征变换。** 14 个放款时点变量各自切成最多 10 个分位箱。缺失值单独占一箱，占比不足 2% 的类别水平并进 `__OTHER__`。每箱换成对应的 weight of evidence，加 0.5 的拉普拉斯平滑，样本太少的箱不会给出无穷大的权重：

$$WOE_b = \ln \frac{P(x \in b \mid y = 1)}{P(x \in b \mid y = 0)}$$

变量再按信息值 IV 排序，按行业惯例读：0.02 以下无用，0.1 弱，0.3 中等，0.5 以上要当异常处理，不能当好消息：

$$IV = \sum_b \left[ P(b \mid y = 1) - P(b \mid y = 0) \right] \cdot WOE_b$$

分箱边界和 WOE 映射只在训练集上估一次，之后冻结。

**模型一，logistic 评分卡。** WOE 矩阵上做 L2 正则的 logistic 回归（L-BFGS，$C = 1.0$，迭代上限 1,000）：

$$\ln \frac{PD}{1 - PD} = \beta_0 + \sum_j \beta_j \, WOE_j$$

WOE 本身按风险单调，所以 $\exp(\beta_j)$ 就是每单位 WOE 对应的优势比倍数。这个形式的好处是信审能逐条对着系数问问题。

**模型二，梯度提升树。** 直方图梯度提升：400 轮，学习率 0.04，树深 4，L2 惩罚 1.0，内部留 15% 做早停，连续 30 轮不改善就停。这属于决策树一族，由一批浅层回归树相加而成，每一轮拟合上一轮对数损失剩下的梯度。喂进去的仍是同一张 14 列 WOE 矩阵，两个模型面对的设计矩阵完全一样。

**评估口径。** 区分度用 AUC、Gini（$2 \cdot AUC - 1$）和 KS 统计量 $\max |F_{bad} - F_{good}|$。稳定性用 PSI，作用在分数和每个 WOE 变量上，10 个分位箱，0.10 以内算稳定，0.10 到 0.25 观察，0.25 以上算漂移，另外逐 vintage 单独算一次 AUC。校准看概率十分位上预测值与实际值的对照。最后一层是决策换算，把 PD 准入线译成通过率、通过件坏账率，以及被拒部分承载的违约占比。

## 结果

真实数据跑出来是这样，训练 183,668 笔、验证 48,321 笔、时间外 99,133 笔：

| 模型 | 切分 | n | 坏账率 | AUC | Gini | KS |
|---|---|---|---|---|---|---|
| 评分卡 | train | 183,668 | 2.64% | 0.7805 | 0.5611 | 0.4187 |
| 评分卡 | valid | 48,321 | 5.15% | 0.7050 | 0.4100 | 0.3025 |
| 评分卡 | OOT | 99,133 | 2.02% | **0.7462** | 0.4924 | 0.3754 |
| GBM | train | 183,668 | 2.64% | 0.7993 | 0.5985 | 0.4514 |
| GBM | valid | 48,321 | 5.15% | 0.7040 | 0.4080 | 0.3021 |
| GBM | OOT | 99,133 | 2.02% | 0.7381 | 0.4762 | 0.3542 |

![逐 vintage 区分度](docs/figures/15_auc_by_vintage.png)

几点说明。

训练集上 GBM 比评分卡高，0.7993 对 0.7805；到 OOT 反过来了，0.7381 对 0.7462。最后留评分卡，一是这个反转，二是系数能解释。

验证集给 0.7050，比 OOT 的 0.7462 还低。只看验证集会低估这个模型，也会让人误以为需要回头重做。

评分卡的十分位坏账率从 6.84% 降到 0.33%，最高一档抓走 33.8% 的违约，lift 3.38。但预测 PD 在十档里全部低于实际值。所以它现在只能用于排序，不能定价；要算预期损失得先重新校准。

PSI 上 `orig_rate` 是 2.60、`dti` 是 0.32，两个都漂了。跨越一个利率周期本来就该这样，所以看的是逐 vintage AUC，不是合并后的单一数字。

## 没做的四件事

**拒绝推断。** 训练样本里只有被批准的贷款，模型对当初被拒的人一无所知。要修需要被拒申请记录，或者 parceling、bivariate probit 那一套。

**生存分析。** 二元的 90+ DPD 标记丢掉了违约发生的时间。Cox 模型或离散时间 hazard 模型能用上这部分信息，也能处理还没到期的 cohort。

**公平性检验。** 数据里没有受保护属性。用地域做代理只能说有迹象，下不了结论。

**利润。** AUC 和 KS 衡量的是排序，不是钱。要判断放贷赚不赚，得有 note rate、LGD 和提前还款假设。

## 数据来源

全部跑在真实、公开、有硬标签的 loan-level 数据上。

| 数据源 | 覆盖 | 获取 | 违约标签 | 为什么用它 |
|---|---|---|---|---|
| Freddie Mac Single-Family Loan-Level | 1999 至当前季度，约 5500 万笔 | 免费注册，限非商业用途 | 90+ 天逾期，或信用事件的 zero balance code | 放款风险因子齐全，另带逐月表现。主线。 |
| SBA 7(a) / 504 FOIA | FY1991 至当前季度 | 免费，免注册，纯 CSV | `loanstatus == 'CHGOFF'` | 对照用：有真实资金流，但没有征信分、DTI、抵押物。能看出区分度里有多少来自数据缺失。 |
| 合成冒烟数据 | 内置 | 无需 | 生成 | 让人几秒钟内跑通整条 pipeline。报告里明确标为示意。 |

2018 年及更早的数据集在教程里最常见，LendingClub、Home Credit、UCI 台湾信用卡都是，这里没用。用 2018 年之前的信贷周期拟合的模型，对今天的组合说明不了什么。Freddie Mac 和 SBA 还在持续追加新 vintage，稳定性分析才有意义。

## 快速开始

```bash
git clone https://github.com/prui0474/credit-default-risk.git
cd credit-default-risk

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python scripts/run_pipeline.py
```

默认数据源是内置的合成数据，不需要下载任何东西。跑完会在 `reports/` 下生成配图和 markdown 报告，终端里打一行指标摘要。这些数字只是示意，跟上面那张结果表不是一回事。

```bash
python tests/test_pipeline.py   # 10/10，不需要 pytest
```

## 接入真实数据

方案 A，Freddie Mac，推荐。在 Clarity Data Intelligence 免费注册，下载入口在 <https://claritydownload.fmapps.freddiemac.com/CRT/>。登录后打开 SFLLD Data 标签页，不是 CRT Data，那是另一个数据集。下载 sample 文件（每个 vintage 5 万笔，笔记本扛得住），解压到 `data/raw/freddie/`，然后：

```bash
python scripts/run_pipeline.py --source freddie
```

方案 B，SBA，免注册。从 <https://data.sba.gov/dataset/7-a-504-foia> 下载 7(a)/504 FOIA 的 CSV，放进 `data/raw/sba/`，在 `config.json` 里列出来，然后：

```bash
python scripts/run_pipeline.py --source sba
```

带截图的逐步操作在 [`docs/guide-zh.md`](docs/guide-zh.md)，从注册到解压，每一步该看到什么都写了。

### 跑之前要知道的四件事

官方公布的字段布局已经过时。`file_layout.xlsx` 写的是每个文件 32 个字段，实际下到的样本文件是 31（origination）和 35（performance）。origination 少了公布的字段 25，performance 后面多了三列。模型用的字段在两种布局里位置一样，所以加载器按位置取列，不按列名，另外用正则校验 join key。出版方再改格式的话，程序会直接报错，不会算出错的标签还继续往下跑。

`sample_YYYY` 不等于 `YYYY` vintage。每个文件约 16% 的行，首付月落在 YYYY+1。vintage 按首付日期定义，落在配置范围外的行被丢掉，并且计数写进运行日志。

有些贷款是中途进样本的。转让进组合的 seasoned loan，放款头几个月没有记录，当成没违约等于凭空造一条干净记录。这类也丢掉并计数，看 `[data]` 行的 `dropped(...)`。

2020–2021 的冲击会污染周围的 vintage。36 个月窗口碰到 2020–2021 的 vintage 都会沾上疫情那波逾期。2017–2019 的违约率比 2015–2016 和 2021 高出好几倍，原因在这里，时间外切分也因此是真的难。

## 目录结构

```
credit-default-risk/
├── README.md                # 英文正本
├── README.zh-CN.md          # 中文版，本文件
├── config.json              # 数据源、切分 cohort、模型参数
├── src/cdr/
│   ├── schema.py            # 每个加载器都必须满足的列契约
│   ├── config.py            # 配置加载与路径解析
│   ├── data/                # 加载器：freddie（分块）、sba、demo（合成）
│   ├── features.py          # WOE 分箱、IV、冻结映射编码器
│   ├── split.py             # 按 vintage 的 OOT 切分，另留一个随机切分做对照
│   ├── models.py            # 评分卡 + 梯度提升
│   ├── evaluate.py          # AUC/Gini/KS、PSI、十分位 lift、校准、准入线
│   ├── report.py            # 配图 + markdown 报告
│   └── pipeline.py          # load -> split -> WOE -> model -> evaluate -> report
├── scripts/
│   ├── run_pipeline.py      # 命令行入口
│   └── build_manual_figures.py  # 重新生成手册里所有配图
├── tests/test_pipeline.py   # 10 项冒烟测试，不需要 pytest
├── reports/                 # 生成的报告与配图，已 gitignore
└── docs/
    ├── index.html           # 静态 demo，评分卡在浏览器里重写
    ├── zh.html              # 同一个页面的中文版
    ├── demo/scorecard.json  # 为这个页面导出的分箱边界、WOE 映射和系数
    ├── 项目全景与知识点手册.md  # 完整讲解手册，中文，含配图
    ├── guide-zh.md          # 取数据的分步指南
    ├── figures/             # 16 张图：pipeline、泄漏、WOE、ROC、vintage 衰减
    └── refs/                # 官方字段布局定义
```

## 测试

```bash
python tests/test_pipeline.py
```

覆盖 schema 完整性、各切分之间 vintage 不重叠、WOE 对未见类别与缺失值的处理、指标合理性（完美排序给 AUC 1.0，PSI 能识别分布漂移）、合成数据上的端到端流程，以及 5 项加载器测试。加载器测试钉在真实文件格式上，包括布局漂移的回归测试（公布 32/32 列，实际到货 31/35）、字段过少的文件必须被拒绝而不是被错解，还有两道数据质量过滤。

## 参考文献

- Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk
  Scorecards* (2nd ed.). Wiley.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2017). *Credit Scoring and Its Applications* (2nd
  ed.). SIAM.
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art
  classification algorithms for credit scoring: An update of research. *European Journal of
  Operational Research*, 247(1), 124–136.
- Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring:
  a review. *Journal of the Royal Statistical Society: Series A*, 160(3), 523–541.

## 许可与数据归属

代码用 MIT，见 `LICENSE`。

数据不在仓库内分发。Freddie Mac 的 Single-Family Loan-Level Dataset 对非商业、学术与研究用途免费，商业再分发需要另行授权。引用格式：Freddie Mac, "Single Family Loan-Level Dataset"，注明获取日期与 vintage 区间。SBA 7(a)/504 FOIA 数据属美国政府公有领域。
