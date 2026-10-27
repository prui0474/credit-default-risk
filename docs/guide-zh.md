# 中文操作指引：从零把项目跑起来

这份文档假设你**没有编程背景**，从头到尾一步步来。每一步都写清楚：在哪里操作、输入什么、
做完之后屏幕上应该出现什么。看到的东西和这里写的不一样，就停下来回看对应那一步。

> **本文只解决「怎么把它跑起来」。**
> 想弄懂「每一步为什么这么做、每个数字是怎么来的、面试怎么讲」，
> 请读 **[《项目全景与知识点手册》](项目全景与知识点手册.md)**（11 章 + 14 张配图 + 术语表 + 12 个面试问答）。
> 完全没学过机器学习？直接看手册的 **第 4.5 章「给零基础读者的 ML 最小知识」**。

---

## 0. 这个项目在做什么（30 秒版）

给一笔贷款申请，模型输出一个 0 到 1 之间的数字——**这人不还钱的概率**（PD，Probability of
Default，违约概率）。银行用这个数字决定：放不放款、给什么利率、要留多少钱做坏账准备。

项目名字里几个词的意思：

| 名词 | 白话解释 |
|---|---|
| 流水线 pipeline | 一串按顺序自动执行的步骤。原始数据进去，报告出来，中间不用人插手。 |
| 评分卡 scorecard | 传统打分表。每个特征对应的分值相加得总分。银行用了几十年，好处是每一项都能解释「为什么扣分」。 |
| 训练集 / 验证集 / 时间外样本 | 训练集是「用来学习的历史贷款」；验证集是「边学边检查的」；时间外样本（OOT，out-of-time）是「比训练数据更晚放款的那批贷款」，用来模拟真实上线后的效果。 |
| 数据泄漏 leakage | 模型偷看了「放款时根本还不知道」的信息，于是成绩虚高。比如用「已追回金额」预测「会不会违约」——已经违约了才有追回金额。 |
| WOE / IV | 一种把特征分箱并换算成「风险倍数」的做法。IV（信息值）是衡量这个特征有没有用的分数，越大越有用。 |
| AUC / Gini / KS | 三个衡量「排序能力」的指标。AUC 0.5 等于瞎猜，1.0 等于完美。Gini = 2×AUC−1。KS 看两组人分数分得多开。 |
| PSI | 衡量「人群有没有变」。小于 0.10 算稳定，0.10–0.25 要盯，超过 0.25 说明人群结构已经变了。 |
| 校准 calibration | 预测 10% 违约的那批人，实际是不是真的 10% 违约。排序好但校准差，定价就是错的。 |
| 虚拟环境 venv | 给这个项目单独准备一个「工具箱」，装的库不污染电脑上其他项目。 |

---

## 1. 先跑通示例数据（不需要下载任何东西，约 3 分钟）

项目自带一份**合成示例数据**，就是为了让你先确认环境没问题。它不是真实数据，报告里标注得很清楚。

### 第 1 步：打开命令行

按键盘 `Win` 键，输入 `powershell`，在弹出的「Windows PowerShell」上按回车。会跳出一个黑色（或蓝色）窗口。

### 第 2 步：进入项目目录

在窗口里输入下面这行，按回车：

```powershell
cd "E:\workbudddy\2026-10-05-19-39-41\credit-default-risk"
```

> 路径里有空格或中文时，两边要加双引号。本路径没有空格，加引号也不会出错。

**应该看到**：命令提示符前面变成 `...\credit-default-risk>`。

### 第 3 步：建虚拟环境

输入：

```powershell
python -m venv .venv
```

**应该看到**：什么都不显示，等几秒后回到提示符。项目文件夹里多出一个 `.venv` 文件夹（可能在资源管理器里看不到，因为被 .gitignore 隐藏了，这正常）。

如果提示 `python 不是内部或外部命令`，说明电脑没装 Python，去 <https://www.python.org/downloads/> 下载安装，安装时**务必勾选 "Add python.exe to PATH"**。

### 第 4 步：激活虚拟环境

```powershell
.venv\Scripts\activate
```

**应该看到**：提示符前面多出 `(.venv)`，像这样：

```
(.venv) PS E:\workbudddy\2026-10-05-19-39-41\credit-default-risk>
```

如果提示「禁止运行脚本」，改成先运行这一行，再重试上面那一行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 第 5 步：安装依赖库

```powershell
pip install -r requirements.txt
```

**应该看到**：一堆 `Collecting ...` 和 `Installing ...`，最后是 `Successfully installed ...`。需要几十秒到几分钟。

### 第 6 步：跑流水线

```powershell
python scripts/run_pipeline.py
```

**应该看到**（数字可能略有不同）：

```
[data] source=demo rows=6,300 vintages=2015-2021 default_rate=8.603%
[report] written to E:\...\credit-default-risk\reports\model_report.md

[metrics]
model     gbm  scorecard
split
oot    0.7899     0.8141
train  0.9096     0.8433
valid  0.7677     0.8077
  OOT scorecard  AUC=0.8141  Gini=0.6283  KS=0.4859
  OOT gbm        AUC=0.7899  Gini=0.5799  KS=0.4404
```

### 第 7 步：打开报告

```powershell
start reports\model_report.md
```

**应该看到**：浏览器或 VS Code 打开一份 markdown 报告；同时 `reports\figures\` 里有 6 张 PNG 图。

### 第 8 步：跑测试（可选，确认代码没坏）

```powershell
python tests\test_pipeline.py
```

**应该看到**：6 个 `PASS`，最后一行 `6/6 passed`。

---

## 2. 换成真实数据 A：Freddie Mac（推荐，数据最新）

Freddie Mac 是美国两大住房抵押贷款机构之一。它公开了**每一笔房贷**的放款信息和逐月还款记录，
数据一直更新到最近的季度，免费供学术和非商业用途使用。缺点是：**要先注册一个账号**才能下载。

### 第 1 步：注册账号（唯一需要你本人做的一步）

（2026-10 核对过：数据覆盖 1999-01-01 至 2026-03-31 放款的贷款，约 5,600 万笔。）

1. 打开浏览器，访问介绍页：<https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>
2. 在页面往下滚一点，找到一个大按钮 **"Access Historical Data"**（页面上有**两个**一样的，
   点哪个都行），点它。
3. 浏览器会跳到下载门户，地址变成：<https://claritydownload.fmapps.freddiemac.com/CRT/>
   ——这个新页面是个需要登录的应用，**没登录时只会显示一句 "Click Provider Link"**，这是正常的。
4. 在新页面上找 **Register** / **Sign up** / **Create account** 之类的链接，用邮箱注册
   （设置密码，收验证邮件点确认）。全程免费。
5. 用注册好的账号和密码登录。登录后看页面**最右上角**，有两个标签：`CRT Data` 和
   **`SFLLD Data`**。**点 `SFLLD Data`**——我们不要 CRT（那是另一套房贷证券化债券的披露
   文件，跟逐笔贷款数据不是一回事）。进去后找 **Sample** 分组，下载 `sample_2015` …
   `sample_2021`。

> 页面措辞可能变。找不到入口时，在这个页面上搜「Download」「Sample」「Login」「Register」
> 这几个词。注册这一步必须你本人操作，我无法代你登录。
> 卡住了发邮件问官方：<clarity@freddiemac.com>（英文）。

### 第 2 步：下载 sample（样本）文件

登录后进入数据下载页，会看到几组文件：

| 文件组 | 说明 | 建议 |
|---|---|---|
| `historical_data_YYYYQn.zip` | 全量数据，几十 GB | 先别下 |
| **`sample_YYYY.zip`** | **每年随机抽 5 万笔贷款** | **下这个** |

**下载 2015 到 2021 这 7 个年份的 `sample_YYYY.zip`**（`sample_2015.zip`、`sample_2016.zip`、
……、`sample_2021.zip`）。每个文件不大，加起来通常几百 MB。

> 为什么是 2015–2021：项目配置里训练用 2015–2018 年放款的贷款，验证用 2019，时间外测试用
> 2020–2021（正好覆盖疫情冲击）。年份不合适也可以改 `config.json` 里的
> `split.train_vintages` / `valid_vintages` / `oot_vintages`。

### 第 3 步：解压并放到指定位置

1. 每个 `sample_YYYY.zip` 解压后是一个文件夹，里面有**两个 txt 文件**：
   - `sample_orig_YYYY.txt` —— 放款时的信息（信用分、贷款价值比、收入负债比、利率……）
   - `sample_perf_YYYY.txt` —— 逐月还款记录，违约标签就是从这里面算出来的
2. 把这些解压出来的内容放到项目的 `data\raw\freddie\` 下面。目录看起来应该是这样：

```
credit-default-risk\
└── data\
    └── raw\
        └── freddie\
            ├── sample_2015\  (内含 sample_orig_2015.txt, sample_perf_2015.txt)
            ├── sample_2016\  (...)
            ├── ...
            └── sample_2021\
```

> 直接把这些 txt 文件平铺摆在 `freddie\` 里也行，程序会自动递归查找。
> 老版本的文件名可能叫 `sample_svcg_YYYY.txt` 而不是 `sample_perf_YYYY.txt`，程序两种都认。

### 第 4 步：改一行配置

用记事本或 VS Code 打开项目根目录的 `config.json`，找到：

```json
"source": "demo",
```

改成：

```json
"source": "freddie",
```

保存。（如果记事本提示编码，选 UTF-8。）

### 第 5 步：跑

```powershell
python scripts/run_pipeline.py --source freddie
```

**应该看到**：先是 `[data] source=freddie rows=350,000 vintages=2015-2021 ...`，
（行数取决于你下到几年）然后生成报告。跑一次大约一到几分钟。

### 第 6 步：看报告和改结论

打开 `reports\model_report.md`。里面每一节都可以直接当简历项目描述的事实来源：

- 第 1 节告诉你样本量和分年违约率；
- 第 3 节是模型排序能力（AUC / Gini / KS），**OOT 那一行才是真本事**；
- 第 4 节看模型有没有随时间变差（vintage 表 + 特征 PSI）；
- 第 6 节把模型接到业务决策上：卡在什么阈值，通过率多少、通过的人里有多少会违约。

### 常见报错

| 报错 | 原因和解决 |
|---|---|
| `No origination files matching sample_orig_*.txt under ...` | 文件没放对，或还没解压。检查 `data\raw\freddie\` 下确实有 `sample_orig_2015.txt` 这类文件。 |
| `Vintage split produced an empty split` | 配置里的年份和实际下载的年份对不上。要么补下缺失年份，要么改 `config.json` 的 `split` 部分。 |
| `MemoryError` / 电脑变卡 | 年份下太多了。先用 2015–2018 四年跑通，再逐年加。 |
| `ModuleNotFoundError: No module named 'pandas'` | 虚拟环境没激活（提示符前面没有 `(.venv)`），回到第 4 步。 |

---

## 3. 换成真实数据 B：SBA（完全不用注册，最快拿到手）

美国小企业管理局（SBA）按《信息自由法》公开了**每一笔担保贷款**的审批记录，包含借款人、
银行、金额、利率、期限、行业，以及**结果**（`PIF` 已还清 / `CHGOFF` 已核销）。好处是零门槛，
缺点是**没有信用分、没有负债收入比、没有抵押物估值**——所以模型能做到的区分度天然更低。
这本身就是值得写进报告的一条结论。

### 第 1 步：下载

1. 打开 <https://data.sba.gov/dataset/7-a-504-foia>
2. 下载这两类 CSV（分十年的那几个文件）：
   - 7(a) 的 **FY2020–present**（想跑更长历史就把更早的分段也下了）
   - 504 的 **FY2010–present**
3. 放到 `data\raw\sba\` 下。

### 第 2 步：在 config.json 里登记文件

```json
"sba": {
  "files": [
    "data/raw/sba/7a_504_foia_7a_fy2020_present.csv"
  ],
  "max_vintage_year": 2020
}
```

`files` 里写**相对项目根目录**的路径，一个文件一行，用逗号分隔。
`max_vintage_year` 是「最晚用到哪一年审批的贷款」——留 4 年以上的观察期，不然新放款的贷款还
没来得及出现问题，拿它跟老贷款比违约率是不公平的。

### 第 3 步：跑

```powershell
python scripts/run_pipeline.py --source sba
```

---

## 4. 常用命令速查

```powershell
python scripts/run_pipeline.py                  # 用 config.json 里的 source
python scripts/run_pipeline.py --source freddie # 临时指定数据源
python scripts/run_pipeline.py --no-report      # 只算指标，不生成报告
python tests/test_pipeline.py                   # 跑测试
```

改模型参数（都写注释了）在 `config.json` 的 `model` 段；改特征清单在 `features` 段。

---

## 5. 下一步可以往哪里加

按「对求职性价比」排序：

1. **加生存分析（Cox / 离散时间风险模型）**——用上「什么时候违约」的信息，而不是只用一个
   是非标签。风控岗面试里很少见，讲出来有区分度。
2. **加拒绝推断（reject inference）**——训练数据里只有「被批准」的贷款，模型看不到被拒的人。
   这是信贷建模公认的难点，能说清楚就已经赢了一半。
3. **加公平性分析**——房贷数据里有州、收入结构等信息，可以做代理变量的公平性检查。
4. **加 API**——用 FastAPI 把模型包成一个 `/score` 接口，配一个 Streamlit 界面，
   演示「输入申请人信息 → 输出 PD 和利率建议」。

---

## 6. 这份数据的几个坑（程序已替你处理，但你该知道）

下面四条都是用你下载的真实文件（2026 年 10 月版本）实测出来的，不是猜的。

| # | 坑 | 现象 | 程序怎么处理的 |
|---|---|---|---|
| 1 | **官方说明书过期了** | `file_layout.xlsx` 写的是每个文件 32 列，但实际文件是 **origination 31 列 / performance 35 列** | 读取时**按"第几列"取数，不按列名**。这样不管官方再加列减列都不会错位；同时校验贷款编号格式，一旦格式不对就**报错停下**，绝不硬猜 |
| 2 | **文件年份 ≠ 放款年份** | `sample_2015` 里约 16% 的贷款首次还款日在 2016 年（2015 那份里只有 84% 是 2015 放款） | 放款年份一律按**首次还款日**判定；落在 2015–2021 之外的（约 7,956 笔）明确剔除并在日志里报数 |
| 3 | **有些贷款"中途才进样本"** | 约 11,000 笔贷款的最早还款记录直接是第 44 个月（收购来的存量贷款），前 36 个月根本不在文件里 | 这类贷款**不能当作"没违约"**，一律剔除并报数。日志里会看到 `dropped(no_opening_window=…, outside_split_vintages=…)` |
| 4 | **疫情把中间几档的违约率抬高了** | 90+ 天逾期的占比在 2020–2021 报告期从 0.35% 以下窜到 2%–3.4% | 不用处理，这是真实历史。但要理解：**2017–2019 那几档违约率偏高（3%–5%）不是模型的问题，是它们的 36 个月窗口撞上了疫情**；2021 那档全程避开，所以只有 1.4% |

> 关于第 4 条，实测的三个文件对比（90+ 天逾期占该年报告记录的比例）：
>
> ```
> sample_perf_2017  2019: 0.30% → 2020: 2.32% → 2021: 3.40% → 2022: 1.66%
> sample_perf_2019  2019: 0.03% → 2020: 1.67% → 2021: 2.82% → 2022: 1.29%
> sample_perf_2021  2022: 0.17% → 2023: 0.34% → 2024: 0.35% → 2025: 0.43%   ← 无峰值
> ```
