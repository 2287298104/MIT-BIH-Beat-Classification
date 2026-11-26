# 基于单导联心电的心拍级心律失常检测（中文版代码说明）

生物医学信号处理课程设计 · 2026 年秋季学期 · 华侨大学

**作者：** Feng Runhang（2315104007）　**指导教师：** 段芳 博士<br>
**仓库地址：** https://github.com/2287298104/MIT-BIH-Beat-Classification

本项目在 **MIT-BIH 心律失常数据库**上实现了一套完整、可复现的心电信号处理流程：
Pan-Tompkins R 峰检测 → 心拍切分 → 40 维手工特征 → 经典分类器；评估采用 de Chazal 等人（2004）提出的
**患者间 DS1/DS2 划分协议**，标签体系遵循 **AAMI EC57** 五大类。

**研究问题。** 一套轻量、完全可解释的经典流程（手工特征 + 树模型/核方法分类器，在一组患者上训练、
在完全不重叠的另一组患者上测试）究竟能把室性异位心拍检出并分类到什么程度？它的错误又来自哪里？

## 主要结果

| 环节 | 结果 |
|---|---|
| R 峰检测（42 条记录、97,075 个标注心拍） | 灵敏度 Se **99.26 %**，阳性预测值 PPV **99.73 %**（汇总值） |
| 心拍分类（DS2，20 位未见患者、45,791 个心拍） | 准确率 **93.04 %**，加权 F1 **0.919**，**V 类 F1 0.906**，N 类 F1 0.963 |
| 少见类别 | S 类 F1 0.143，F 类 F1 0.014 |
| 同一模型在 DS1 内做心拍级交叉验证 | macro-F1 **0.907**，而跨患者只有 **0.507** |

0.907 与 0.507 之间的差距就是本项目的核心发现：MIT-BIH 上人们报告的“难”，主要难在
**患者无关（patient-independent）的评估方式**，而不是分类器本身。

## 快速开始

```bash
pip install -r requirements.txt

python src/download_data.py     # 从 PhysioNet 下载约 90 MB（48 条记录中用 42 条）
python run_all.py               # 特征 -> 训练/评估 -> 出图
python src/build_report.py      # 重建 report/Course_Project_Report.docx
```

`run_all.py --skip-data` 会复用已存在的 `outputs/features_mitdb.csv`。

笔记本 CPU 上的耗时：数据集构建约 1 分钟，随机森林约 7 秒，Platt 校准的 RBF-SVM 约 2.5 分钟。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/config.py` | 路径、采样率、记录清单（DS1/DS2）、AAMI 映射、窗口长度 |
| `src/download_data.py` | 从 PhysioNet 下载所需记录 |
| `src/preprocessing.py` | 0.5–40 Hz 零相位带通、Pan-Tompkins 包络、两套阈值规则、峰匹配 |
| `src/segmentation.py` | AAMI 标签映射、心拍窗口、RR 间期特征 |
| `src/features.py` | 40 维形态/频谱/小波/节律描述子 |
| `src/build_dataset.py` | 逐记录处理流程 → `outputs/features_mitdb.csv` |
| `src/train_eval.py` | 训练 LR / RF / SVM，在 DS1 的验证半区上调决策规则，再在 DS2 上评估 |
| `src/make_figures.py` | 生成全部插图到 `outputs/figures/` |
| `src/build_report.py` | 依据实测结果生成 Word 报告 |
| `run_all.py` | 端到端运行入口 |
| `outputs/` | 特征表、指标、逐类报告、`summary.json`、`figures/` |
| `report/` | 课程报告（`Course_Project_Report.docx`） |

## 方法

1. **预处理** —— 巴特沃斯带通 0.5–40 Hz，正反向各滤一次（`sosfiltfilt`），保证零相位，
   不会因为相位偏移把 R 峰挪位。
2. **R 峰检测** —— Pan & Tompkins（1985）：5–15 Hz 带通、五点差分、平方、150 ms 滑动窗积分、
   阈值判决、200 ms 不应期，最后把峰点细化到 ±100 ms 内最大偏转处。项目实现了两套阈值规则：
   * `detect_r_peaks_dual_threshold` —— 论文原始的 SPKI/NPKI 自适应规则（报告中作为对照基线）；
   * `detect_r_peaks` —— 本项目采用的规则：阈值 = **局部** ±3 s 窗口内包络 90 分位幅值的 12 %，
     并加入 T 波抑制。它对心拍振幅交替出现的记录更稳健（第 228 号记录上，原始规则灵敏度掉到 29 %，
     局部规则仍有 96 %）。
3. **心拍切分** —— 每个检测到的 R 峰取一个窗口，−250 ms 到 +450 ms（360 Hz 下 252 个采样点），
   通过与最近的心内科标注做一对一匹配（容差 150 ms）赋标签。
4. **特征（40 维）** —— 13 个形态描述子、9 个归一化幅值采样点（−200 … +200 ms）、
   6 个频谱描述子（各频带功率、熵、谱重心、高频/低频比）、5 个 db4 四层小波子带能量、
   6 个节律描述子（前/后 RR、局部平均 RR、RR 比值、|ΔRR|）。
5. **分类** —— 多分类逻辑回归（线性基线）、随机森林（300 棵树）、RBF 核 SVM。全部使用
   `class_weight='balanced'`（不做人工合成心拍）；每个模型的类别乘子在 DS1 内部一个
   **患者不重叠**的验证半区上调好，再应用到 DS2。
6. **评估** —— 逐类精确率/召回率/F1、macro-F1、加权 F1、平衡准确率、Cohen's κ，
   以及室性类的二分类 F1 与其精确率-召回率曲线。

### 数据协议

* 48 条 MIT-BIH 记录中分析 42 条。102、104、107、217 号被排除（起搏节律，不在 AAMI 类别体系内）；
  113、121 号不属于本项目遵循的公开 DS1/DS2 划分。
* **DS1**（22 条记录）用于训练，**DS2**（另外 20 条、互不重叠）用于测试。任何患者都不会同时出现在两个集合中。
* 优先使用 MLII 导联，缺失时取第一个可用导联。

## 输出

| 图 | 内容 |
|---|---|
| `fig01_pipeline.png` | 处理流程框图 |
| `fig02_detection_example.png` | 检测到的 R 峰与标注对比；按类别着色的切分心拍 |
| `fig03_class_distribution.png` | DS1 / DS2 的类别分布（对数坐标） |
| `fig04_beat_templates.png` | 各 AAMI 类别的中位心拍波形 |
| `fig05_confusion_matrices.png` | 三个模型的混淆矩阵 |
| `fig06_feature_importance.png` | 随机森林最重要的 15 个特征 |
| `fig07_pr_curves.png` | 室性类的精确率-召回率曲线 |
| `fig08_detector_per_record.png` | 逐记录的检测灵敏度与 PPV |

数值结果：`outputs/model_metrics.csv`、`outputs/per_class_<model>.csv`、
`outputs/detection_stats.csv`、`outputs/summary.json`。

## 复现说明

* Python 3.13，NumPy、SciPy、PyWavelets、scikit-learn、wfdb、Matplotlib、python-docx
  （见 `requirements.txt`）。
* 所有随机过程都固定了种子（`RANDOM_STATE = 42`）：交叉验证折、随机森林、SVM。
* 超参数与决策乘子的选择全程没有接触 DS2。
* 特征提取对整条记录的心拍矩阵做了向量化，因此约 9.6 万个心拍的数据集构建只需约 1 分钟。

## 数据与许可

MIT-BIH 心律失常数据库由 PhysioNet 分发，采用 ODC Attribution 许可
（https://physionet.org/content/mitdb/1.0.0/）。本仓库不再分发原始数据，
由 `src/download_data.py` 负责下载。

## 参考文献

1. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database.
   *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.
2. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
   *IEEE TBME*, 32(3), 230-236.
3. de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of
   heartbeats using ECG morphology and heartbeat interval features. *IEEE TBME*,
   51(7), 1196-1206.
4. ANSI/AAMI EC57 (2012). Testing and reporting performance results of cardiac rhythm
   and ST segment measurement algorithms.
5. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
   *Circulation*, 101(23), e215-e220.
