# 微信公众号文本挖掘分析工具

> 一键分析微信公众号文章，自动生成 **14 张专业可视化图表 + 完整 HTML 报告**。  
> 无需编程基础，替换数据即可运行。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 这是什么？

一个 Python 脚本，读取微信公众号文章的纯文本文件，自动完成：

1. **数据加载与清洗** — 解析文件名中的日期和标题，清洗品牌词/停用词
2. **中文分词与词频统计** — 基于 jieba 分词，支持自定义词典
3. **文章主题分类** — 基于关键词规则自动归类（10 个预设类别）
4. **14 张可视化图表** — 概况、趋势、词云、TF-IDF、主题演变、共现网络等
5. **HTML 报告** — 将所有图表汇总为一份可直接浏览的交互式报告

适用于：**公众号运营复盘、内容策略分析、学术研究数据预处理**等场景。

---

## 功能总览

### 分析能力

| 模块 | 说明 |
|------|------|
| 文章分类 | 基于关键词规则自动归类：政策动态、评估方法、行业洞察、ESG/可持续、公益项目、数据与技术、健康医疗、机构动态、转载、其他 |
| 高频词分析 | jieba 分词 + 停用词过滤 + 自定义词典 |
| TF-IDF 关键词 | 各年度核心关键词的重要性矩阵 |
| 主题发现 | NMF（非负矩阵分解）自动发现隐含主题 |
| 共现网络 | 高频词之间的共现关系，基于 NetworkX |
| 话题趋势 | 自定义话题组的年度热度演变 |
| 文本统计 | 字数分布、词汇丰富度、Zipf 定律检验 |

### 生成的 14 张图表

| # | 图名 | 内容 |
|---|------|------|
| 01 | 总体概况仪表盘 | 年度发文量 + 类别分布 + 月度热图 + 字数分布 |
| 02 | 发文时间线 | 月度发文趋势 + 各类别堆叠面积图 |
| 03 | 词频词云 | 全局词云 + Top30 高频词条形图 |
| 04 | 类别词云矩阵 | 6 大主题类别各自的关键词云 |
| 05 | TF-IDF 热图 | 年度 × 关键词重要性矩阵 |
| 06 | 主题演变（NMF） | 6 个隐含主题的年度权重分布 |
| 07 | 关键词共现网络 | 词频 × 共现强度的关系网络 |
| 08 | 类别-年份交叉 | 各年度类别数量堆叠 + 占比热图 |
| 09 | 核心话题趋势 | 6 大话题组的年度热度折线 |
| 10 | 统计摘要 | KPI 卡片 + 词汇丰富度 + 字数趋势 |
| 11 | 词频分布 | Zipf 双对数曲线 + 累积词频长尾 |
| 12 | 文章长度 | 字数分段柱状图 + 年度箱线图 |
| 13 | 关键词气泡图 | 词频 × 共现伙伴数二维散点 |
| 14 | 年度特色词汇 | 每年首次出现的高频词 |

### HTML 报告

运行后自动生成 `analysis/report.html`，包含：
- 顶部 KPI 卡片（文章数、年份、原创数、平均字数）
- 类别分布与高频词 Top20 表格
- 14 张图表按序排列（图片内嵌为 Base64，可离线浏览）
- 响应式布局，支持手机/平板查看

---

## 快速开始

### 前置要求

- Python 3.8+
- Windows / macOS / Linux

### 三步运行

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/wechat-text-mining.git
cd wechat-text-mining

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行分析
python analysis/text_mining.py
```

### 准备数据

将微信公众号文章的 `.txt` 文件放入 `data/` 目录，文件名格式为：

```
YYYY-MM-DD_文章标题.txt
```

示例：
```
data/
├── 2020-01-15_关于乡村振兴的政策解读.txt
├── 2020-02-20_公益项目评估方法论.txt
├── 2020-03-10_社会组织能力建设实践.txt
└── ...
```

> txt 文件内容就是文章正文，每篇一个文件。可通过 [WeChat Article Exporter](https://github.com/pengzhendong/WeChatArticleExporter) 等工具批量导出。

### 查看结果

- **图表**：`analysis/figures/` 目录（14 张 PNG，160dpi）
- **HTML 报告**：`analysis/report.html`（双击即可在浏览器中打开）
- **数据摘要**：`analysis/figures/summary.json`（JSON 格式）
- **文章元数据**：`analysis/figures/article_meta.csv`（日期、标题、字数、分类等）

---

## 项目结构

```
wechat-text-mining/
├── data/                       # ← 放入你的 txt 文章文件
│   ├── 2020-01-15_xxx.txt
│   └── ...
├── analysis/
│   ├── text_mining.py          # 主分析脚本（全部逻辑）
│   ├── report.html             # 生成的 HTML 报告
│   └── figures/                # 生成的图表（自动创建）
│       ├── 01_overview.png
│       ├── ...
│       ├── 14_yearly_new_words.png
│       ├── summary.json
│       └── article_meta.csv
├── requirements.txt            # Python 依赖
├── LICENSE                     # MIT 协议
└── README.md
```

---

## 自定义配置

编辑 `analysis/text_mining.py` 顶部的配置区：

### 品牌词过滤

```python
BRAND_WORDS = [
    '品牌词A', '品牌词B',       # 替换为你要过滤的词
]
```

脚本会在加载时自动从标题和正文中移除这些词，并加入停用词列表。

### 文章分类规则

```python
CATEGORY_RULES = {
    '类别名': ['关键词1', '关键词2', ...],
    ...
}
```

根据你的公众号内容领域修改分类规则。默认预设了 10 个类别，覆盖公益、政策、ESG 等领域。

### 自定义分词词典

```python
CUSTOM_WORDS = [
    '专有名词A', '专有名词B',   # jieba 会将其作为整体切分
]
```

### 停用词

```python
STOPWORDS = set("""...""".split())
```

在现有停用词基础上增删。

### 图表配色

```python
PALETTE = ['#3B7DD8', '#E15759', '#4EABA7', ...]
```

8 色调色板，所有图表统一使用。

---

## 应用于其他公众号

只需两步：

1. **清空 `data/` 目录**，放入目标公众号的文章 txt 文件（文件名格式 `YYYY-MM-DD_标题.txt`）
2. **修改配置**（可选）：
   - `BRAND_WORDS` — 过滤目标公众号的品牌词
   - `CATEGORY_RULES` — 调整分类规则以匹配内容领域
   - `CUSTOM_WORDS` — 添加领域专有名词
   - `STOPWORDS` — 增减停用词

然后运行 `python analysis/text_mining.py` 即可。

---

## 技术栈

| 库 | 用途 |
|----|------|
| [jieba](https://github.com/fxsjy/jieba) | 中文分词 |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF、NMF 主题模型 |
| [matplotlib](https://matplotlib.org/) | 图表绘制 |
| [seaborn](https://seaborn.pydata.org/) | 热图等统计图表 |
| [wordcloud](https://github.com/amueller/word_cloud) | 词云生成 |
| [networkx](https://networkx.org/) | 共现网络图 |
| [numpy](https://numpy.org/) / [pandas](https://pandas.pydata.org/) | 数据处理 |

---

## 注意事项

- **中文字体**：脚本会自动检测 Windows 系统字体（微软雅黑、黑体等）。macOS/Linux 用户可能需要修改 `find_chinese_font()` 中的字体路径。
- **文章数量**：建议 ≥ 50 篇以获得有意义的分析结果。少于 10 篇时部分图表可能效果不佳。
- **编码格式**：txt 文件请使用 UTF-8 编码保存。
- **内存占用**：10,000+ 篇文章时共现网络计算可能较慢，可适当降低 `top_words` 阈值。

---

## 未来扩展方向

以下是一些可以进一步改进的方向，欢迎贡献：

### 分析能力
- [ ] **情感分析**：基于 SnowNLP 或预训练模型的文章情感极性分析
- [ ] **关键词提取优化**：引入 TextRank 算法，替代/补充 TF-IDF
- [ ] **LDA 主题模型**：与现有 NMF 并行，提供可对比的主题发现
- [ ] **时间序列预测**：基于 ARIMA/Prophet 预测未来发文趋势
- [ ] **文章聚类**：K-Means / DBSCAN 对文章自动聚类
- [ ] **相似文章检测**：基于余弦相似度的重复/相似文章发现

### 数据采集
- [ ] **微信公众号爬虫**：集成 WeChat Article Exporter 或 Selenium 方案，自动采集文章
- [ ] **多公众号对比**：支持同时分析多个公众号并生成对比报告
- [ ] **评论数据**：分析文章评论的文本特征

### 可视化
- [ ] **交互式 Dashboard**：基于 Plotly Dash / Streamlit 的交互式分析面板
- [ ] **动态时间线**：可播放的动画时间线（年度变化过程）
- [ ] **主题河流图**：Topic River 可视化主题演变
- [ ] **词频动态气泡**：Gapminder 风格的词频动态演变

### 工程
- [ ] **命令行参数**：支持 `--data-dir`、`--output-dir` 等参数
- [ ] **配置文件**：将 BRAND_WORDS、CATEGORY_RULES 等抽到 YAML/JSON 配置文件
- [ ] **进度条**：集成 tqdm 显示分词、图表生成进度
- [ ] **Docker 支持**：一键容器化运行，解决环境依赖问题

### 输出
- [ ] **PDF 报告**：基于 WeasyPrint / ReportLab 生成可打印 PDF
- [ ] **Excel 摘要**：将统计结果导出为格式化的 Excel 文件
- [ ] **PPT 自动生成**：将图表嵌入演示文稿模板

---

## License

[MIT License](LICENSE)
