#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat Official Account Text Mining Analysis Tool  v2.0
微信公众号文章文本挖掘分析工具
- 统一配色/字体/布局
- 词频长尾、文章长度、关键词气泡、年度新词等14张可视化图表
- 支持任意公众号数据，替换 data/ 目录下的txt文件即可运行
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import re
import json
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import networkx as nx

# ── 字体配置 ──────────────────────────────────────────────────────────
def find_chinese_font():
    candidates = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simkai.ttf',
        'C:/Windows/Fonts/STZHONGS.TTF',
    ]
    for f in candidates:
        if os.path.exists(f):
            return f
    return None

FONT_PATH = find_chinese_font()
if FONT_PATH:
    prop = fm.FontProperties(fname=FONT_PATH)
    FONT_NAME = prop.get_name()
    plt.rcParams['font.family'] = FONT_NAME
    plt.rcParams['font.sans-serif'] = [FONT_NAME, 'SimHei', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.dpi'] = 160
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'

# 全局字体属性（所有图复用）
FP_TITLE = fm.FontProperties(fname=FONT_PATH, size=15, weight='bold') if FONT_PATH else None
FP_SUBTITLE = fm.FontProperties(fname=FONT_PATH, size=11) if FONT_PATH else None
FP_AXIS = fm.FontProperties(fname=FONT_PATH, size=10) if FONT_PATH else None
FP_TICK = fm.FontProperties(fname=FONT_PATH, size=9) if FONT_PATH else None
FP_SMALL = fm.FontProperties(fname=FONT_PATH, size=8) if FONT_PATH else None
FP_ANNO = fm.FontProperties(fname=FONT_PATH, size=9, weight='bold') if FONT_PATH else None

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 统一调色板（全部图使用同一配色体系）──────────────────────────────
# 主色：柔和但有辨识度，避免饱和度过高
C_PRIMARY   = '#3B7DD8'   # 主蓝
C_ACCENT    = '#E15759'   # 强调红
C_TEAL      = '#4EABA7'   # 青
C_AMBER     = '#F0A03C'   # 琥珀
C_PURPLE    = '#7B68AE'   # 紫
C_SLATE     = '#5C6B77'   # 石板灰
C_STEEL     = '#4A6FA5'   # 钢蓝
C_CORAL     = '#E8846B'   # 珊瑚

PALETTE = [C_PRIMARY, C_ACCENT, C_TEAL, C_AMBER, C_PURPLE, C_SLATE, C_STEEL, C_CORAL]
PALETTE_SOFT = ['#C5D9F0', '#F5C4C4', '#BFE0DE', '#FCDFB0', '#D5CCE8', '#C3CBD2', '#B8CDDE', '#F5D5CB']

# 统一背景色
BG_FIG   = '#FFFFFF'
BG_AX    = '#FAFBFD'
BG_CARD  = '#F4F6F9'

CMAP_MAIN = LinearSegmentedColormap.from_list('main', ['#EEF3FA', C_PRIMARY], N=256)
CMAP_HEAT = LinearSegmentedColormap.from_list('heat', ['#FFF8F0', '#FC8D59', '#D73027'], N=256)
CMAP_BLUE = LinearSegmentedColormap.from_list('blue', ['#EBF0F7', '#2166AC'], N=256)

# ── 停用词 ────────────────────────────────────────────────────────────
STOPWORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 那 跟 着
但 与 于 及 以 为 之 其 而 或 又 所 由 这 此 她 他 它 们 个 些 里 来 过 把
对 后 再 还 却 能 从 已 以 向 当 用 多 如 将 那么 什么 这样 因为 所以 但是
如果 虽然 然后 这个 这些 那些 通过 进行 可以 需要 包括 主要 相关 不同
提供 实现 建立 发展 开展 推进 促进 加强 等等 内容 方面 情况 工作 项目
服务 组织 社会 中国 机构 活动 行业 参与 公益 评估 一是 二是 三是 四是
同时 其中 其次 此外 另外 以及 并且 或者 无论 无论如何 对于 关于 根据
目前 已经 可能 应该 应当 需要 必须 通过 实际 具体 相比 相较 整体 总体
因此 从而 进而 由此 因此 这一 这种 该 各 第 年 月 日 第一 第二 第三 第四
总结 分析 研究 数据 报告 来说 方面 程度 基础 基于 层面 部分 全面 力量
了解 合作 推动 发挥 形成 构建 建设 体系 问题 解决 不仅 也是 还是 只是
丨 ｜ 【 】 「 」 、 ° 》 《 ▲ ▼ ? ? ? 转载
益生信 北京益生信管理咨询 阿信 阿信说 阿信策话 YESLIN Yesiin yesiin yeslin
""".split())

# ── 品牌词清洗（在分词和文本分析前统一替换）─────────────────────────
BRAND_WORDS = [
    '益生信',
    '北京益生信管理咨询有限责任公司',
    '北京益生信管理咨询',
    'YESLIN',
    'Yesiin',
    'yesiin',
    'yeslin',
    '阿信说',
    '阿信策话',
    '阿信',
]

def clean_brand(text):
    """移除文本中的品牌相关词汇"""
    for bw in BRAND_WORDS:
        text = text.replace(bw, '')
    return text


# ── 类别识别 ──────────────────────────────────────────────────────────
CATEGORY_RULES = {
    '政策动态': ['政府采购','政购','财政','民政','法规','政策','通知','条例','意见','标准'],
    '评估方法': ['评估','评价','方法','框架','指标','模型','逻辑','工具','指南'],
    '行业洞察': ['行业','洞察','趋势','发展','分析','观察','报告','扫描'],
    'ESG/可持续': ['ESG','可持续','碳','环境','绿色','责任','双碳'],
    '公益项目': ['公益','慈善','项目','社区','助残','扶贫','志愿','留守'],
    '数据与技术': ['数据','技术','数字','可视化','系统','平台','互联网','软件'],
    '健康医疗': ['健康','医疗','卫生','医务','康复','疾病','防治','养老'],
    '机构动态': ['招聘','招募','实习','招标','报名','活动预告','发布','周年'],
    '转载': ['「转」','转载'],
}

def classify_article(title, content=''):
    text = title + ' ' + content[:200]
    for cat, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in text:
                return cat
    return '其他'

# ══════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════════
def load_articles(txt_dir):
    records = []
    for fname in sorted(os.listdir(txt_dir)):
        if not fname.endswith('.txt'):
            continue
        m = re.match(r'^(\d{4}-\d{2}-\d{2})_(.+)\.txt$', fname)
        if not m:
            continue
        date_str, title = m.group(1), m.group(2)
        title = clean_brand(title)
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            continue
        fpath = os.path.join(txt_dir, fname)
        try:
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            content = ''
        content = clean_brand(content)
        content_clean = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：、""''（）《》【】]', ' ', content)
        word_count = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
        category = classify_article(title, content)
        records.append({
            'filename': fname,
            'date': date,
            'year': date.year,
            'month': date.month,
            'year_month': date.strftime('%Y-%m'),
            'title': title,
            'content': content,
            'content_clean': content_clean,
            'word_count': word_count,
            'category': category,
            'is_repost': '「转」' in title or '转载' in title,
        })
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    print("Loading complete: %d articles" % len(df))
    print("Date range: %s ~ %s" % (df['date'].min().date(), df['date'].max().date()))
    return df

# ══════════════════════════════════════════════════════════════════════
# 2. 分词与词频统计
# ══════════════════════════════════════════════════════════════════════
# 用户可在此添加自定义词汇（如品牌名、专有名词等），脚本会自动从推文标题中
# 提取最高频的2-4字词作为品牌词，无需手动配置
CUSTOM_WORDS = [
    '政购','政府购买','公益评估','评估方法','逻辑模型',
    '影响力投资','社会组织','第三方评估','健康中国','社会企业','公益项目',
    '监测评估','项目管理','专业评估','慈展会','99公益日','ESG','公益行业',
    '公益慈善','项目评估','助残服务','留守儿童','乡村医生','数字化',
    '可持续发展','社会价值','公益传播','志愿服务',
]
for w in CUSTOM_WORDS:
    jieba.add_word(w)

def tokenize(text):
    words = jieba.cut(text)
    return [w for w in words 
            if len(w) >= 2 
            and w not in STOPWORDS 
            and not re.match(r'^[\d\s\W]+$', w)
            and not w.strip() == '']

def get_all_tokens(df):
    all_tokens = []
    per_article = []
    for _, row in df.iterrows():
        tokens = tokenize(row['content_clean'])
        per_article.append(tokens)
        all_tokens.extend(tokens)
    return all_tokens, per_article

# ══════════════════════════════════════════════════════════════════════
# 3. 通用绘图辅助
# ══════════════════════════════════════════════════════════════════════
def fig_save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor=BG_FIG, dpi=160, pad_inches=0.3)
    plt.close(fig)
    print(f"  → 保存: {name}")
    return path

def style_ax(ax, title='', xlabel='', ylabel=''):
    """统一轴样式"""
    ax.set_facecolor(BG_AX)
    ax.grid(axis='y', alpha=0.25, color='#C0C8D0', linewidth=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D0D5DD')
    ax.spines['bottom'].set_color('#D0D5DD')
    if title:
        ax.set_title(title, fontproperties=FP_SUBTITLE, fontweight='bold', pad=12, color='#2C3E50')
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    ax.tick_params(axis='both', labelsize=9, colors='#5A6570', length=3)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(FP_TICK)
    return ax

def set_fig_title(fig, text, y=0.97):
    fig.text(0.5, y, text, ha='center', va='top',
             fontproperties=FP_TITLE, color='#1A2332')

def set_fig_subtitle(fig, text, y=0.935):
    fig.text(0.5, y, text, ha='center', va='top',
             fontproperties=FP_SMALL, color='#8899AA')


# ══════════════════════════════════════════════════════════════════════
# 4. 可视化函数
# ══════════════════════════════════════════════════════════════════════

# ── 图1: 总体概况仪表盘 ───────────────────────────────────────────────
def plot_overview(df):
    fig = plt.figure(figsize=(17, 10))
    fig.patch.set_facecolor(BG_FIG)
    set_fig_title(fig, '公众号文章总体概况')
    set_fig_subtitle(fig, f'数据区间：{df["date"].min().strftime("%Y年%m月")} — {df["date"].max().strftime("%Y年%m月")}  |  共 {len(df)} 篇文章')

    gs = gridspec.GridSpec(2, 4, figure=fig, top=0.89, bottom=0.07,
                           left=0.05, right=0.97, hspace=0.50, wspace=0.30)

    # ① 每年发文量
    ax1 = fig.add_subplot(gs[0, 0:2])
    year_cnt = df.groupby('year').size().reset_index(name='count')
    colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(year_cnt))]
    bars = ax1.bar(year_cnt['year'].astype(str), year_cnt['count'],
                   color=colors_bar, edgecolor='white', linewidth=0.8, zorder=3, width=0.65)
    for bar, val in zip(bars, year_cnt['count']):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                 str(val), ha='center', va='bottom', fontsize=10, color='#3A4550',
                 fontweight='bold', fontproperties=FP_TICK)
    style_ax(ax1, title='各年度发文量', xlabel='年份', ylabel='篇数')
    ax1.tick_params(axis='x', rotation=0)

    # ② 文章类别分布（水平条形，避免饼图标签重叠）
    ax2 = fig.add_subplot(gs[0, 2:4])
    cat_cnt = df['category'].value_counts()
    y_pos = range(len(cat_cnt))
    bars2 = ax2.barh(y_pos, cat_cnt.values, color=PALETTE[:len(cat_cnt)],
                     edgecolor='white', linewidth=0.5, height=0.65, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cat_cnt.index, fontproperties=FP_TICK)
    ax2.invert_yaxis()
    for bar, val in zip(bars2, cat_cnt.values):
        pct = val / len(df) * 100
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f'{val}篇 ({pct:.0f}%)', va='center', fontsize=9, color='#5A6570',
                 fontproperties=FP_TICK)
    style_ax(ax2, title='内容类别分布', xlabel='篇数')
    ax2.set_xlim(0, cat_cnt.values[0] * 1.3)
    ax2.grid(axis='x', alpha=0.2)

    # ③ 月度发文热图（给更大空间）
    ax3 = fig.add_subplot(gs[1, 0:3])
    monthly = df.groupby(['year', 'month']).size().reset_index(name='count')
    pivot = monthly.pivot(index='year', columns='month', values='count').fillna(0)
    im = ax3.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(['1月','2月','3月','4月','5月','6月',
                          '7月','8月','9月','10月','11月','12月'], fontproperties=FP_TICK)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index.astype(str), fontproperties=FP_TICK)
    vmax = pivot.values.max()
    for i in range(len(pivot.index)):
        for j in range(12):
            v = int(pivot.values[i, j])
            if v > 0:
                ax3.text(j, i, str(v), ha='center', va='center', fontsize=8,
                         color='white' if v >= vmax*0.5 else '#4A3020',
                         fontweight='bold', fontproperties=FP_SMALL)
    cbar = plt.colorbar(im, ax=ax3, orientation='vertical', pad=0.02, shrink=0.9)
    cbar.set_label('篇数', fontproperties=FP_TICK, color='#5A6570')
    cbar.ax.tick_params(labelsize=8, colors='#5A6570')
    ax3.set_title('月度发文热图（各年份 × 月份）', fontproperties=FP_SUBTITLE,
                  fontweight='bold', pad=12, color='#2C3E50')

    # ④ 文章字数分布
    ax4 = fig.add_subplot(gs[1, 3])
    wc = df['word_count'].clip(upper=4000)
    ax4.hist(wc, bins=20, color=C_PRIMARY, edgecolor='white', alpha=0.85, zorder=3)
    median_wc = wc.median()
    mean_wc = wc.mean()
    ax4.axvline(median_wc, color=C_ACCENT, linewidth=2, linestyle='--',
                label=f'中位数 {int(median_wc)}字')
    ax4.axvline(mean_wc, color=C_AMBER, linewidth=2, linestyle='-.',
                label=f'均值 {int(mean_wc)}字')
    style_ax(ax4, title='文章字数分布', xlabel='字数', ylabel='篇数')
    ax4.legend(prop=FP_SMALL, fontsize=8, framealpha=0.9, loc='upper right')

    return fig_save(fig, '01_overview.png')


# ── 图2: 发文时间线（带注释）────────────────────────────────────────────
def plot_timeline(df):
    fig, axes = plt.subplots(2, 1, figsize=(17, 10),
                              gridspec_kw={'height_ratios': [3, 2]})
    fig.patch.set_facecolor(BG_FIG)

    monthly = df.groupby('year_month').agg(
        count=('title', 'size'),
        avg_words=('word_count', 'mean')
    ).reset_index()
    monthly['date'] = pd.to_datetime(monthly['year_month'] + '-01')
    monthly = monthly.sort_values('date')

    # 上图：发文量时间线
    ax = axes[0]
    ax.set_facecolor(BG_AX)
    ax.fill_between(monthly['date'], monthly['count'], alpha=0.15, color=C_PRIMARY)
    ax.plot(monthly['date'], monthly['count'], color=C_PRIMARY, linewidth=2, zorder=3)
    ax.scatter(monthly['date'], monthly['count'], color=C_ACCENT, s=28, zorder=4, alpha=0.8)

    for yr in df['year'].unique():
        sub = monthly[monthly['date'].dt.year == yr]
        if len(sub) == 0: continue
        peak = sub.loc[sub['count'].idxmax()]
        if peak['count'] >= 6:
            ax.annotate(f'{int(peak["count"])}篇',
                        xy=(peak['date'], peak['count']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, color=C_ACCENT,
                        fontweight='bold', fontproperties=FP_ANNO)

    ax.set_title('公众号月度发文量趋势',
                 fontproperties=FP_SUBTITLE, fontweight='bold', pad=12, color='#2C3E50')
    ax.set_ylabel('月度发文篇数', fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    ax.grid(axis='y', alpha=0.25, color='#C0C8D0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D0D5DD')
    ax.spines['bottom'].set_color('#D0D5DD')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(FP_TICK)
        t.set_fontsize(9)
        t.set_color('#5A6570')

    years = sorted(df['year'].unique())
    colors_bg = [BG_CARD, '#FFFFFF']
    for i, yr in enumerate(years):
        yr_start = pd.Timestamp(f'{yr}-01-01')
        yr_end = pd.Timestamp(f'{yr}-12-31')
        ax.axvspan(yr_start, yr_end, alpha=0.35, color=colors_bg[i % 2], zorder=0)
        mid = pd.Timestamp(f'{yr}-07-01')
        ax.text(mid, ax.get_ylim()[1]*0.95, str(yr), ha='center', fontsize=9,
                color='#AAB5C0', fontproperties=FP_TICK)

    # 下图：各类别堆叠面积
    ax2 = axes[1]
    ax2.set_facecolor(BG_AX)
    cat_monthly = df.groupby(['year_month', 'category']).size().unstack(fill_value=0)
    cat_monthly.index = pd.to_datetime(cat_monthly.index + '-01')
    cat_monthly = cat_monthly.sort_index()
    cats_sorted = df['category'].value_counts().index[:6].tolist()
    cats_plot = [c for c in cats_sorted if c in cat_monthly.columns]

    ax2.stackplot(cat_monthly.index,
                  [cat_monthly[c].values for c in cats_plot],
                  labels=cats_plot,
                  colors=PALETTE[:len(cats_plot)], alpha=0.80)
    ax2.set_title('各类别文章月度分布（堆叠）', fontproperties=FP_SUBTITLE,
                  fontweight='bold', pad=10, color='#2C3E50')
    ax2.set_ylabel('篇数', fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    legend = ax2.legend(loc='upper right', ncol=3, fontsize=8,
                         prop=FP_SMALL, framealpha=0.95, edgecolor='#D0D5DD')
    ax2.grid(axis='y', alpha=0.25, color='#C0C8D0')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#D0D5DD')
    ax2.spines['bottom'].set_color('#D0D5DD')
    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_fontproperties(FP_TICK)
        t.set_fontsize(9)
        t.set_color('#5A6570')

    plt.tight_layout()
    return fig_save(fig, '02_timeline.png')


# ── 图3: 词频词云（统一浅色背景）──────────────────────────────────────
def plot_wordcloud(all_tokens):
    freq = Counter(all_tokens)

    if not FONT_PATH:
        print("  警告: 未找到中文字体，跳过词云")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG_FIG)

    # 整体词云 — 浅色背景，统一风格
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800, height=550,
        background_color='#F8FAFC',
        colormap='viridis',
        max_words=120,
        min_font_size=12,
        max_font_size=90,
        prefer_horizontal=0.7,
        collocations=False,
        contour_width=1,
        contour_color='#D0D5DD',
    ).generate_from_frequencies(dict(freq.most_common(150)))

    axes[0].imshow(wc, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('全部文章高频词云', fontproperties=FP_SUBTITLE, fontweight='bold',
                       pad=14, color='#2C3E50')

    # Top30条形图 — 浅色背景
    top30 = freq.most_common(30)
    words, counts = zip(*top30)

    # 渐变色：从深到浅
    colors_bar = [matplotlib.colors.to_rgba(C_PRIMARY, alpha=0.4 + 0.6 * (1 - i/30))
                  for i in range(30)]

    axes[1].set_facecolor(BG_AX)
    axes[1].barh(range(len(words)), counts, color=colors_bar, edgecolor='none', height=0.65, zorder=3)
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words, fontsize=10, color='#3A4550', fontproperties=FP_TICK)
    axes[1].set_xlabel('词频（次）', fontsize=10, color='#5A6570', fontproperties=FP_TICK, labelpad=6)
    axes[1].set_title('Top 30 高频词', fontproperties=FP_SUBTITLE, fontweight='bold',
                       pad=14, color='#2C3E50')
    axes[1].tick_params(axis='both', labelsize=9, colors='#5A6570', length=3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_color('#D0D5DD')
    axes[1].spines['bottom'].set_color('#D0D5DD')
    axes[1].grid(axis='x', alpha=0.15, color='#C0C8D0')
    axes[1].invert_yaxis()
    for i, (w, c) in enumerate(zip(words, counts)):
        axes[1].text(c + max(counts)*0.01, i, str(c), va='center',
                     fontsize=8, color='#7A8590', fontproperties=FP_SMALL)
    for t in axes[1].get_xticklabels():
        t.set_fontproperties(FP_TICK)
        t.set_color('#5A6570')

    plt.tight_layout()
    return fig_save(fig, '03_wordcloud.png')


# ── 图4: 各类别词云矩阵 ───────────────────────────────────────────────
def plot_category_wordclouds(df):
    if not FONT_PATH:
        return None

    main_cats = df['category'].value_counts().head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('各主题类别关键词云', fontproperties=FP_TITLE,
                 y=0.97, color='#1A2332')
    fig.text(0.5, 0.935, '每个类别独立展示其标志性高频词，边框颜色对应调色板',
             ha='center', va='top', fontproperties=FP_SMALL, color='#8899AA')

    cmaps = ['Blues', 'OrRd', 'Greens', 'Purples', 'YlOrBr', 'PuBu']
    fp = FP_TICK

    for idx, (cat, ax) in enumerate(zip(main_cats, axes.flat)):
        sub = df[df['category'] == cat]
        text = ' '.join(sub['content_clean'].tolist())
        tokens = tokenize(text)
        freq = Counter(tokens)
        if len(freq) < 5:
            ax.axis('off')
            continue
        wc = WordCloud(
            font_path=FONT_PATH,
            width=520, height=380,
            background_color='#FAFBFD',
            colormap=cmaps[idx],
            max_words=80,
            min_font_size=10,
            collocations=False,
        ).generate_from_frequencies(dict(freq.most_common(100)))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{cat}（{len(sub)}篇）', fontsize=11, fontweight='bold',
                     pad=8, fontproperties=FP_TICK, color='#2C3E50')
        rect = plt.Rectangle((0,0), 1, 1, fill=False, edgecolor=PALETTE[idx],
                              linewidth=2.5, transform=ax.transAxes)
        ax.add_patch(rect)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig_save(fig, '04_category_wordclouds.png')


# ── 图5: TF-IDF关键词热图 ────────────────────────────────────────────
def plot_tfidf_heatmap(df):
    year_texts = df.groupby('year')['content_clean'].apply(lambda x: ' '.join(x)).reset_index()
    year_texts = year_texts[year_texts['year'] >= 2018]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, max_features=200, min_df=1,
        token_pattern=None
    )
    tfidf_matrix = vectorizer.fit_transform(year_texts['content_clean'])
    feature_names = vectorizer.get_feature_names_out()

    top_words_per_year = {}
    for i, yr in enumerate(year_texts['year']):
        row = tfidf_matrix[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:15]
        top_words_per_year[yr] = [(feature_names[j], row[j]) for j in top_idx]

    all_top = set()
    for words in top_words_per_year.values():
        all_top.update([w for w,_ in words])
    all_top = list(all_top)[:35]

    years = sorted(year_texts['year'].tolist())
    matrix = pd.DataFrame(index=all_top, columns=years, dtype=float).fillna(0)
    for i, yr in enumerate(years):
        row = tfidf_matrix[i].toarray().flatten()
        for j, word in enumerate(feature_names):
            if word in all_top:
                matrix.loc[word, yr] = row[j]

    matrix['total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('total', ascending=False).head(20).drop(columns='total')

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(BG_FIG)

    sns.heatmap(matrix, ax=ax, cmap=CMAP_HEAT, linewidths=0.4, linecolor='white',
                annot=False, cbar_kws={'label': 'TF-IDF 分值', 'shrink': 0.85})

    ax.set_title('各年度核心关键词 TF-IDF 热图',
                 fontproperties=FP_SUBTITLE, fontweight='bold', pad=14, color='#2C3E50')
    ax.set_xlabel('年份', fontproperties=FP_TICK, color='#5A6570', labelpad=8)
    ax.set_ylabel('关键词', fontproperties=FP_TICK, color='#5A6570', labelpad=8)

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(FP_TICK)
        t.set_fontsize(10)
        t.set_color('#3A4550')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    return fig_save(fig, '05_tfidf_heatmap.png')


# ── 图6: 主题演变（基于NMF）────────────────────────────────────────────
def plot_topic_evolution(df, per_article_tokens):
    corpus = [' '.join(tokens) for tokens in per_article_tokens if len(tokens) >= 5]
    valid_idx = [i for i, tokens in enumerate(per_article_tokens) if len(tokens) >= 5]

    vectorizer = TfidfVectorizer(max_features=300, min_df=2, tokenizer=tokenize, token_pattern=None)
    X = vectorizer.fit_transform(corpus)

    n_topics = 6
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()

    topic_labels = []
    for i in range(n_topics):
        top_idx = H[i].argsort()[::-1][:4]
        top_words = [feature_names[j] for j in top_idx]
        topic_labels.append(f'T{i+1}: ' + '/'.join(top_words[:3]))

    df_valid = df.iloc[valid_idx].copy().reset_index(drop=True)
    for i in range(n_topics):
        df_valid[f'topic_{i}'] = W[:, i]
    df_valid['dominant_topic'] = W.argmax(axis=1)

    year_topic = df_valid.groupby('year')[[f'topic_{i}' for i in range(n_topics)]].mean()
    year_topic.columns = topic_labels

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG_FIG)

    # 左：堆叠面积
    ax1 = axes[0]
    ax1.set_facecolor(BG_AX)
    x = year_topic.index.astype(str)
    bottoms = np.zeros(len(x))
    for i, col in enumerate(year_topic.columns):
        vals = year_topic[col].values
        ax1.bar(x, vals, bottom=bottoms, label=col, color=PALETTE[i % len(PALETTE)],
                edgecolor='white', linewidth=0.5)
        bottoms += vals

    style_ax(ax1, title='各年度主题分布（NMF）', xlabel='年份', ylabel='主题权重比例')
    ax1.tick_params(axis='x', rotation=45)
    leg = ax1.legend(loc='upper left', fontsize=7, prop=FP_SMALL, bbox_to_anchor=(1.01, 1),
                      framealpha=0.95, edgecolor='#D0D5DD')

    # 右：每个主题Top词条形
    ax2 = axes[1]
    ax2.set_facecolor(BG_AX)
    ax2.axis('off')

    y_pos = 0.96
    for i in range(n_topics):
        top_idx = H[i].argsort()[::-1][:8]
        top_words_scores = [(feature_names[j], H[i][j]) for j in top_idx]
        words_str = '  '.join([f'{w}({s:.2f})' for w, s in top_words_scores])
        ax2.text(0.02, y_pos, f'【{topic_labels[i]}】',
                 fontsize=10, fontweight='bold', color=PALETTE[i % len(PALETTE)],
                 transform=ax2.transAxes, fontproperties=FP_TICK)
        ax2.text(0.02, y_pos - 0.045, words_str,
                 fontsize=8.5, color='#5A6570',
                 transform=ax2.transAxes, fontproperties=FP_SMALL)
        y_pos -= 0.14

    ax2.set_title('各主题关键词', fontproperties=FP_SUBTITLE, fontweight='bold', pad=14, color='#2C3E50')

    plt.tight_layout()
    return fig_save(fig, '06_topic_evolution.png')


# ── 图7: 关键词共现网络（统一浅色背景）──────────────────────────────────
def plot_keyword_network(all_tokens, per_article_tokens):
    freq = Counter(all_tokens)
    top_words = {w for w, c in freq.most_common(50)}

    co_occur = defaultdict(int)
    for tokens in per_article_tokens:
        article_words = set(tokens) & top_words
        article_list = list(article_words)
        for i in range(len(article_list)):
            for j in range(i+1, len(article_list)):
                pair = tuple(sorted([article_list[i], article_list[j]]))
                co_occur[pair] += 1

    G = nx.Graph()
    for word in top_words:
        if freq[word] >= 5:
            G.add_node(word, count=freq[word])

    for (w1, w2), cnt in co_occur.items():
        if cnt >= 8 and w1 in G and w2 in G:
            G.add_edge(w1, w2, weight=cnt)

    if len(G.nodes) == 0:
        return None
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_FIG)

    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=100)

    node_sizes = [G.nodes[n].get('count', 10) * 8 for n in G.nodes]
    node_colors = [G.degree(n) for n in G.nodes]

    edge_weights = [G[u][v]['weight'] for u,v in G.edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [w/max_w * 2.5 + 0.3 for w in edge_weights]
    edge_alphas = [w/max_w * 0.5 + 0.08 for w in edge_weights]

    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        x_vals = [pos[u][0], pos[v][0]]
        y_vals = [pos[u][1], pos[v][1]]
        ax.plot(x_vals, y_vals, color=C_PRIMARY, alpha=alpha, linewidth=width, zorder=1)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                            node_size=node_sizes,
                            node_color=node_colors,
                            cmap=CMAP_MAIN,
                            alpha=0.85,
                            edgecolors='white', linewidths=1)

    nx.draw_networkx_labels(G, pos, ax=ax,
                             font_size=9, font_color='#2C3E50',
                             font_family=FONT_NAME if FONT_PATH else 'sans-serif',
                             font_weight='bold')

    ax.set_title('关键词共现网络图\n（节点大小 = 词频，连线粗细 = 共现强度）',
                 fontproperties=FP_SUBTITLE, fontweight='bold', pad=16, color='#2C3E50')
    ax.axis('off')

    plt.tight_layout()
    return fig_save(fig, '07_keyword_network.png')


# ── 图8: 类别-年份交叉分析 ─────────────────────────────────────────────
def plot_category_year(df):
    cross = pd.crosstab(df['year'], df['category'])
    top_cats = df['category'].value_counts().head(7).index.tolist()
    cross = cross[[c for c in top_cats if c in cross.columns]]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG_FIG)

    # 左：堆叠条形
    ax1 = axes[0]
    cross.plot(kind='bar', stacked=True, ax=ax1, color=PALETTE[:len(cross.columns)],
               edgecolor='white', linewidth=0.5)
    style_ax(ax1, title='各年度不同类别文章数量', xlabel='年份', ylabel='篇数')
    ax1.tick_params(axis='x', rotation=45)
    leg1 = ax1.legend(loc='upper right', prop=FP_SMALL, fontsize=8,
                       framealpha=0.95, edgecolor='#D0D5DD')

    # 右：归一化热图
    ax2 = axes[1]
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    im = ax2.imshow(cross_pct.values, cmap=CMAP_BLUE, aspect='auto', vmin=0, vmax=60)
    ax2.set_xticks(range(len(cross.columns)))
    ax2.set_xticklabels(cross.columns, fontsize=9, rotation=25, ha='right', fontproperties=FP_TICK)
    ax2.set_yticks(range(len(cross.index)))
    ax2.set_yticklabels(cross.index.astype(str), fontsize=9, fontproperties=FP_TICK)

    for i in range(len(cross.index)):
        for j in range(len(cross.columns)):
            v = cross_pct.values[i, j]
            if v > 0:
                ax2.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=8,
                         color='white' if v > 30 else '#3A4550', fontweight='bold')

    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_color('#3A4550')

    cbar = plt.colorbar(im, ax=ax2, label='占比 %', shrink=0.9)
    cbar.set_label('占比 %', fontproperties=FP_TICK, color='#5A6570')
    cbar.ax.tick_params(labelsize=8, colors='#5A6570')
    ax2.set_title('各年度文章类别占比热图', fontproperties=FP_SUBTITLE,
                  fontweight='bold', pad=12, color='#2C3E50')

    plt.tight_layout()
    return fig_save(fig, '08_category_year.png')


# ── 图9: 关键话题词演变趋势 ──────────────────────────────────────────
def plot_topic_trend(df):
    topic_groups = {
        '政府购买/政购': ['政府购买', '政购', '采购', '财政', '招标'],
        '项目评估': ['评估', '第三方评估', '项目评估', '评价'],
        '社会组织': ['社会组织', '基金会', '社工', '志愿', '公益组织'],
        '数据/技术': ['数据', '技术', '数字化', '可视化', '互联网', '平台'],
        '影响力/ESG': ['影响力', 'ESG', '可持续', '社会价值', '社会影响'],
        '健康中国': ['健康', '医疗', '卫生', '助残', '养老'],
    }

    yearly_topic = {}
    for topic, keywords in topic_groups.items():
        yearly = {}
        for yr, grp in df.groupby('year'):
            text_all = ' '.join(grp['content'].tolist() + grp['title'].tolist())
            count = sum(text_all.count(kw) for kw in keywords)
            yearly[yr] = count / (len(grp) + 1e-6)
        yearly_topic[topic] = yearly

    topic_df = pd.DataFrame(yearly_topic).fillna(0)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('核心话题年度热度演变趋势', fontproperties=FP_TITLE,
                 y=0.97, color='#1A2332')
    fig.text(0.5, 0.935, '纵轴：每篇文章中该话题词组的平均出现次数',
             ha='center', va='top', fontproperties=FP_SMALL, color='#8899AA')

    for idx, (topic, ax) in enumerate(zip(topic_groups.keys(), axes.flat)):
        col = PALETTE[idx % len(PALETTE)]
        vals = topic_df[topic].values
        years = topic_df.index.astype(str)

        ax.set_facecolor(BG_AX)
        ax.fill_between(range(len(years)), vals, alpha=0.15, color=col)
        ax.plot(range(len(years)), vals, color=col, linewidth=2.5, marker='o',
                markersize=6, zorder=3, markerfacecolor='white', markeredgecolor=col, markeredgewidth=2)

        peak_i = np.argmax(vals)
        ax.annotate(f'{vals[peak_i]:.1f}',
                    xy=(peak_i, vals[peak_i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color=col, fontweight='bold',
                    fontproperties=FP_ANNO)

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, fontsize=8, fontproperties=FP_TICK, color='#5A6570')
        ax.set_title(topic, fontsize=11, fontweight='bold', color=col,
                     pad=8, fontproperties=FP_TICK)
        ax.set_ylabel('平均提及次数', fontsize=8, fontproperties=FP_SMALL, color='#7A8590')
        ax.grid(axis='y', alpha=0.2, color='#C0C8D0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#D0D5DD')
        ax.spines['bottom'].set_color('#D0D5DD')
        ax.tick_params(axis='y', labelsize=8, colors='#5A6570')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig_save(fig, '09_topic_trends.png')


# ── 图10: 综合统计摘要 ───────────────────────────────────────────────
def plot_stats_summary(df, all_tokens):
    fig = plt.figure(figsize=(17, 8))
    fig.patch.set_facecolor(BG_FIG)
    set_fig_title(fig, '公众号文章统计摘要')

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.87, bottom=0.10, left=0.05, right=0.97)

    # KPI 卡片
    stats = [
        ('总文章数', f'{len(df)} 篇', C_PRIMARY),
        ('活跃年份', f'{df["year"].nunique()} 年', C_ACCENT),
        ('原创文章', f'{(~df["is_repost"]).sum()} 篇', C_TEAL),
        ('词汇量', f'{len(set(all_tokens))} 个', C_AMBER),
    ]

    for i, (label, val, color) in enumerate(stats):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color + '10')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.62, val, ha='center', va='center', fontsize=24,
                fontweight='bold', color=color, fontproperties=FP_SUBTITLE)
        ax.text(0.5, 0.22, label, ha='center', va='center', fontsize=11,
                color='#7A8590', fontproperties=FP_TICK)
        for spine in ax.spines.values():
            spine.set_edgecolor(color + '40')
            spine.set_linewidth(2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # 各年词汇丰富度
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.set_facecolor(BG_AX)
    year_vocab = {}
    for yr, grp in df.groupby('year'):
        texts = ' '.join(grp['content_clean'].tolist())
        tokens = tokenize(texts)
        year_vocab[yr] = len(set(tokens))
    yrs = list(year_vocab.keys())
    vocabs = list(year_vocab.values())
    colors_v = [PALETTE[i % len(PALETTE)] for i in range(len(yrs))]
    ax5.bar([str(y) for y in yrs], vocabs, color=colors_v, edgecolor='white', width=0.6, zorder=3)
    style_ax(ax5, title='各年词汇丰富度（独特词汇量）', ylabel='独特词数')
    ax5.tick_params(axis='x', rotation=45)

    # 平均字数趋势
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.set_facecolor(BG_AX)
    year_wc = df.groupby('year')['word_count'].agg(['mean','median']).reset_index()
    x = [str(y) for y in year_wc['year']]
    ax6.plot(x, year_wc['mean'], marker='o', color=C_PRIMARY, linewidth=2,
             label='平均字数', markerfacecolor='white', markeredgecolor=C_PRIMARY, markeredgewidth=2)
    ax6.plot(x, year_wc['median'], marker='s', color=C_ACCENT, linewidth=2,
             linestyle='--', label='中位字数', markerfacecolor='white', markeredgecolor=C_ACCENT, markeredgewidth=2)
    ax6.fill_between(x, year_wc['mean'], year_wc['median'], alpha=0.08, color=C_PRIMARY)
    style_ax(ax6, title='各年度平均文章字数趋势', ylabel='字数')
    ax6.legend(prop=FP_SMALL, fontsize=9, framealpha=0.95, edgecolor='#D0D5DD')
    ax6.tick_params(axis='x', rotation=45)

    return fig_save(fig, '10_stats_summary.png')


# ══════════════════════════════════════════════════════════════════════
# 5. 新增分析图
# ══════════════════════════════════════════════════════════════════════

# ── 图11: 词频长尾分布（Zipf曲线）─────────────────────────────────────
def plot_word_frequency_distribution(all_tokens):
    freq = Counter(all_tokens)
    ranks = range(1, len(freq)+1)
    counts = sorted(freq.values(), reverse=True)

    # 对数坐标画 Zipf
    log_ranks = np.log10(list(ranks[:200]))
    log_counts = np.log10(counts[:200])

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('词频分布分析', fontproperties=FP_TITLE, y=0.97, color='#1A2332')
    fig.text(0.5, 0.935, '检验是否符合Zipf定律（少数高频词 + 大量低频词的长尾特征）',
             ha='center', va='top', fontproperties=FP_SMALL, color='#8899AA')

    # 左：双对数坐标下的 Zipf 曲线
    ax1 = axes[0]
    ax1.set_facecolor(BG_AX)
    ax1.scatter(log_ranks, log_counts, s=12, alpha=0.5, color=C_PRIMARY, zorder=3)
    # 理论Zipf线（最小二乘拟合）
    coeffs = np.polyfit(log_ranks, log_counts, 1)
    fit_line = np.polyval(coeffs, log_ranks)
    ax1.plot(log_ranks, fit_line, color=C_ACCENT, linewidth=2, linestyle='--',
             label=f'拟合斜率: {coeffs[0]:.2f}', zorder=4)
    ax1.set_xlabel('log10(词序)', fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    ax1.set_ylabel('log10(词频)', fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    ax1.set_title('Zipf 曲线（双对数坐标）', fontproperties=FP_SUBTITLE,
                  fontweight='bold', pad=12, color='#2C3E50')
    ax1.legend(prop=FP_SMALL, fontsize=9, framealpha=0.95, edgecolor='#D0D5DD')
    ax1.grid(alpha=0.2, color='#C0C8D0')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#D0D5DD')
    ax1.spines['bottom'].set_color('#D0D5DD')
    for t in ax1.get_xticklabels() + ax1.get_yticklabels():
        t.set_fontproperties(FP_TICK)
        t.set_color('#5A6570')

    # 右：词频分段分布（累积占比）
    ax2 = axes[1]
    ax2.set_facecolor(BG_AX)
    total = sum(counts)
    cumsum = np.cumsum(counts)
    cum_pct = cumsum / total * 100

    # 标注关键节点
    n_50 = np.searchsorted(cum_pct, 50) + 1
    n_80 = np.searchsorted(cum_pct, 80) + 1

    ax2.fill_between(range(min(500, len(counts))), cum_pct[:min(500, len(counts))],
                     alpha=0.2, color=C_PRIMARY)
    ax2.plot(range(min(500, len(counts))), cum_pct[:min(500, len(counts))],
             color=C_PRIMARY, linewidth=2, zorder=3)

    ax2.axhline(50, color=C_ACCENT, linewidth=1, linestyle='--', alpha=0.7)
    ax2.axhline(80, color=C_AMBER, linewidth=1, linestyle='--', alpha=0.7)
    ax2.axvline(n_50, color=C_ACCENT, linewidth=1, linestyle=':', alpha=0.5)
    ax2.axvline(n_80, color=C_AMBER, linewidth=1, linestyle=':', alpha=0.5)

    ax2.annotate(f'前{n_50}词覆盖50%词频', xy=(n_50, 50),
                 xytext=(n_50+50, 40), fontsize=9, color=C_ACCENT,
                 arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=1.2),
                 fontproperties=FP_SMALL, fontweight='bold')
    ax2.annotate(f'前{n_80}词覆盖80%词频', xy=(n_80, 80),
                 xytext=(n_80+50, 70), fontsize=9, color=C_AMBER,
                 arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=1.2),
                 fontproperties=FP_SMALL, fontweight='bold')

    ax2.set_xlabel('词序（按词频降序）', fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    ax2.set_ylabel('累积词频占比 %', fontproperties=FP_TICK, color='#5A6570', labelpad=6)
    ax2.set_title('累积词频分布（长尾效应）', fontproperties=FP_SUBTITLE,
                  fontweight='bold', pad=12, color='#2C3E50')
    ax2.grid(alpha=0.2, color='#C0C8D0')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#D0D5DD')
    ax2.spines['bottom'].set_color('#D0D5DD')
    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_fontproperties(FP_TICK)
        t.set_color('#5A6570')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig_save(fig, '11_word_freq_distribution.png')


# ── 图12: 文章长度分段统计 ────────────────────────────────────────────
def plot_article_length_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('文章长度分析', fontproperties=FP_TITLE, y=0.97, color='#1A2332')
    fig.text(0.5, 0.935, '文章字数（中文字）的分段分布与年度演变',
             ha='center', va='top', fontproperties=FP_SMALL, color='#8899AA')

    # 左：分段柱状图
    ax1 = axes[0]
    ax1.set_facecolor(BG_AX)
    bins_labels = ['<500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000-5000', '>5000']
    bin_edges = [0, 500, 1000, 1500, 2000, 3000, 5000, float('inf')]
    bin_counts = pd.cut(df['word_count'], bins=bin_edges, labels=bins_labels).value_counts().reindex(bins_labels)

    colors_len = [C_PRIMARY, C_STEEL, C_TEAL, C_AMBER, C_CORAL, C_ACCENT, C_PURPLE]
    bars = ax1.bar(range(len(bins_labels)), bin_counts.values, color=colors_len,
                   edgecolor='white', linewidth=0.8, width=0.65, zorder=3)
    ax1.set_xticks(range(len(bins_labels)))
    ax1.set_xticklabels(bins_labels, fontproperties=FP_TICK, rotation=30, ha='right', color='#5A6570')

    for bar, val in zip(bars, bin_counts.values):
        if val > 0:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                     str(val), ha='center', va='bottom', fontsize=9, color='#3A4550',
                     fontweight='bold', fontproperties=FP_SMALL)

    style_ax(ax1, title='文章字数分段分布', xlabel='字数区间', ylabel='篇数')
    ax1.grid(axis='x', alpha=0)

    # 右：各年度文章长度箱线图
    ax2 = axes[1]
    ax2.set_facecolor(BG_AX)
    years_sorted = sorted(df['year'].unique())
    data_by_year = [df[df['year']==y]['word_count'].clip(upper=6000).values for y in years_sorted]

    bp = ax2.boxplot(data_by_year, labels=[str(y) for y in years_sorted],
                     patch_artist=True, widths=0.6,
                     medianprops=dict(color=C_ACCENT, linewidth=2),
                     whiskerprops=dict(color='#AAB5C0', linewidth=1),
                     capprops=dict(color='#AAB5C0', linewidth=1),
                     flierprops=dict(marker='o', markerfacecolor=C_ACCENT, markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], PALETTE[:len(years_sorted)]):
        patch.set_facecolor(color + '40')
        patch.set_edgecolor(color)

    style_ax(ax2, title='各年度文章字数箱线图', xlabel='年份', ylabel='字数')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig_save(fig, '12_article_length.png')


# ── 图13: 关键词气泡图（词频×度中心性）───────────────────────────────
def plot_keyword_bubble(all_tokens, per_article_tokens):
    freq = Counter(all_tokens)
    top_words = {w for w, c in freq.most_common(80)}

    co_occur = defaultdict(int)
    for tokens in per_article_tokens:
        article_words = set(tokens) & top_words
        article_list = list(article_words)
        for i in range(len(article_list)):
            for j in range(i+1, len(article_list)):
                pair = tuple(sorted([article_list[i], article_list[j]]))
                co_occur[pair] += 1

    # 计算每个词的度中心性（共现伙伴数）
    degree = defaultdict(int)
    for (w1, w2), cnt in co_occur.items():
        if cnt >= 5:
            degree[w1] += 1
            degree[w2] += 1

    # 选Top 35高频词
    top35 = freq.most_common(35)
    words = [w for w, c in top35]
    word_freqs = [c for w, c in top35]
    word_degrees = [degree.get(w, 0) for w in words]

    fig, ax = plt.subplots(figsize=(17, 9))
    fig.patch.set_facecolor(BG_FIG)
    set_fig_title(fig, '关键词二维气泡图')
    set_fig_subtitle(fig, '横轴 = 词频，纵轴 = 共现伙伴数，气泡大小 = 频次占比')

    ax.set_facecolor(BG_AX)

    # 气泡颜色：按度中心性映射
    max_deg = max(word_degrees) if word_degrees else 1
    colors_bubble = [matplotlib.colors.to_rgba(PALETTE[i % len(PALETTE)],
                        alpha=0.4 + 0.5 * (d / max_deg))
                     for i, d in enumerate(word_degrees)]

    sizes = [f / max(word_freqs) * 2000 + 80 for f in word_freqs]

    ax.scatter(word_freqs, word_degrees, s=sizes, c=colors_bubble,
               edgecolors='white', linewidths=1.2, zorder=3, alpha=0.85)

    # 标注文字
    for i, w in enumerate(words):
        if i < 20 or word_freqs[i] > 200:
            ax.annotate(w, (word_freqs[i], word_degrees[i]),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=9, color='#2C3E50', fontproperties=FP_SMALL,
                        fontweight='bold')

    style_ax(ax, title='', xlabel='词频（出现次数）', ylabel='共现伙伴数')
    ax.grid(alpha=0.2, color='#C0C8D0')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig_save(fig, '13_keyword_bubble.png')


# ── 图14: 年度新词发现 ────────────────────────────────────────────────
def plot_yearly_new_words(df):
    """每年首次出现、且后续再未出现的独特词（年度特色词）"""
    year_word_sets = {}
    for yr, grp in df.groupby('year'):
        text = ' '.join(grp['content_clean'].tolist())
        tokens = set(tokenize(text))
        # 只取长度>=2且有一定频率的词
        all_toks = tokenize(text)
        freq = Counter(all_toks)
        important_words = {w for w, c in freq.items() if c >= 2 and len(w) >= 2}
        year_word_sets[yr] = important_words

    years = sorted(year_word_sets.keys())

    # 找每年的"独特词"：该年有，但之前年份没有的词
    yearly_unique = {}
    seen = set()
    for yr in years:
        new_words = year_word_sets[yr] - seen
        yearly_unique[yr] = new_words
        seen.update(year_word_sets[yr])

    fig, axes = plt.subplots(2, 5, figsize=(17, 9))
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle('年度特色词汇发现', fontproperties=FP_TITLE, y=0.97, color='#1A2332')
    fig.text(0.5, 0.935, '每年首次出现的高频词（至少出现2次），反映各年的独特关注点',
             ha='center', va='top', fontproperties=FP_SMALL, color='#8899AA')

    for idx, (yr, ax) in enumerate(zip(years, axes.flat)):
        words = yearly_unique.get(yr, set())
        if not words:
            ax.axis('off')
            ax.set_title(f'{yr}', fontproperties=FP_TICK, color='#2C3E50', fontsize=11, fontweight='bold')
            continue

        # 取该年全文本中的词频
        grp = df[df['year'] == yr]
        text = ' '.join(grp['content_clean'].tolist())
        freq = Counter(tokenize(text))

        # 筛选独特词并按频次排序
        word_freqs = [(w, freq.get(w, 0)) for w in words if freq.get(w, 0) >= 2]
        word_freqs.sort(key=lambda x: x[1], reverse=True)
        word_freqs = word_freqs[:12]  # 取前12

        if not word_freqs:
            ax.axis('off')
            ax.set_title(f'{yr}', fontproperties=FP_TICK, color='#2C3E50', fontsize=11, fontweight='bold')
            continue

        ws, fs = zip(*word_freqs)
        color = PALETTE[idx % len(PALETTE)]

        ax.set_facecolor(BG_AX)
        ax.barh(range(len(ws)), fs, color=color + '80', edgecolor=color,
                linewidth=0.8, height=0.6, zorder=3)
        ax.set_yticks(range(len(ws)))
        ax.set_yticklabels(ws, fontsize=8, fontproperties=FP_SMALL, color='#3A4550')
        ax.invert_yaxis()

        ax.set_title(f'{yr}（{len(words)}个新词）', fontproperties=FP_TICK,
                     color=color, fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#D0D5DD')
        ax.spines['bottom'].set_color('#D0D5DD')
        ax.grid(axis='x', alpha=0.15, color='#C0C8D0')
        ax.tick_params(axis='both', labelsize=7, colors='#7A8590')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig_save(fig, '14_yearly_new_words.png')


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 数据目录：默认读取项目根目录下的 data/ 子目录
    # 用户只需将 txt 推文文件放入 data/ 目录即可
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    TXT_DIR = os.path.join(PROJECT_ROOT, 'data')
    # 兼容旧目录名"推文"
    if not os.path.isdir(TXT_DIR) and os.path.isdir(os.path.join(PROJECT_ROOT, '推文')):
        TXT_DIR = os.path.join(PROJECT_ROOT, '推文')

    print("=" * 55)
    print("  微信公众号文章文本挖掘分析 v2.0")
    print("=" * 55)
    print(f"  数据目录: {TXT_DIR}")

    print("\n[1/6] 加载文章数据...")
    df = load_articles(TXT_DIR)

    print("\n[2/6] 分词与词频统计...")
    all_tokens, per_article_tokens = get_all_tokens(df)
    freq = Counter(all_tokens)
    print(f"  → 总词数: {len(all_tokens):,}  独特词: {len(freq):,}")
    print(f"  → Top10词: {[w for w,_ in freq.most_common(10)]}")

    # 保存数据摘要
    summary = {
        'total_articles': len(df),
        'date_range': [str(df['date'].min().date()), str(df['date'].max().date())],
        'years': sorted(df['year'].unique().tolist()),
        'category_counts': df['category'].value_counts().to_dict(),
        'total_tokens': len(all_tokens),
        'unique_tokens': len(freq),
        'avg_word_count': float(df['word_count'].mean()),
        'top50_words': freq.most_common(50),
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n[3/6] 生成基础可视化图表（1-10）...")
    paths = {}

    print("  生成图1: 总体概况...")
    paths['overview'] = plot_overview(df)

    print("  生成图2: 发文时间线...")
    paths['timeline'] = plot_timeline(df)

    print("  生成图3: 词频词云...")
    paths['wordcloud'] = plot_wordcloud(all_tokens)

    print("  生成图4: 类别词云矩阵...")
    paths['cat_wc'] = plot_category_wordclouds(df)

    print("  生成图5: TF-IDF热图...")
    paths['tfidf'] = plot_tfidf_heatmap(df)

    print("  生成图6: 主题演变...")
    paths['topics'] = plot_topic_evolution(df, per_article_tokens)

    print("  生成图7: 关键词共现网络...")
    paths['network'] = plot_keyword_network(all_tokens, per_article_tokens)

    print("  生成图8: 类别-年份分析...")
    paths['cat_year'] = plot_category_year(df)

    print("  生成图9: 话题趋势...")
    paths['trends'] = plot_topic_trend(df)

    print("  生成图10: 统计摘要...")
    paths['stats'] = plot_stats_summary(df, all_tokens)

    print("\n[4/6] 生成新增分析图表（11-14）...")
    print("  生成图11: 词频分布分析...")
    paths['freq_dist'] = plot_word_frequency_distribution(all_tokens)

    print("  生成图12: 文章长度分析...")
    paths['article_len'] = plot_article_length_analysis(df)

    print("  生成图13: 关键词气泡图...")
    paths['bubble'] = plot_keyword_bubble(all_tokens, per_article_tokens)

    print("  生成图14: 年度特色词汇...")
    paths['new_words'] = plot_yearly_new_words(df)

    print("\n[5/6] 保存元数据...")
    meta_path = os.path.join(OUTPUT_DIR, 'article_meta.csv')
    df[['filename','date','year','month','title','word_count','category','is_repost']].to_csv(
        meta_path, index=False, encoding='utf-8-sig')
    print(f"  → {meta_path}")

    print("\n[6/6] 完成！")
    print(f"  → 图表已保存至: {OUTPUT_DIR}")
    print(f"  → 共生成 {len([p for p in paths.values() if p])} 张图")

    # 输出路径供HTML生成使用
    with open(os.path.join(OUTPUT_DIR, 'paths.json'), 'w', encoding='utf-8') as f:
        json.dump({k: os.path.basename(v) if v else None for k,v in paths.items()},
                  f, ensure_ascii=False, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # 7. 生成 HTML 报告（含数据分析解读）
    # ══════════════════════════════════════════════════════════════════════
    print("\n[7/7] 生成 HTML 报告...")

    def fig_to_base64(fig_path):
        import base64
        if not fig_path or not os.path.exists(fig_path):
            return ''
        with open(fig_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f'data:image/png;base64,{data}'

    # ── 计算分析解读所需的数据 ──────────────────────────────────────────
    total = len(df)
    years = sorted(df['year'].unique())
    n_years = len(years)
    date_min = df['date'].min().strftime('%Y.%m')
    date_max = df['date'].max().strftime('%Y.%m')
    n_original = int((~df['is_repost']).sum())
    n_repost = int(df['is_repost'].sum())
    orig_pct = n_original / total * 100
    repost_pct = n_repost / total * 100
    avg_wc = int(df['word_count'].mean())
    median_wc = int(df['word_count'].median())
    max_wc = int(df['word_count'].max())
    min_wc = int(df['word_count'].min())
    total_chars = int(df['word_count'].sum())

    cat_counts = df['category'].value_counts()
    top_cat = cat_counts.index[0]
    top_cat_cnt = cat_counts.iloc[0]
    top_cat_pct = top_cat_cnt / total * 100
    n_cats = len(cat_counts)

    # 年度发文量
    yearly_counts = df.groupby('year').size()
    peak_year = yearly_counts.idxmax()
    peak_count = yearly_counts.max()
    low_year = yearly_counts.idxmin()
    low_count = yearly_counts.min()

    # 月度发文
    monthly_counts = df.groupby('month').size()
    peak_month = monthly_counts.idxmax()
    peak_month_cnt = monthly_counts.max()

    # 词频分析
    top10 = freq.most_common(10)
    top10_str = '、'.join([w for w, c in top10])
    top1_word, top1_count = top10[0]
    total_tokens = len(all_tokens)
    top1_pct = top1_count / total_tokens * 100

    # HHI 指数（词汇集中度）
    total_freq_sum = sum(freq.values())
    hhi = sum((c / total_freq_sum) ** 2 for _, c in freq.items())

    # 词汇丰富度
    ttr = len(freq) / total_tokens if total_tokens > 0 else 0

    # 年度趋势关键词（近3年 vs 前3年）
    mid = len(years) // 2
    early_years = years[:mid] if mid > 0 else years[:1]
    late_years = years[mid:] if mid > 0 else years
    early_freq = Counter()
    late_freq = Counter()
    for _, row in df.iterrows():
        f_bucket = early_freq if row['year'] in early_years else late_freq
        tokens = tokenize(row['content_clean'])
        f_bucket.update(tokens)

    rising_words = []
    for w in late_freq:
        if w in early_freq and late_freq[w] >= 5 and early_freq[w] >= 3:
            ratio = late_freq[w] / early_freq[w]
            rising_words.append((w, ratio, late_freq[w], early_freq[w]))
    rising_words.sort(key=lambda x: x[1], reverse=True)
    top_rising = rising_words[:5]

    # 类别年度趋势
    cat_year_pivot = df.pivot_table(index='category', columns='year', aggfunc='size', fill_value=0)
    growing_cats = []
    for cat in cat_year_pivot.index:
        vals = cat_year_pivot.loc[cat].values
        if len(vals) >= 2 and vals[-1] > vals[0]:
            growing_cats.append((cat, vals[-1] - vals[0], vals[-1]))
    growing_cats.sort(key=lambda x: x[1], reverse=True)

    # 文章长度分段
    short_pct = (df['word_count'] < 1000).sum() / total * 100
    mid_pct = ((df['word_count'] >= 1000) & (df['word_count'] < 3000)).sum() / total * 100
    long_pct = (df['word_count'] >= 3000).sum() / total * 100

    # 生成各章节解读文字
    def safe_join(words_list):
        return '、'.join([w for w, _, _, _ in words_list]) if words_list else '（数据不足）'

    analysis = {}

    # 1. 总体概况
    analysis['overview'] = f"""本期分析涵盖 <strong>{total}</strong> 篇文章，时间跨度从 <strong>{date_min}</strong> 至 <strong>{date_max}</strong>，横跨 <strong>{n_years}</strong> 个年份。
其中原创文章 <strong>{n_original}</strong> 篇（占 {orig_pct:.1f}%），转载文章 <strong>{n_repost}</strong> 篇（占 {repost_pct:.1f}%），
平均每篇文章约 <strong>{avg_wc}</strong> 字，中位数为 <strong>{median_wc}</strong> 字。
文章共涉及 <strong>{n_cats}</strong> 个类别，其中「<strong>{top_cat}</strong>」类文章最多，共 {top_cat_cnt} 篇（占 {top_cat_pct:.1f}%）。"""

    # 2. 发文时间线
    analysis['timeline'] = f"""发文量在 <strong>{peak_year}</strong> 年达到峰值（{peak_count} 篇），而 <strong>{low_year}</strong> 年最少（{low_count} 篇）。
从月度分布来看，<strong>{peak_month} 月</strong>是发文最密集的月份（{peak_month_cnt} 篇）。
通过时间线可以直观观察公众号的运营节奏与内容产出周期。"""

    # 3. 词频词云
    analysis['wordcloud'] = f"""全量文本共提取 <strong>{total_tokens:,}</strong> 个有效词，其中独特词 <strong>{len(freq):,}</strong> 个。
出现频率最高的词是「<strong>{top1_word}</strong>」（{top1_count} 次，占比 {top1_pct:.2f}%）。
Top 10 高频词为：{top10_str}。这些词汇反映了公众号的核心关注领域。"""

    # 4. 类别词云
    cat_names = list(cat_counts.index)
    cat_wc_desc = '各主题类别的词云展示了不同领域的关注焦点。'
    for cat in cat_names[:3]:
        cat_top = df[df['category'] == cat]
        if len(cat_top) > 0:
            cat_text = ' '.join(cat_top['content_clean'].tolist())
            cat_freq = Counter(tokenize(cat_text))
            cat_top3 = '、'.join([w for w, _ in cat_freq.most_common(3)])
            cat_wc_desc += f'「{cat}」类的高频词以 {cat_top3} 为代表；'
    analysis['cat_wc'] = cat_wc_desc.rstrip('；') + '。'

    # 5. TF-IDF
    analysis['tfidf'] = """TF-IDF 热图展示了各年度最具区分度的关键词。
颜色越深表示该词在该年度的 TF-IDF 权重越高，即该词在该年度的文章中具有更强的代表性和特异性。
通过纵向对比，可以发现不同年份的话题重心迁移；通过横向对比，可以识别长期稳定的核心术语。"""

    # 6. 主题演变
    analysis['topics'] = """基于 NMF（非负矩阵分解）的主题模型将全量文章归纳为若干隐含主题。
每个子图展示了一个主题在各年份的占比变化，反映该主题热度的升降趋势。
主题演变的分析有助于理解公众号内容焦点的历史变迁和未来走向。"""

    # 7. 共现网络
    network_top = top10[:5]
    network_words = '、'.join([w for w, _ in network_top])
    analysis['network'] = f"""关键词共现网络以 <strong>{network_words}</strong> 等高频词为核心节点，
节点越大表示词频越高，连线越粗表示两词在同一篇文章中共同出现的次数越多。
网络的中心性和聚类结构揭示了关键词之间的语义关联，帮助发现隐含的话题群落。"""

    # 8. 类别-年份
    if growing_cats:
        gc = growing_cats[0]
        gc_str = '、'.join([f'「{c[0]}」（+{c[1]}篇）' for c in growing_cats[:3]])
        analysis['cat_year'] = f"""从类别与年份的交叉分析来看，{gc_str} 呈现上升趋势。
各类别的年度分布变化反映了公众号内容策略的调整和行业关注点的迁移。"""
    else:
        analysis['cat_year'] = """从类别与年份的交叉分析来看，各类别文章的年度分布相对均衡，
反映了公众号内容策略的持续性和稳定性。"""

    # 9. 话题趋势
    analysis['trends'] = """核心话题趋势图追踪了六大话题组（由语义相近的关键词聚合而成）的年度热度变化。
通过折线的上升和下降，可以识别出：正在兴起的新话题、持续走热的核心话题、以及逐渐淡出的衰退话题。
这为内容规划和选题决策提供了数据支撑。"""

    # 10. 统计摘要
    analysis['stats'] = f"""词汇丰富度（Type-Token Ratio）为 <strong>{ttr:.3f}</strong>，
词汇集中度 HHI 指数为 <strong>{hhi:.4f}</strong>（越接近 1 表示词汇越集中于少数词）。
全量文本总字数约 <strong>{total_chars:,}</strong> 字，文章字数范围从 {min_wc} 到 {max_wc} 字，标准差较大，
说明文章篇幅差异明显，公众号可能同时包含短评/快讯和深度长文两种风格。"""

    # 11. 词频分布
    analysis['freq_dist'] = f"""词频分布检验是否符合 Zipf 定律（少数高频词占据大量出现次数）。
通常前 <strong>5%</strong> 的高频词占据了总词频的 <strong>{sum(c for _, c in freq.most_common(max(1, len(freq)//20))) / total_freq_sum * 100:.1f}%</strong>。
累积词频曲线呈现典型的长尾特征，说明大部分低频词虽然出现次数少，但承载了丰富的语义信息。"""

    # 12. 文章长度
    analysis['article_len'] = f"""文章长度分布中，短文（<1000字）占 <strong>{short_pct:.1f}%</strong>，
中等长度（1000-3000字）占 <strong>{mid_pct:.1f}%</strong>，
长文（≥3000字）占 <strong>{long_pct:.1f}%</strong>。
年度箱线图展示了各年文章长度的中位数和离散程度的变化趋势。"""

    # 13. 关键词气泡
    analysis['bubble'] = """气泡图将每个关键词映射到「词频 × 共现伙伴数」的二维空间中：
横轴表示该词出现的总次数，纵轴表示与该词共现的不同词汇数量，气泡大小也反映词频。
位于右上角的词既高频又"百搭"，是连接多个话题的桥梁词；位于右下角的词虽然高频但共现伙伴少，
可能是某一垂直领域的专有名词。"""

    # 14. 年度新词
    if top_rising:
        rising_str = safe_join(top_rising)
        analysis['new_words'] = f"""每年出现的"特色新词"反映了该年度的独特关注点。
跨年度对比发现，近年出现频率上升显著的词汇包括：{rising_str}。
这些词汇的增长轨迹可能预示着新兴话题或行业趋势。"""
    else:
        analysis['new_words'] = """每年出现的"特色新词"反映了该年度的独特关注点。
这些词在该年首次出现且后续再次使用，是追踪内容演变的重要信号。"""

    # 综合分析
    insight_text = f"""<h3>核心发现</h3>
<ul>
<li>内容定位：公众号以「{top_cat}」（{top_cat_pct:.1f}%）为核心主题，同时覆盖 {n_cats-1} 个其他类别，内容多元化程度{'较高' if n_cats >= 5 else '适中'}。</li>
<li>高频关键词：{top10_str}——这些词构成了公众号的语义核心。</li>
<li>发文节奏：{peak_year}年产出最为活跃（{peak_count}篇），运营力度最强。</li>
<li>原创比例：原创内容占 {orig_pct:.1f}%，{'以原创为主' if orig_pct > 60 else '转载与原创并存'}，体现了{'较强的内容生产' if orig_pct > 60 else '内容策展与原创混合'}能力。</li>
<li>文章深度：平均 {avg_wc} 字/篇，{'偏好深度长文' if avg_wc >= 2000 else '以中等篇幅为主'}，{'字数差异较大，内容形式多样' if max_wc - min_wc > 5000 else '篇幅较为统一'}。</li>
"""
    if top_rising:
        rising_names = '、'.join([w for w, _, _, _ in top_rising[:3]])
        insight_text += f"<li>新兴趋势：近年「{rising_names}」等词汇关注度显著上升，值得持续关注。</li>\n"
    insight_text += "</ul>"

    # ── 构建图表章节列表 ──────────────────────────────────────────────
    chapters = [
        ('01', '总体概况', analysis['overview'], paths.get('overview')),
        ('02', '发文时间线', analysis['timeline'], paths.get('timeline')),
        ('03', '词频词云', analysis['wordcloud'], paths.get('wordcloud')),
        ('04', '类别词云矩阵', analysis['cat_wc'], paths.get('cat_wc')),
        ('05', 'TF-IDF 热图', analysis['tfidf'], paths.get('tfidf')),
        ('06', '主题演变（NMF）', analysis['topics'], paths.get('topics')),
        ('07', '关键词共现网络', analysis['network'], paths.get('network')),
        ('08', '类别-年份交叉分析', analysis['cat_year'], paths.get('cat_year')),
        ('09', '核心话题趋势', analysis['trends'], paths.get('trends')),
        ('10', '统计摘要', analysis['stats'], paths.get('stats')),
        ('11', '词频分布分析', analysis['freq_dist'], paths.get('freq_dist')),
        ('12', '文章长度分析', analysis['article_len'], paths.get('article_len')),
        ('13', '关键词气泡图', analysis['bubble'], paths.get('bubble')),
        ('14', '年度特色词汇', analysis['new_words'], paths.get('new_words')),
    ]

    # 分类统计
    cat_rows = ''
    for cat, cnt in cat_counts.items():
        pct = cnt / len(df) * 100
        cat_rows += f'<tr><td>{cat}</td><td>{cnt}</td><td>{pct:.1f}%</td></tr>\n'

    # Top20 词
    top20_rows = ''
    for w, c in freq.most_common(20):
        top20_rows += f'<tr><td>{w}</td><td>{c}</td></tr>\n'

    # 图表章节 HTML
    fig_sections = ''
    for num, title, insight, fpath in chapters:
        b64 = fig_to_base64(fpath)
        if b64:
            fig_sections += f'''
            <section class="fig-section" id="fig{num}">
                <div class="fig-header">
                    <span class="fig-number">{num}</span>
                    <div>
                        <h2>{title}</h2>
                    </div>
                </div>
                <div class="fig-insight">
                    <div class="insight-icon">💡</div>
                    <div class="insight-text">{insight}</div>
                </div>
                <div class="fig-img-wrap">
                    <img src="{b64}" alt="{title}" loading="lazy" />
                </div>
            </section>
            '''

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>微信公众号文本挖掘分析报告</title>
<style>
:root {{
    --primary: #3B7DD8;
    --accent: #E15759;
    --bg: #F4F6F9;
    --card: #FFFFFF;
    --text: #1A2332;
    --text2: #5A6570;
    --border: #E2E8F0;
    --radius: 12px;
    --insight-bg: #EBF5FF;
    --insight-border: #B3D4FC;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB",
                 "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.8;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px 24px 80px; }}

/* 导航 */
.toc {{
    position: sticky; top: 0; z-index: 100;
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 10px 0; margin-bottom: 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.toc-inner {{
    display: flex; flex-wrap: wrap; gap: 6px;
    max-width: 1100px; margin: 0 auto; padding: 0 24px;
}}
.toc a {{
    font-size: 13px; color: var(--text2); text-decoration: none;
    padding: 4px 10px; border-radius: 20px;
    background: var(--bg); transition: all .2s;
}}
.toc a:hover {{ background: var(--primary); color: #fff; }}

/* 头部 */
.hero {{
    background: linear-gradient(135deg, #3B7DD8 0%, #7B68AE 100%);
    color: #fff; border-radius: var(--radius);
    padding: 40px 36px; margin-bottom: 32px;
}}
.hero h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.hero p {{ font-size: 15px; opacity: 0.85; }}
.hero .meta {{
    display: flex; flex-wrap: wrap; gap: 24px;
    margin-top: 20px; font-size: 14px;
}}
.hero .meta span {{
    background: rgba(255,255,255,0.18);
    padding: 6px 14px; border-radius: 8px;
}}

/* KPI 卡片 */
.kpi-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 32px;
}}
.kpi-card {{
    background: var(--card); border-radius: var(--radius);
    padding: 20px; border-left: 4px solid var(--primary);
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.kpi-card:nth-child(2) {{ border-left-color: var(--accent); }}
.kpi-card:nth-child(3) {{ border-left-color: #4EABA7; }}
.kpi-card:nth-child(4) {{ border-left-color: #F0A03C; }}
.kpi-card .val {{ font-size: 28px; font-weight: 700; color: var(--text); }}
.kpi-card .lbl {{ font-size: 13px; color: var(--text2); margin-top: 4px; }}

/* 双列统计 */
.stats-grid {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 20px; margin-bottom: 32px;
}}
@media (max-width: 700px) {{ .stats-grid {{ grid-template-columns: 1fr; }} }}
.stat-card {{
    background: var(--card); border-radius: var(--radius);
    padding: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.stat-card h3 {{
    font-size: 15px; color: var(--primary);
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 2px solid var(--bg);
}}
.stat-card table {{ width: 100%; border-collapse: collapse; }}
.stat-card td {{
    padding: 6px 8px; font-size: 13px; border-bottom: 1px solid var(--bg);
}}
.stat-card td:last-child {{ text-align: right; color: var(--text2); }}

/* 图表章节 */
.fig-section {{
    background: var(--card); border-radius: var(--radius);
    margin-bottom: 24px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.fig-header {{
    display: flex; align-items: flex-start; gap: 16px;
    padding: 20px 24px 12px; border-bottom: none;
}}
.fig-number {{
    background: var(--primary); color: #fff;
    font-size: 13px; font-weight: 700;
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.fig-header h2 {{ font-size: 17px; font-weight: 600; }}

/* 分析解读卡片 */
.fig-insight {{
    display: flex; gap: 12px; align-items: flex-start;
    margin: 0 24px 16px; padding: 14px 18px;
    background: var(--insight-bg);
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    font-size: 14px; line-height: 1.8; color: #2C3E50;
}}
.insight-icon {{ font-size: 16px; flex-shrink: 0; margin-top: 2px; }}
.insight-text {{ flex: 1; }}
.insight-text strong {{ color: var(--primary); }}

.fig-img-wrap {{
    padding: 16px 20px 20px; text-align: center;
    background: #FAFBFD; border-top: 1px solid var(--border);
}}
.fig-img-wrap img {{
    max-width: 100%; height: auto; border-radius: 8px;
}}

/* 综合分析 */
.insight-section {{
    background: var(--card); border-radius: var(--radius);
    padding: 28px 32px; margin-bottom: 32px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border-top: 4px solid var(--primary);
}}
.insight-section h3 {{
    font-size: 18px; font-weight: 700; color: var(--primary);
    margin-bottom: 16px;
}}
.insight-section ul {{
    padding-left: 20px;
}}
.insight-section li {{
    font-size: 14px; line-height: 1.9; color: #2C3E50;
    margin-bottom: 6px;
}}

/* 方法说明 */
.method-section {{
    background: var(--card); border-radius: var(--radius);
    padding: 24px 28px; margin-bottom: 32px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.method-section h3 {{
    font-size: 16px; font-weight: 600; color: var(--text);
    margin-bottom: 12px;
}}
.method-section p {{
    font-size: 13px; line-height: 1.8; color: var(--text2);
    margin-bottom: 8px;
}}

/* 页脚 */
.footer {{
    text-align: center; font-size: 13px; color: var(--text2);
    margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border);
}}
</style>
</head>
<body>
<div class="container">

<div class="hero">
    <h1>微信公众号文本挖掘分析报告</h1>
    <p>基于 NLP 的自动化内容分析 · 14 张专业图表 + 数据解读</p>
    <div class="meta">
        <span>📊 文章总数: {total} 篇</span>
        <span>📅 时间跨度: {date_min} — {date_max}</span>
        <span>📝 独特词汇: {len(freq):,} 个</span>
        <span>⏰ 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
    </div>
</div>

<!-- KPI 卡片 -->
<div class="kpi-row">
    <div class="kpi-card">
        <div class="val">{total}</div>
        <div class="lbl">总文章数</div>
    </div>
    <div class="kpi-card">
        <div class="val">{n_years}</div>
        <div class="lbl">活跃年份</div>
    </div>
    <div class="kpi-card">
        <div class="val">{n_original}</div>
        <div class="lbl">原创文章（{orig_pct:.0f}%）</div>
    </div>
    <div class="kpi-card">
        <div class="val">{avg_wc:,}</div>
        <div class="lbl">平均字数</div>
    </div>
</div>

<!-- 数据概览 -->
<div class="stats-grid">
    <div class="stat-card">
        <h3>📁 类别分布</h3>
        <table>{cat_rows}</table>
    </div>
    <div class="stat-card">
        <h3>🔤 高频词 Top 20</h3>
        <table>{top20_rows}</table>
    </div>
</div>

<!-- 综合分析 -->
<div class="insight-section">
    {insight_text}
</div>

<!-- 方法说明 -->
<div class="method-section">
    <h3>🔬 分析方法说明</h3>
    <p><strong>分词引擎：</strong>jieba 中文分词，已加载自定义领域词典和停用词表。</p>
    <p><strong>主题模型：</strong>TF-IDF + NMF（非负矩阵分解），自动发现隐含主题及其年度分布。</p>
    <p><strong>共现网络：</strong>基于文章级别的词共现关系构建图，节点大小表示词频，连线粗细表示共现强度。</p>
    <p><strong>话题趋势：</strong>将语义相近的关键词归为话题组，追踪各组在不同年份的相对热度变化。</p>
    <p><strong>年度新词：</strong>识别各年首次出现、且之前年份未出现的高频词（出现 ≥2 次），反映年度特色。</p>
    <p><strong>类别识别：</strong>基于标题和正文前 200 字的关键词匹配规则进行自动分类。</p>
</div>

<!-- 图表章节 -->
{fig_sections}

<div class="footer">
    Generated by <strong>wechat-text-mining</strong> · {datetime.now().year}
</div>

</div>
</body>
</html>'''

    report_path = os.path.join(SCRIPT_DIR, 'report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  → HTML 报告已保存: {report_path}")
    print(f"  → 完成！共生成 14 张图 + 1 份 HTML 报告")
