import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
import os

# --- 1. 字体路径动态获取 ---
# 确保你已经在仓库中创建了 fonts 文件夹，并上传了 simhei.ttf (文件名建议全小写)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(CURRENT_DIR, "fonts", "simhei.ttf")

# 创建字体对象
if os.path.exists(FONT_PATH):
    my_font = fm.FontProperties(fname=FONT_PATH)
else:
    my_font = None
    print(f"⚠️ 警告：路径下未找到字体文件 {FONT_PATH}，中文将显示为乱码！")

# --- 2. 基础风格配置 ---
# 注意：不要在 set_theme 中设置 font 参数，否则在云端会报错
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['axes.unicode_minus'] = False # 解决负号乱码

# --- 3. 辅助函数：统一设置轴字体 ---
def set_ax_font(ax, title, xlabel, ylabel):
    """强制将所有文本元素设为中文字体，解决乱码和英文问题"""
    ax.set_title(title, fontproperties=my_font, fontweight='bold', fontsize=16)
    ax.set_xlabel(xlabel, fontproperties=my_font)
    ax.set_ylabel(ylabel, fontproperties=my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    for tick in ax.get_yticklabels(): tick.set_fontproperties(my_font)
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), fontproperties=my_font)

# --- 4. 绘图函数定义 ---

def plot_mianyang_ranking(df_metrics, col, title):
    fig, ax = plt.subplots(figsize=(12, 7))
    pal = {"重点产业": "#d62728", "其他产业": "#1f77b4"}
    sorted_df = df_metrics.sort_values(col, ascending=False)
    # 这里的 col 和 '产业名称' 是你的中文变量名
    sns.barplot(data=sorted_df, x=col, y="产业名称", hue="分类", dodge=False, ax=ax, palette=pal)
    set_ax_font(ax, title, col, "产业名称")
    return fig

def plot_scale_pie(df, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = df['企业划型名称'].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
           colors=sns.color_palette('pastel'),
           textprops={'fontproperties': my_font}) # 饼图内部文字
    ax.set_title(title, fontproperties=my_font, fontweight='bold', fontsize=15)
    return fig

def plot_age_dist(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['企业年限'].dropna(), bins=15, kde=True, ax=ax, color='teal')
    # 强制将 Seaborn 默认的 'count' 改为 '频数'
    set_ax_font(ax, title, "企业年限", "频数") 
    return fig

def plot_metric_trend(df_sub, col, label, r_val):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='专利数量_f', y=col, data=df_sub, ax=ax, scatter_kws={'alpha':0.4}, line_kws={'color':'#d62728'})
    ax.set_ylim(0, 110)
    set_ax_font(ax, f'【趋势】{label} vs 专利数量 (R={r_val})', "专利数量", label)
    return fig

def plot_metric_violin(df_sub, col, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='企业划型名称', y=col, data=df_sub, ax=ax, palette='Blues_r', cut=0, order=['大型','中型','小型','微型'])
    ax.set_ylim(0, 110)
    set_ax_font(ax, f'【分布密度】{label} vs 企业规模', "企业规模", label)
    return fig

def plot_bubble_chart(df_sub):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(df_sub['支撑得分'], df_sub['护城河得分'], s=df_sub['专利数量_f'] * 3 + 30, alpha=0.6, c='#1f77b4', edgecolors='w')
    sns.regplot(x='支撑得分', y='护城河得分', data=df_sub, scatter=False, color='#d62728', line_kws={'ls': '--'})
    ax.set_ylim(0, 110); ax.set_xlim(0, 110)
    set_ax_font(ax, '研发聚焦度 vs. 技术护城河深度', "业务支撑指数(研发聚焦度)", "技术护城河深度")
    return fig

def plot_age_matrix_row(df_sub, col, label):
    age_order = ['青年企业(≤5年)', '中坚企业(6-15年)', '老牌企业(>15年)']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, group in enumerate(age_order):
        subset = df_sub[df_sub['年限梯队'] == group]
        if not subset.empty:
            sns.histplot(subset[col], kde=True, ax=axes[j], color='steelblue')
            axes[j].axvline(subset[col].mean(), color='red', linestyle='--')
            set_ax_font(axes[j], group, label, "频数") # 强制中文标签
            axes[j].set_xlim(0, 110)
    return fig

def plot_time_trend_sd(df_sub, col, label, color):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_sub, x='企业年限', y=col, ax=ax, color=color, marker='o', errorbar='sd')
    ax.set_ylim(0, 110); ax.axvline(x=15, color='red', linestyle='--')
    set_ax_font(ax, f'{label} 随成立年限变化趋势', "企业年限", label)
    return fig
