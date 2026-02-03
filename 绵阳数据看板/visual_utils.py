import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
import os

# --- 字体注入逻辑 ---
# 获取当前文件所在目录，定位 fonts 文件夹
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(CURRENT_DIR, "fonts", "simhei.ttf")

# 加载字体属性对象
if os.path.exists(FONT_PATH):
    my_font = fm.FontProperties(fname=FONT_PATH)
else:
    # 兜底：如果没找到字体，使用系统默认并打印警告（方便调试）
    my_font = None 
    print(f"⚠️ 警告: 未在路径 {FONT_PATH} 找到字体文件")

# 基础样式配置
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号乱码问题

def plot_mianyang_ranking(df_metrics, col, title):
    fig, ax = plt.subplots(figsize=(12, 7))
    pal = {"重点产业": "#d62728", "其他产业": "#1f77b4"}
    sorted_df = df_metrics.sort_values(col, ascending=False)
    sns.barplot(data=sorted_df, x=col, y="产业名称", hue="分类", dodge=False, ax=ax, palette=pal)
    
    # 注入字体：标题、坐标轴、刻度、图例
    ax.set_title(title, fontweight='bold', fontsize=16, fontproperties=my_font)
    ax.set_xlabel(col, fontproperties=my_font)
    ax.set_ylabel("产业名称", fontproperties=my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    for tick in ax.get_yticklabels(): tick.set_fontproperties(my_font)
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), fontproperties=my_font)
    return fig

def plot_special_geo_stacked(df_all_raw, special_list):
    df_spec = df_all_raw[df_all_raw['所属产业集群'].isin(special_list)].copy()
    geo_spec = df_spec.groupby(['区县', '所属产业集群']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    geo_spec.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    
    ax.set_title("重点科技产业之区县分布", fontsize=16, fontweight='bold', fontproperties=my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    for tick in ax.get_yticklabels(): tick.set_fontproperties(my_font)
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), fontproperties=my_font)
    plt.xticks(rotation=0)
    return fig

def plot_scale_pie(df, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    counts = df['企业划型名称'].value_counts()
    # 饼图通过 textprops 注入字体
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
           colors=sns.color_palette('pastel'),
           textprops={'fontproperties': my_font})
    ax.set_title(title, fontsize=15, fontweight='bold', fontproperties=my_font)
    return fig

def plot_age_dist(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['企业年限'].dropna(), bins=15, kde=True, ax=ax, color='teal')
    ax.set_title(title, fontsize=15, fontweight='bold', fontproperties=my_font)
    ax.set_xlabel("企业年限", fontproperties=my_font)
    ax.set_ylabel("频数", fontproperties=my_font)
    return fig

def plot_capital_dist(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    bins = [0, 100, 500, 1000, 5000, 10000, np.inf]
    labels = ['<100万', '100-500万', '500-1000万', '1000-5000万', '5000万-1亿', '>1亿']
    df['资本区间'] = pd.cut(df['注册资本_f'], bins=bins, labels=labels, include_lowest=True)
    sns.countplot(x='资本区间', data=df, ax=ax, palette='Blues_r', hue='资本区间', legend=False)
    ax.set_title('注册资本梯队分布 (单位: 万元)', fontsize=15, fontweight='bold', fontproperties=my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    ax.set_xlabel("资本区间", fontproperties=my_font)
    return fig

def plot_region_bar(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df['区县'].value_counts()
    sns.barplot(x=data.values, y=data.index, palette='coolwarm', ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold', fontproperties=my_font)
    for tick in ax.get_yticklabels(): tick.set_fontproperties(my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    ax.set_xlabel("企业数量", fontproperties=my_font)
    return fig

def plot_risk_barh(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    risk_cols = ['经营异常', '严重违法', '行政处罚', '被执行人', '失信被执行人', '对外担保', '股权出质']
    risk_summary = df[risk_cols].apply(lambda x: (x == '是').sum()).sort_values()
    risk_summary.plot(kind='barh', ax=ax, color='salmon')
    ax.set_title('企业风险项统计', fontsize=15, fontweight='bold', fontproperties=my_font)
    for tick in ax.get_yticklabels(): tick.set_fontproperties(my_font)
    return fig

def plot_qual_dist(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['资质总量_f'], bins=15, kde=True, ax=ax, color='orange')
    ax.set_title('企业资质总量分布直方图', fontsize=16, fontweight='bold', fontproperties=my_font)
    ax.set_xlabel("资质总量", fontproperties=my_font)
    return fig

def plot_metric_trend(df_sub, col, label, r_val):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='专利数量_f', y=col, data=df_sub, ax=ax, scatter_kws={'alpha':0.4}, line_kws={'color':'#d62728'})
    ax.set_ylim(0, 110)
    ax.set_title(f'【趋势】{label} vs 专利数量 (R={r_val})', fontweight='bold', fontproperties=my_font)
    ax.set_xlabel("专利数量", fontproperties=my_font)
    ax.set_ylabel(label, fontproperties=my_font)
    return fig

def plot_metric_violin(df_sub, col, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='企业划型名称', y=col, data=df_sub, ax=ax, palette='Blues_r', cut=0, order=['大型','中型','小型','微型'])
    ax.set_ylim(0, 110)
    ax.set_title(f'【分布密度】{label} vs 企业规模', fontweight='bold', fontproperties=my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    ax.set_xlabel("企业规模", fontproperties=my_font)
    ax.set_ylabel(label, fontproperties=my_font)
    return fig

def plot_bubble_chart(df_sub):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(df_sub['支撑得分'], df_sub['护城河得分'], s=df_sub['专利数量_f'] * 3 + 30, alpha=0.6, c='#1f77b4', edgecolors='w')
    sns.regplot(x='支撑得分', y='护城河得分', data=df_sub, scatter=False, color='#d62728', line_kws={'ls': '--'})
    ax.set_ylim(0, 110); ax.set_xlim(0, 110)
    ax.set_xlabel("业务支撑指数(研发聚焦度)", fontproperties=my_font)
    ax.set_ylabel("技术护城河深度", fontproperties=my_font)
    ax.set_title('研发聚焦度 vs. 技术护城河深度', fontsize=18, fontweight='bold', fontproperties=my_font)
    return fig

def plot_funding_box(df_sub, col, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.boxplot(x='是否融资', y=col, data=df_sub, ax=ax, palette=['#1f77b4', '#aec7e8'], order=['获投企业', '未获投'])
    ax.set_ylim(0, 110)
    ax.set_title(title, fontweight='bold', fontproperties=my_font)
    for tick in ax.get_xticklabels(): tick.set_fontproperties(my_font)
    ax.set_xlabel("融资状态", fontproperties=my_font)
    return fig

def plot_age_matrix_row(df_sub, col, label):
    age_order = ['青年企业(≤5年)', '中坚企业(6-15年)', '老牌企业(>15年)']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, group in enumerate(age_order):
        subset = df_sub[df_sub['年限梯队'] == group]
        if not subset.empty:
            sns.histplot(subset[col], kde=True, ax=axes[j], color='steelblue')
            axes[j].axvline(subset[col].mean(), color='red', linestyle='--')
            axes[j].set_title(f"【{group}】", fontproperties=my_font)
            axes[j].set_xlabel(f"均值:{subset[col].mean():.1f}", fontproperties=my_font)
            axes[j].set_xlim(0, 110)
    return fig

def plot_time_trend_sd(df_sub, col, label, color):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_sub, x='企业年限', y=col, ax=ax, color=color, marker='o', errorbar='sd')
    ax.set_ylim(0, 110); ax.axvline(x=15, color='red', linestyle='--')
    ax.set_title(f'{label} 随成立年限变化趋势', fontweight='bold', fontproperties=my_font)
    ax.set_xlabel("企业年限", fontproperties=my_font)
    ax.set_ylabel(label, fontproperties=my_font)
    return fig



