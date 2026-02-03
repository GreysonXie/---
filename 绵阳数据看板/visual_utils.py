import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 移除中文字体依赖，使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False
# 设置全局风格
sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_mianyang_ranking(df_metrics, col, title):
    fig, ax = plt.subplots(figsize=(12, 7))
    # 类别翻译
    pal = {"Key Industry": "#d62728", "Other Industries": "#1f77b4"}
    sorted_df = df_metrics.sort_values(col, ascending=False).copy()
    # 假设数据中的分类字段已映射或直接在此处映射显示
    sorted_df['Category_EN'] = sorted_df['分类'].map({"重点产业": "Key Industry", "其他产业": "Other Industries"})
    
    sns.barplot(data=sorted_df, x=col, y="产业名称", hue="Category_EN", dodge=False, ax=ax, palette=pal)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_xlabel(col)
    ax.set_ylabel("Industry Name")
    return fig

def plot_special_geo_stacked(df_all_raw, special_list):
    df_spec = df_all_raw[df_all_raw['所属产业集群'].isin(special_list)].copy()
    geo_spec = df_spec.groupby(['区县', '所属产业集群']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 7))
    geo_spec.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    ax.set_title("District Distribution of Key Tech Industries", fontsize=16, fontweight='bold')
    ax.set_xlabel("District")
    ax.set_ylabel("Company Count")
    plt.xticks(rotation=45)
    return fig

def plot_scale_pie(df, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    # 规模标签映射
    scale_map = {"大型": "Large", "中型": "Medium", "小型": "Small", "微型": "Micro"}
    counts = df['企业划型名称'].value_counts()
    labels = [scale_map.get(x, x) for x in counts.index]
    
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title(title, fontsize=15, fontweight='bold')
    return fig

def plot_age_dist(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['企业年限'].dropna(), bins=15, kde=True, ax=ax, color='teal')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel("Years in Business")
    ax.set_ylabel("Frequency")
    return fig

def plot_capital_dist(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    bins = [0, 100, 500, 1000, 5000, 10000, np.inf]
    # 资本区间翻译
    labels = ['<1M', '1-5M', '5-10M', '10-50M', '50-100M', '>100M']
    df['Capital_Range'] = pd.cut(df['注册资本_f'], bins=bins, labels=labels, include_lowest=True)
    sns.countplot(x='Capital_Range', data=df, ax=ax, palette='Blues_r', hue='Capital_Range', legend=False)
    ax.set_title('Registered Capital Distribution (Unit: 10k CNY)', fontsize=15, fontweight='bold')
    ax.set_xlabel("Capital Range")
    ax.set_ylabel("Count")
    return fig

def plot_region_bar(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df['区县'].value_counts()
    sns.barplot(x=data.values, y=data.index, palette='coolwarm', ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Company Count")
    ax.set_ylabel("District")
    return fig

def plot_risk_barh(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    # 风险项翻译参考
    risk_cols = ['经营异常', '严重违法', '行政处罚', '被执行人', '失信被执行人', '对外担保', '股权出质']
    risk_labels = ['Op. Abnormal', 'Serious Violations', 'Admin Penalty', 'Judgment Debtor', 'Dishonest Debtor', 'Guarantee', 'Pledge']
    
    risk_summary = df[risk_cols].apply(lambda x: (x == '是').sum()).sort_values()
    # 映射 y 轴标签为英文
    risk_mapping = dict(zip(risk_cols, risk_labels))
    risk_summary.index = [risk_mapping.get(x, x) for x in risk_summary.index]
    
    risk_summary.plot(kind='barh', ax=ax, color='salmon')
    ax.set_title('Corporate Risk Items Statistics', fontsize=15, fontweight='bold')
    ax.set_xlabel("Number of Companies")
    return fig

def plot_qual_dist(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['资质总量_f'], bins=15, kde=True, ax=ax, color='orange')
    ax.set_title('Corporate Qualifications Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel("Total Qualifications")
    ax.set_ylabel("Frequency")
    return fig

def plot_metric_trend(df_sub, col, label, r_val):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='专利数量_f', y=col, data=df_sub, ax=ax, scatter_kws={'alpha':0.4}, line_kws={'color':'#d62728'})
    ax.set_ylim(0, 110)
    # 标签翻译
    label_en_map = {"业务支撑指数": "Business Support Index", "产品技术覆盖率": "Product Coverage", "技术护城河深度": "Tech Moat Depth"}
    label_en = label_en_map.get(label, label)
    ax.set_title(f'[Trend] {label_en} vs Patents (R={r_val})', fontweight='bold')
    ax.set_xlabel("Number of Patents")
    ax.set_ylabel("Score")
    return fig

def plot_metric_violin(df_sub, col, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    # 映射规模标签
    df_sub = df_sub.copy()
    scale_map = {"大型": "Large", "中型": "Medium", "小型": "Small", "微型": "Micro"}
    df_sub['Scale_EN'] = df_sub['企业划型名称'].map(scale_map)
    
    sns.violinplot(x='Scale_EN', y=col, data=df_sub, ax=ax, palette='Blues_r', cut=0, order=['Large','Medium','Small','Micro'])
    ax.set_ylim(0, 110)
    label_en_map = {"业务支撑指数": "Business Support Index", "产品技术覆盖率": "Product Coverage", "技术护城河深度": "Tech Moat Depth"}
    label_en = label_en_map.get(label, label)
    ax.set_title(f'[Density] {label_en} vs Company Scale', fontweight='bold')
    ax.set_xlabel("Company Scale")
    ax.set_ylabel("Score")
    return fig

def plot_bubble_chart(df_sub):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(df_sub['支撑得分'], df_sub['护城河得分'], s=df_sub['专利数量_f'] * 3 + 30, alpha=0.6, c='#1f77b4', edgecolors='w')
    sns.regplot(x='支撑得分', y='护城河得分', data=df_sub, scatter=False, color='#d62728', line_kws={'ls': '--'})
    ax.set_ylim(0, 110); ax.set_xlim(0, 110)
    ax.set_xlabel("Business Support Index (R&D Focus)")
    ax.set_ylabel("Tech Moat Depth")
    ax.set_title('R&D Focus vs. Tech Moat Depth', fontsize=18, fontweight='bold')
    return fig

def plot_funding_box(df_sub, col, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    # 融资状态翻译
    df_sub = df_sub.copy()
    funding_map = {"获投企业": "Funded", "未获投": "Not Funded"}
    df_sub['Funding_Status'] = df_sub['是否融资'].map(funding_map)
    
    sns.boxplot(x='Funding_Status', y=col, data=df_sub, ax=ax, palette=['#1f77b4', '#aec7e8'], order=['Funded', 'Not Funded'])
    ax.set_ylim(0, 110)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Score")
    ax.set_xlabel("Funding Status")
    return fig

def plot_age_matrix_row(df_sub, col, label):
    # 年限梯队翻译
    age_order_cn = ['青年企业(≤5年)', '中坚企业(6-15年)', '老牌企业(>15年)']
    age_order_en = ['Youth (<=5y)', 'Backbone (6-15y)', 'Established (>15y)']
    age_map = dict(zip(age_order_cn, age_order_en))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, group_cn in enumerate(age_order_cn):
        group_en = age_map[group_cn]
        subset = df_sub[df_sub['年限梯队'] == group_cn]
        if not subset.empty:
            sns.histplot(subset[col], kde=True, ax=axes[j], color='steelblue')
            axes[j].axvline(subset[col].mean(), color='red', linestyle='--')
            axes[j].set_title(f"{group_en} Mean: {subset[col].mean():.1f}")
            axes[j].set_xlim(0, 110)
            axes[j].set_xlabel("Score")
    return fig

def plot_time_trend_sd(df_sub, col, label, color):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_sub, x='企业年限', y=col, ax=ax, color=color, marker='o', errorbar='sd')
    ax.set_ylim(0, 110)
    # 15年分水岭
    ax.axvline(x=15, color='red', linestyle='--', label='15-Year Watershed')
    ax.set_title(f'{label} Trend by Years in Business', fontweight='bold')
    ax.set_xlabel("Years in Business")
    ax.set_ylabel("Score")
    ax.legend()
    return fig

