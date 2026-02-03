import pandas as pd
import numpy as np

def clean_pct(x):
    """处理百分比字符串与数值的统一转换"""
    if pd.isna(x) or x == 'NA': return np.nan
    try:
        val_str = str(x).replace('%', '')
        val = float(val_str)
        return val if val > 1 else val * 100
    except: return np.nan

def process_chain_data(df):
    """严格映射审计所需的清洗字段"""
    df.columns = df.columns.str.strip()
    df['注册资本_f'] = pd.to_numeric(df.get('注册资本', 0), errors='coerce').fillna(0) / 10000 
    df['专利数量_f'] = pd.to_numeric(df.get('专利数量', 0), errors='coerce').fillna(0)
    df['资质总量_f'] = pd.to_numeric(df.get('企业资质总量', 0), errors='coerce').fillna(0)
    
    if '成立日期' in df.columns:
        df['成立日期'] = pd.to_datetime(df['成立日期'], errors='coerce')
        df['企业年限'] = df['成立日期'].apply(lambda x: 2024 - x.year if pd.notnull(x) else np.nan)
    
    if '公司所在地' in df.columns:
        df['区县'] = df['公司所在地'].apply(lambda x: str(x).split(',')[-1].strip() if pd.notnull(x) else "未知")

    # 审计核心三指标
    df['支撑得分'] = df.get('业务支撑指数', pd.Series(dtype=float)).apply(clean_pct)
    df['覆盖得分'] = df.get('产品技术覆盖率', pd.Series(dtype=float)).apply(clean_pct)
    df['护城河得分'] = pd.to_numeric(df.get('技术护城河深度', np.nan), errors='coerce')

    df['是否融资'] = (pd.to_numeric(df.get('融资次数', 0), errors='coerce').fillna(0) > 0).map({True: '获投企业', False: '未获投'})
    
    def age_group(age):
        if pd.isna(age): return '未知'
        return '青年企业(≤5年)' if age <= 5 else '中坚企业(6-15年)' if age <= 15 else '老牌企业(>15年)'
    df['年限梯队'] = df['企业年限'].apply(age_group)
    return df