import pandas as pd
import os

# ================= 配置区域 =================
# 1. 你的“本地数据库”文件 (Excel 格式)
SOURCE_DB_FILE = "all_final_again_pubchem_cleaned.xlsx"

# 2. 待填写的成分表 (第一阶段生成的 CSV)
TARGET_FILE = "pesticide_list_to_fill_final.csv"

# 3. 输出文件
OUTPUT_FILE = "pesticide_list_matched_local.csv"
# ===========================================

def normalize_name(name):
    """名称标准化：转小写，去首尾空格"""
    if pd.isna(name): return ""
    return str(name).strip().lower()

def find_column_fuzzy(df, keywords):
    """模糊查找列名"""
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in str(col).lower():
                return col
    return None

def run_local_matching_xlsx():
    print(f"正在读取本地 Excel 数据库: {SOURCE_DB_FILE} ...")
    
    try:
        # 使用 openpyxl 引擎读取 Excel
        # 默认读取第一个 Sheet
        df_source = pd.read_excel(SOURCE_DB_FILE, engine='openpyxl')
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        print("请确认文件名正确，并已安装 openpyxl (pip install openpyxl)")
        return

    print(f"正在读取待填表: {TARGET_FILE} ...")
    if not os.path.exists(TARGET_FILE):
        print(f"错误：找不到待填表 {TARGET_FILE}")
        return
    df_target = pd.read_csv(TARGET_FILE)

    # 1. 自动探测列名
    print("正在分析 Excel 列名...")
    # 找“英文名”列 (关键词：English, Name, Ingredient, 英文)
    src_name_col = find_column_fuzzy(df_source, ['English', 'Name', 'Ingredient', '英文'])
    # 找“SMILES”列 (关键词：SMILES, smiles)
    src_smiles_col = find_column_fuzzy(df_source, ['SMILES', 'smiles'])

    if not src_name_col or not src_smiles_col:
        print("错误：无法自动识别列名。")
        print(f"检测到的列名: {df_source.columns.tolist()}")
        return

    print(f"锁定列名映射: 英文名=[{src_name_col}], SMILES=[{src_smiles_col}]")

    # 2. 构建查找字典
    # 过滤掉没有 SMILES 的行
    df_source_clean = df_source.dropna(subset=[src_name_col, src_smiles_col])
    
    name_to_smiles = {}
    count_source = 0
    for _, row in df_source_clean.iterrows():
        norm_name = normalize_name(row[src_name_col])
        smiles = str(row[src_smiles_col]).strip()
        
        # 只有当 SMILES 有效时才存入字典
        if norm_name and smiles and smiles.lower() != 'nan':
            name_to_smiles[norm_name] = smiles
            count_source += 1
            
    print(f"本地知识库构建完成，有效词条数: {count_source}")

    # 3. 执行匹配
    if 'SMILES' not in df_target.columns:
        df_target['SMILES'] = ""
        
    matched_count = 0
    total_rows = len(df_target)
    
    print("开始匹配...")
    for index, row in df_target.iterrows():
        # 如果已有 SMILES 则跳过
        if pd.notna(row['SMILES']) and str(row['SMILES']).strip() != "":
            continue
            
        target_name = row.get('English_Name', '')
        norm_target_name = normalize_name(target_name)
        
        # 查字典
        if norm_target_name in name_to_smiles:
            df_target.at[index, 'SMILES'] = name_to_smiles[norm_target_name]
            matched_count += 1
    
    # 4. 保存
    df_target.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*30)
    print(f"匹配完成！")
    print(f"新增匹配: {matched_count} 条")
    print(f"当前覆盖率: {len(df_target[df_target['SMILES'] != ''])} / {total_rows}")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("="*30)

if __name__ == "__main__":
    run_local_matching_xlsx()