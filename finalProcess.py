import pandas as pd

def filter_missing_smiles():
    print("正在读取文件...")
    # 读取三元组数据 和 SMILES对照表
    df_stage1 = pd.read_csv('cleaned_dataset_stage1_final.csv')
    df_smiles = pd.read_csv('pesticide_list_matched_local.csv')

    print(f"原始农药种类: {len(df_smiles)}")
    print(f"原始数据行数: {len(df_stage1)}")

    # 1. 过滤 SMILES 表
    # 剔除 SMILES 为空 (NaN) 或 空字符串 的行
    df_smiles_clean = df_smiles[df_smiles['SMILES'].notna() & (df_smiles['SMILES'].str.strip() != '')].copy()
    
    # 获取有效的农药英文名列表 (Set结构查找更快)
    valid_pesticides = set(df_smiles_clean['English_Name'].str.strip().unique())

    print(f"保留有效农药: {len(valid_pesticides)} 种 (剔除了 {len(df_smiles) - len(valid_pesticides)} 种)")

    # 2. 过滤主数据集
    # 确保列名为字符串并去空格，防止匹配失败
    df_stage1['pure_ingredient_name'] = df_stage1['pure_ingredient_name'].astype(str).str.strip()
    
    # 只保留那些 "pure_ingredient_name" 在 "valid_pesticides" 里的行
    df_stage1_clean = df_stage1[df_stage1['pure_ingredient_name'].isin(valid_pesticides)].copy()

    print(f"保留有效数据: {len(df_stage1_clean)} 行 (剔除了 {len(df_stage1) - len(df_stage1_clean)} 行)")

    # 3. 保存结果
    output_data_file = 'cleaned_dataset_stage1_final_filtered.csv'
    output_list_file = 'pesticide_list_final_filtered.csv'
    
    df_stage1_clean.to_csv(output_data_file, index=False, encoding='utf-8-sig')
    df_smiles_clean.to_csv(output_list_file, index=False, encoding='utf-8-sig')
    
    print(f"\n处理完成！")
    print(f"清洗后的数据集已保存至: {output_data_file}")
    print(f"清洗后的成分表已保存至: {output_list_file}")

if __name__ == "__main__":
    filter_missing_smiles()