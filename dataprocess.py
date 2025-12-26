import pandas as pd
import re
import os

# ================= 配置区域 =================
INPUT_FILE = "1109-2WXPesticides Data1.xlsx" 
OUTPUT_DATA_FILE = "cleaned_dataset_stage1_final.csv"
OUTPUT_PEST_LIST = "pesticide_list_to_fill_final.csv"

# 列名映射 (请确保与您的 Excel 表头一致)
COL_ID = "Registered number"
COL_PEST_NAME = "农药名称"
COL_INGREDIENT = "有效成分英文名"   
COL_PLANT = "作物/场所"
COL_DISEASE = "防治对象"
COL_FORMULATION = "剂型"      
# ===========================================

def split_multi_values(text):
    """
    [核心拆分函数]
    支持分隔符：换行符, 逗号, 顿号, 斜杠, 空格
    """
    if not isinstance(text, str):
        return []
    # 过滤掉纯数字干扰
    text = str(text)
    # 正则：匹配任意一个或多个分隔符
    items = re.split(r'[\n\r、,，/ ]+', text.strip())
    # 过滤空字符串，并去除每个词的首尾空格
    return [item.strip() for item in items if item.strip()]

def clean_text_entity(text):
    """
    清洗实体名称：去除括号及内容
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    # 去除中文或英文括号及其内部内容
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"（.*?）", "", text)
    return text.strip()

def is_technical_material(row):
    """
    判断是否为原药
    """
    pest_name = str(row.get(COL_PEST_NAME, ""))
    formulation = str(row.get(COL_FORMULATION, ""))
    if "原药" in pest_name or "原药" in formulation:
        return True
    return False

def get_aligned_pairs(row):
    """
    [核心对齐逻辑]
    """
    plants = row['plant_list']
    diseases = row['disease_list']
    
    num_p = len(plants)
    num_d = len(diseases)
    
    pairs = []
    
    # 1. 严格一一对应 (N 对 N)
    if num_p == num_d and num_p > 0:
        pairs = list(zip(plants, diseases))
        
    # 2. 作物广播 (1 对 N)
    elif num_p == 1 and num_d > 1:
        pairs = [(plants[0], d) for d in diseases]
        
    # 3. 病害广播 (N 对 1)
    elif num_p > 1 and num_d == 1:
        pairs = [(p, diseases[0]) for p in plants]
        
    # 4. 数量不一致 (N 对 M) -> 截断
    elif num_p > 0 and num_d > 0:
        pairs = list(zip(plants, diseases))
    
    # 5. 空
    else:
        pairs = []
        
    return pairs

def extract_pure_name(text):
    """
    [修改版] 提取活性成分名
    仅去除首尾空格，不进行去数字、去符号等额外处理
    """
    if pd.isna(text):
        return ""
    # 仅转换为字符串并去除首尾空格
    return str(text).strip()

def process_stage_1_final():
    print(f"正在读取 Excel 文件: {INPUT_FILE} ...")
    
    try:
        # 指定 sheet_name='一种活性成分'
        df = pd.read_excel(INPUT_FILE, engine='openpyxl', sheet_name='一种活性成分')
    except Exception as e:
        print(f"读取失败: {e}")
        print("请确认文件名正确且已安装 openpyxl 库")
        return

    print(f"原始行数: {len(df)}")

    # 1. 基础清洗
    df.dropna(how='all', inplace=True)
    df.fillna("", inplace=True) 

    # 2. 剔除原药
    df['is_tc'] = df.apply(is_technical_material, axis=1)
    num_tc = df['is_tc'].sum()
    df = df[df['is_tc'] == False].copy()
    print(f"已剔除原药数据: {num_tc} 条")
    
    # 3. 确保核心列有内容
    df = df[df[COL_INGREDIENT].astype(str).str.len() > 0]
    df = df[df[COL_PLANT].astype(str).str.len() > 0]
    df = df[df[COL_DISEASE].astype(str).str.len() > 0]

    # 4. 文本清洗与拆分 (作物和病害依然需要拆分)
    print("正在拆分作物与病害...")
    df[COL_PLANT] = df[COL_PLANT].apply(clean_text_entity)
    df[COL_DISEASE] = df[COL_DISEASE].apply(clean_text_entity)
    
    df['plant_list'] = df[COL_PLANT].apply(split_multi_values)
    df['disease_list'] = df[COL_DISEASE].apply(split_multi_values)

    # 5. 生成对齐的 Pair
    df['pairs'] = df.apply(get_aligned_pairs, axis=1)

    # 6. 炸开 (Explode)
    df_exploded = df.explode('pairs')
    df_exploded.dropna(subset=['pairs'], inplace=True)
    
    df_exploded['final_plant'] = df_exploded['pairs'].apply(lambda x: x[0])
    df_exploded['final_disease'] = df_exploded['pairs'].apply(lambda x: x[1])

    print(f"拆分处理后的三元组数量: {len(df_exploded)}")

    # 7. [修改] 提取纯净成分名 (只做 strip)
    df_exploded['pure_ingredient_name'] = df_exploded[COL_INGREDIENT].apply(extract_pure_name)

    # 8. 导出清洗后的全量数据
    final_cols = [COL_ID, 'pure_ingredient_name', 'final_plant', 'final_disease']
    df_export = df_exploded[final_cols].rename(columns={
        'final_plant': COL_PLANT, 
        'final_disease': COL_DISEASE
    })
    
    df_export.to_csv(OUTPUT_DATA_FILE, index=False, encoding='utf-8-sig')
    print(f"清洗后的数据已保存至: {OUTPUT_DATA_FILE}")

    # 9. 导出唯一的成分表
    unique_pesticides = df_export['pure_ingredient_name'].unique()
    unique_pesticides = [p for p in unique_pesticides if p] # 去空
    
    # 这里将列名设为 English_Name，方便后续 PubChem 脚本直接使用
    df_unique = pd.DataFrame(unique_pesticides, columns=['English_Name'])
    df_unique['SMILES'] = "" 
    df_unique.to_csv(OUTPUT_PEST_LIST, index=False, encoding='utf-8-sig')
    
    print(f"\n[完成] 共提取 {len(unique_pesticides)} 种活性成分。")
    print(f"请使用 {OUTPUT_PEST_LIST} 进行 SMILES 查询。")

if __name__ == "__main__":
    process_stage_1_final()