import pandas as pd
import os
from datetime import datetime

# --------------------------
# 参数设置（已适配你的场景）
# --------------------------
folder_path = "D:/stock data"  # 股票文件文件夹
output_path = "D:/stock data.csv"  # 输出路径
date_column_name = "trade_date"  # 日期列名
raw_date_format = "%Y%m%d"  # 原始格式：20240930（8位数字）
target_date_format = "%Y-%m-%d"  # 目标格式：强制两位数月/日（2024-09-30）

# --------------------------
# 第一步：读取文件并严格校验日期格式
# --------------------------
all_stocks = []
invalid_dates = []  # 记录不规范的日期（方便排查）

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        stock_code = filename.replace(".csv", "")
        file_path = os.path.join(folder_path, filename)
        try:
            # 强制日期列读为字符串（避免自动截断前导0，如"09"变"9"）
            df = pd.read_csv(file_path, dtype={date_column_name: str})
        except Exception as e:
            print(f"⚠️ 跳过文件 {filename}：{e}")
            continue

        if date_column_name not in df.columns:
            print(f"⚠️ 文件 {filename} 无「{date_column_name}」列，已跳过")
            continue

        # 检查原始日期是否为8位数字（关键：避免7位或9位导致解析错误）
        df["is_valid_length"] = df[date_column_name].str.len() == 8  # 8位才有效
        invalid = df[~df["is_valid_length"]]
        if not invalid.empty:
            invalid_samples = invalid[date_column_name].head(3).tolist()
            invalid_dates.extend([(filename, d) for d in invalid_samples])
            print(f"⚠️ 文件 {filename} 存在非8位日期（如{invalid_samples}），已过滤")

        # 只保留8位日期的数据，添加股票代码
        df_valid_length = df[df["is_valid_length"]].copy()
        df_valid_length["ts_code"] = stock_code
        df_valid_length["original_date"] = df_valid_length[date_column_name]  # 保留原始值
        all_stocks.append(df_valid_length)

if not all_stocks:
    print("❌ 未读取到有效数据，请检查文件")
    exit()

# 打印不规范日期（如有）
if invalid_dates:
    print("\n❌ 检测到非8位日期（需格式为20240930）：")
    for fn, d in invalid_dates[:5]:
        print(f"- 文件 {fn}：{d}")


# --------------------------
# 第二步：强制解析为两位数月/日（核心优化）
# --------------------------
def parse_date(raw_date):
    """将8位数字（20240930）转为2024-09-30（强制两位数月/日）"""
    try:
        # 严格按8位格式解析
        dt = datetime.strptime(raw_date, raw_date_format)
        # 用target_date_format强制输出两位数（%m和%d会自动补0）
        return dt.strftime(target_date_format)
    except:
        return f"解析失败：{raw_date}"


# 合并数据并解析
df_total = pd.concat(all_stocks, ignore_index=True)
df_total["parsed_date"] = df_total[date_column_name].apply(parse_date)

# 打印解析示例（验证双位数日是否正确）
print("\n📌 解析后日期示例（检查双位数日）：")
sample_dates = df_total[["original_date", "parsed_date"]].drop_duplicates().head(5)
for _, row in sample_dates.iterrows():
    print(f"原始：{row['original_date']} → 解析后：{row['parsed_date']}")

# --------------------------
# 第三步：筛选2024年数据并处理显示问题
# --------------------------
# 保留解析成功的日期（格式为YYYY-MM-DD）
df_success = df_total[df_total["parsed_date"].str.match(r"\d{4}-\d{2}-\d{2}")].copy()

# 筛选2024年
df_2024 = df_success[
    (df_success["parsed_date"] >= "2024-01-01") &
    (df_success["parsed_date"] <= "2024-12-31")
    ]

if len(df_2024) == 0:
    print("\n❌ 未找到2024年有效数据")
    exit()

print(f"\n✅ 筛选到2024年数据：{len(df_2024)}行，日期范围：{df_2024['parsed_date'].min()}至{df_2024['parsed_date'].max()}")

# --------------------------
# 第四步：保存为兼容格式（解决Excel显示#####问题）
# --------------------------
# 1. 确保日期格式为YYYY-MM-DD（Excel可识别的标准格式）
# 2. 保存时不压缩列宽，Excel打开后双击列标题即可自动调整宽度

# 保留关键列
result = df_2024[["ts_code", "original_date", "parsed_date", "close_qfq", "pct_chg"]]
result.to_csv(output_path, index=False)

print(f"\n🎉 数据已保存至：{output_path}")
print("💡 解决Excel显示#####的方法：打开文件后，双击parsed_date列的列标题右侧边缘（自动调整列宽）")

# --------------------------
# 额外：生成Excel格式文件（可选，更兼容）
# --------------------------
excel_path = output_path.replace(".csv", ".xlsx")
result.to_excel(excel_path, index=False)
print(f"📌 同时生成Excel格式文件（自动适配列宽）：{excel_path}")