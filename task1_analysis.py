# 1. 导入依赖库（若缺失，执行命令：pip install pandas seaborn matplotlib openpyxl numpy）
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

# 2. 核心参数配置（严格适配文档“沪深300成分股数据”与“军工板块股票”要求）
CONFIG: Dict = {
    "data_path": r"D:\stock data.xlsx",  # 文档指定数据来源路径
    "sheet_name": "Sheet1",  # 数据工作表名（文档未指定，默认Sheet1）
    "start_date": "2024-01-01",  # 分析时间范围（可按需调整）
    "end_date": "2024-09-30",
    "military_stocks": {  # 文档任务一隐含“军工板块相关股票”，此处为用户指定标的
        "601698.SH": "中国卫通",
        "002049.SZ": "紫光国微",
        "600111.SH": "北方稀土",
        "600893.SH": "航发动力",
        "000425.SZ": "徐工机械",
        "600760.SH": "中航沈飞",
        "002179.SZ": "中航光电",
        "002180.SZ": "纳思达",
        "600372.SH": "中航机载",
        "000768.SZ": "中航西飞",
        "600150.SH": "中国船舶",
        "000800.SZ": "一汽解放"
    },
    "fields": {  # 适配文档“股票代码、时间戳、收盘价”数据结构
        "stock_code": "ts_code",
        "date": "parsed_date",
        "close_price": "close_qfq"
    }
}


# 3. 数据读取函数（处理文档数据格式异常）
def load_stock_data(config: Dict) -> pd.DataFrame:
    try:
        df = pd.read_excel(
            io=config["data_path"],
            sheet_name=config["sheet_name"],
            engine="openpyxl"
        )
        print(f"✅ 读取文档数据：{df.shape[0]}行 × {df.shape[1]}列，字段：{list(df.columns)}")

        # 验证文档要求的关键字段
        required_fields = [config["fields"]["stock_code"], config["fields"]["date"], config["fields"]["close_price"]]
        if missing := [f for f in required_fields if f not in df.columns]:
            raise ValueError(f"❌ 缺失文档要求字段：{missing}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 未找到文档数据：{config['data_path']}")
    except Exception as e:
        raise RuntimeError(f"❌ 数据读取失败：{str(e)}")


# 4. 数据预处理函数（复用文档任务一的时间序列筛选逻辑）
def preprocess_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    # 筛选军工股
    sc_field = config["fields"]["stock_code"]
    df_military = df[df[sc_field].isin(config["military_stocks"].keys())].copy()
    if df_military.empty:
        raise ValueError(f"❌ 无匹配军工股数据（代码：{list(config['military_stocks'].keys())}）")

    # 筛选时间范围（文档任务一“给定交易时间段”要求）
    date_field = config["fields"]["date"]
    df_military[date_field] = pd.to_datetime(df_military[date_field], errors="coerce")
    df_military = df_military[
        (df_military[date_field] >= config["start_date"]) &
        (df_military[date_field] <= config["end_date"]) &
        (df_military[date_field].notna())
        ]
    if df_military.empty:
        raise ValueError(f"❌ {config['start_date']}-{config['end_date']}无军工股数据")

    # 计算对数收益率（文档“时间序列联动”核心指标）
    cp_field = config["fields"]["close_price"]
    pivot_cp = df_military.pivot(index=date_field, columns=sc_field, values=cp_field).dropna()
    pivot_return = pivot_cp.apply(lambda x: np.log(x / x.shift(1))).dropna()

    # 转换为长格式并映射股票名称
    df_return = pivot_return.reset_index().melt(
        id_vars=date_field,
        var_name=sc_field,
        value_name="daily_return"
    )
    df_return["stock_name"] = df_return[sc_field].map(config["military_stocks"])
    print(f"✅ 预处理完成：{len(pivot_return)}个交易日，{len(config['military_stocks'])}只军工股")
    return df_return, pivot_return  # 返回长格式+矩阵格式，供后续计算


# 5. 核心计算函数：关联系数+单支股票平均相关系数排名（新增平均排名逻辑）
def calculate_correlations(df_return: pd.DataFrame, pivot_return: pd.DataFrame, config: Dict) -> tuple:
    # 5.1 计算两两股票关联系数矩阵（文档任务一核心要求）
    corr_matrix = pivot_return.corr()
    print(f"✅ 计算关联系数矩阵：{corr_matrix.shape[0]}×{corr_matrix.shape[1]}")

    # 5.2 两两股票关联系数排名（复用原有逻辑，排名从1开始）
    sc_field = config["fields"]["stock_code"]
    corr_long = corr_matrix.reset_index().melt(
        id_vars=sc_field,
        var_name=f"{sc_field}_2",
        value_name="correlation"
    )
    corr_long = corr_long[corr_long[sc_field] != corr_long[f"{sc_field}_2"]].copy()
    corr_long["stock_name_1"] = corr_long[sc_field].map(config["military_stocks"])
    corr_long["stock_name_2"] = corr_long[f"{sc_field}_2"].map(config["military_stocks"])
    pair_ranking = corr_long.sort_values("correlation", ascending=False).reset_index(drop=True)
    pair_ranking["rank"] = pair_ranking.index + 1
    pair_ranking = pair_ranking[["rank", "stock_name_1", sc_field, "stock_name_2", f"{sc_field}_2", "correlation"]]
    pair_ranking["correlation"] = pair_ranking["correlation"].round(4)

    # 5.3 新增：单支股票平均相关系数排名（文档任务一深度分析补充）
    avg_corr_list = []
    for stock_code in corr_matrix.columns:
        # 排除自身相关性，计算与其他所有股票的平均相关系数
        other_corrs = corr_matrix.loc[stock_code, corr_matrix.columns != stock_code]
        avg_corr = other_corrs.mean()
        avg_corr_list.append({
            "stock_code": stock_code,
            "stock_name": config["military_stocks"][stock_code],
            "avg_correlation": round(avg_corr, 4),
            "corr_count": len(other_corrs)  # 参与计算的股票数量（验证完整性）
        })
    # 平均相关系数降序排名（排名从1开始）
    avg_ranking = pd.DataFrame(avg_corr_list).sort_values("avg_correlation", ascending=False).reset_index(drop=True)
    avg_ranking["rank"] = avg_ranking.index + 1
    avg_ranking = avg_ranking[["rank", "stock_name", "stock_code", "avg_correlation", "corr_count"]]

    return pair_ranking, avg_ranking, corr_matrix


# 6. 可视化函数（修正seaborn FutureWarning）
def visualize_results(pair_ranking: pd.DataFrame, avg_ranking: pd.DataFrame, corr_matrix: pd.DataFrame,
                      config: Dict) -> None:
    # 配置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 6.1 平均相关系数条形图（修正警告：将x赋值给hue，关闭图例）
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=avg_ranking,
        x="stock_name",  # x轴：股票名称
        y="avg_correlation",  # y轴：平均相关系数
        hue="stock_name",  # 新增：将x变量赋值给hue（解决警告）
        palette="RdYlBu_r",  # 保持原有颜色方案
        legend=False  # 新增：关闭图例（避免重复显示）
    )
    plt.title(f"{config['start_date']}-{config['end_date']}军工股平均关联系数排名", fontsize=12)
    plt.xlabel("股票名称")
    plt.ylabel("平均相关系数")
    plt.xticks(rotation=45, ha="right")
    # 添加数值标签
    for i, v in enumerate(avg_ranking["avg_correlation"]):
        plt.text(i, v + 0.01, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    avg_plot_path = f"./军工股平均关联系数排名_{config['start_date']}_{config['end_date']}.png"
    plt.savefig(avg_plot_path, dpi=300)
    plt.close()
    print(f"✅ 平均排名图表保存：{avg_plot_path}")

    # 6.2 关联系数热力图（保持不变）
    plt.figure(figsize=(12, 10))
    stock_names = [config["military_stocks"][code] for code in corr_matrix.columns]
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdYlBu_r",
        vmin=-1, vmax=1,
        fmt=".2f",
        xticklabels=stock_names,
        yticklabels=stock_names
    )
    plt.title(f"{config['start_date']}-{config['end_date']}军工股关联系数热力图", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    heatmap_path = f"./军工股关联系数热力图_{config['start_date']}_{config['end_date']}.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✅ 热力图保存：{heatmap_path}")


# 7. 主函数（整合文档任务一全流程）
def main():
    try:
        # 步骤1：读取文档数据
        df_raw = load_stock_data(CONFIG)

        # 步骤2：数据预处理
        df_return, pivot_return = preprocess_data(df_raw, CONFIG)

        # 步骤3：计算关联系数+平均相关系数排名
        pair_ranking, avg_ranking, corr_matrix = calculate_correlations(df_return, pivot_return, CONFIG)

        # 步骤4：输出结果（符合文档“提交内容”要求）
        # 4.1 控制台打印关键结果
        print("\n" + "=" * 80)
        print(f"1. 两两股票关联系数排名（前10名）")
        print("=" * 80)
        print(pair_ranking.head(10).to_string(index=False))

        print("\n" + "=" * 60)
        print(f"2. 单支股票平均关联系数排名（全量）")
        print("=" * 60)
        print(avg_ranking.to_string(index=False))

        # 4.2 保存结果到Excel（文档“可运行代码+结果”提交要求）
        result_path = f"./军工股板块联动分析结果_{CONFIG['start_date']}_{CONFIG['end_date']}.xlsx"
        with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
            pair_ranking.to_excel(writer, sheet_name="两两股票关联系数排名", index=False)
            avg_ranking.to_excel(writer, sheet_name="单支股票平均关联系数排名", index=False)
            corr_matrix.to_excel(writer, sheet_name="关联系数矩阵")
        print(f"\n✅ 结果文件保存：{result_path}")

        # 步骤5：可视化结果
        visualize_results(pair_ranking, avg_ranking, corr_matrix, CONFIG)

        print("\n🎉 文档任务一执行完成！核心输出：")
        print(f"1. 分析结果：{result_path}")
        print(f"2. 平均排名图表：军工股平均关联系数排名_{CONFIG['start_date']}_{CONFIG['end_date']}.png")
        print(f"3. 关联系数热力图：军工股关联系数热力图_{CONFIG['start_date']}_{CONFIG['end_date']}.png")

    except Exception as e:
        print(f"\n❌ 任务一执行失败：{str(e)}")


if __name__ == "__main__":
    main()