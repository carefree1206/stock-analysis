import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import re
from datetime import datetime, timedelta


# ========== 基础数据处理函数 ==========
def standardize_tushare_columns(df):
    """标准化Tushare格式列名"""
    column_mapping = {
        'ts_code': 'code',
        'trade_date': 'date',
        'close': 'close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'vol': 'volume',
        'amount': 'amount',
        'pct_chg': 'pct_change'
    }

    existing_columns = df.columns
    mapping_to_apply = {k: v for k, v in column_mapping.items() if k in existing_columns}
    df = df.rename(columns=mapping_to_apply)
    return df


def extract_stock_code_from_filename(file_path):
    """从文件名提取股票代码"""
    filename = os.path.basename(file_path)
    patterns = [
        r'([0123569]\d{5}\.[Ss][ZzHh])',
        r'(\d{6}\.[Ss][ZzHh])',
        r'(\d{6})'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            code = match.group(1).upper()
            if '.' not in code:
                if code.startswith(('6', '5', '9')):
                    return code + '.SH'
                else:
                    return code + '.SZ'
            return code
    return os.path.splitext(filename)[0]


def merge_tushare_stock_files(folder_path, file_pattern="*.csv"):
    """合并Tushare格式的股票数据文件"""
    print("开始合并股票数据文件...")

    if os.path.isdir(folder_path):
        file_paths = glob.glob(os.path.join(folder_path, file_pattern))
        file_paths.extend(glob.glob(os.path.join(folder_path, "*.xlsx")))
        file_paths.extend(glob.glob(os.path.join(folder_path, "*.xls")))
    else:
        file_paths = [folder_path]

    print(f"找到 {len(file_paths)} 个数据文件")

    all_data = []
    successful_files = 0

    for file_path in tqdm(file_paths, desc="读取文件"):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                continue

            df = standardize_tushare_columns(df)

            if 'code' not in df.columns:
                stock_code = extract_stock_code_from_filename(file_path)
                df['code'] = stock_code
            else:
                df['code'] = df['code'].astype(str)

            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                except:
                    df['date'] = pd.to_datetime(df['date'])
            else:
                continue

            if 'close' not in df.columns:
                continue

            available_cols = ['date', 'code', 'close']
            optional_cols = ['open', 'high', 'low', 'volume', 'amount', 'pct_change']
            for col in optional_cols:
                if col in df.columns:
                    available_cols.append(col)

            df = df[available_cols].copy()
            df = df.sort_values('date')
            all_data.append(df)
            successful_files += 1

        except Exception as e:
            print(f"读取文件 {os.path.basename(file_path)} 时出错: {e}")
            continue

    print(f"成功读取 {successful_files}/{len(file_paths)} 个文件")

    if not all_data:
        raise ValueError("没有成功读取任何数据文件")

    merged_data = pd.concat(all_data, ignore_index=True)
    print(f"合并后数据形状: {merged_data.shape}")
    print(f"包含股票数量: {merged_data['code'].nunique()}")
    print(f"日期范围: {merged_data['date'].min()} 至 {merged_data['date'].max()}")

    return merged_data


def optimize_data_period(merged_data, min_date=None, max_date=None, years=2):
    """优化数据期间"""
    print("优化数据期间...")

    if max_date is None:
        max_date = merged_data['date'].max()
    if min_date is None:
        min_date = max_date - timedelta(days=365 * years)

    filtered_data = merged_data[
        (merged_data['date'] >= min_date) &
        (merged_data['date'] <= max_date)
        ].copy()

    print(f"优化后数据期间: {filtered_data['date'].min()} 至 {filtered_data['date'].max()}")
    print(f"优化后交易日数: {filtered_data['date'].nunique()}")
    print(f"优化后数据量: {len(filtered_data)} 行")

    return filtered_data


def filter_high_quality_stocks(merged_data, min_trading_days_ratio=0.7):
    """筛选高质量股票"""
    print("筛选高质量股票数据...")

    total_trading_days = merged_data['date'].nunique()
    min_required_days = int(total_trading_days * min_trading_days_ratio)

    print(f"总交易日数: {total_trading_days}")
    print(f"要求最小交易日数: {min_required_days}")

    stock_trading_days = merged_data.groupby('code').size()
    qualified_stocks = stock_trading_days[stock_trading_days >= min_required_days].index
    filtered_data = merged_data[merged_data['code'].isin(qualified_stocks)].copy()

    print(f"原始股票数量: {merged_data['code'].nunique()}")
    print(f"筛选后股票数量: {filtered_data['code'].nunique()}")

    return filtered_data


def efficient_wide_format_creation(merged_data):
    """高效创建宽表格式"""
    print("创建优化后的宽表格式...")

    quality_data = filter_high_quality_stocks(merged_data)
    wide_data = quality_data.pivot_table(index='date', columns='code', values='close')

    print(f"宽表初始形状: {wide_data.shape}")

    wide_data_filled = wide_data.ffill(limit=5).bfill(limit=5)
    missing_ratio = wide_data_filled.isnull().sum() / len(wide_data_filled)
    valid_stocks = missing_ratio[missing_ratio < 0.1].index
    wide_data_clean = wide_data_filled[valid_stocks]

    print(f"最终宽表形状: {wide_data_clean.shape}")
    print(f"最终股票数量: {len(wide_data_clean.columns)}")
    print(f"最终交易日数: {len(wide_data_clean)}")

    return wide_data_clean


def calculate_returns_safe(wide_data):
    """安全计算收益率，避免警告"""
    print("计算股票收益率...")

    data_clean = wide_data.ffill().bfill()
    returns_data = data_clean.pct_change(fill_method=None).dropna()

    print(f"收益率数据形状: {returns_data.shape}")

    # 处理极端值
    returns_clean = returns_data.clip(
        returns_data.quantile(0.01),
        returns_data.quantile(0.99),
        axis=1
    )

    return returns_clean


def calculate_correlation_matrix(returns_data):
    """计算相关性矩阵"""
    print("计算股票相关性...")
    correlation_matrix = returns_data.corr(method='spearman')
    print(f"相关性矩阵计算完成，形状: {correlation_matrix.shape}")

    # 显示相关性统计
    corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    print(f"相关性统计:")
    print(f"  均值: {corr_values.mean():.3f}")
    print(f"  中位数: {np.median(corr_values):.3f}")
    print(f"  标准差: {corr_values.std():.3f}")
    print(f"  最小值: {corr_values.min():.3f}")
    print(f"  最大值: {corr_values.max():.3f}")
    print(f"  25%分位数: {np.percentile(corr_values, 25):.3f}")
    print(f"  75%分位数: {np.percentile(corr_values, 75):.3f}")

    return correlation_matrix


def main_efficient_analysis(data_folder_path, years=2):
    """高效的主分析函数"""
    print(f"=== 开始高效股票分析（使用最近{years}年数据）===")

    merged_data = merge_tushare_stock_files(data_folder_path)
    optimized_data = optimize_data_period(merged_data, years=years)
    wide_data = efficient_wide_format_creation(optimized_data)
    returns_data = calculate_returns_safe(wide_data)
    correlation_matrix = calculate_correlation_matrix(returns_data)

    print("\n=== 数据准备完成 ===")
    print(f"最终数据: {wide_data.shape[1]} 只股票, {wide_data.shape[0]} 个交易日")
    print(f"日期范围: {wide_data.index.min()} 至 {wide_data.index.max()}")

    return wide_data, correlation_matrix, returns_data


# ========== 多重板块分析类 ==========
class ImprovedMultiSectorAnalyzer:
    def __init__(self, min_stocks_per_sector=8, max_stocks_per_sector=30,
                 correlation_threshold=0.4, sector_count_target=8):
        self.min_stocks_per_sector = min_stocks_per_sector
        self.max_stocks_per_sector = max_stocks_per_sector
        self.correlation_threshold = correlation_threshold
        self.sector_count_target = sector_count_target
        self.sectors = {}
        self.belonging_factors = {}

    def detect_sectors_intelligently(self, correlation_matrix):
        """
        智能板块检测：避免所有股票归入同一板块
        """
        print("使用智能板块检测算法...")

        # 分析相关性分布，动态调整阈值
        optimal_threshold = self._find_optimal_threshold(correlation_matrix)
        print(f"动态调整相关性阈值: {optimal_threshold:.3f}")

        # 使用优化后的阈值进行板块发现
        sectors = self._hierarchical_sector_detection(correlation_matrix, optimal_threshold)

        self.sectors = sectors
        print(f"发现 {len(self.sectors)} 个有效板块")

        if self.sectors:
            self._analyze_sector_statistics()
        else:
            print("警告：未发现任何有效板块")

        return self.sectors

    def _find_optimal_threshold(self, correlation_matrix):
        """
        根据相关性分布找到最优阈值
        """
        corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]

        # 分析相关性分布
        percentiles = [60, 70, 75, 80, 85]
        best_threshold = 0.4  # 默认值

        for percentile in percentiles:
            threshold = np.percentile(corr_values, percentile)
            # 测试这个阈值能产生多少有效的板块
            test_sectors = self._test_threshold(correlation_matrix, threshold)
            if 3 <= len(test_sectors) <= 15:  # 理想的板块数量范围
                best_threshold = threshold
                print(f"  尝试{percentile}%分位数({threshold:.3f}) -> {len(test_sectors)}个板块")
                break
            else:
                print(f"  尝试{percentile}%分位数({threshold:.3f}) -> {len(test_sectors)}个板块 (不合适)")

        return best_threshold

    def _test_threshold(self, correlation_matrix, threshold):
        """测试特定阈值下的板块效果"""
        sectors = {}
        stock_codes = correlation_matrix.columns.tolist()
        used_stocks = set()
        sector_id = 0

        for stock in stock_codes:
            if stock in used_stocks:
                continue

            # 找到高度相关的股票
            high_corr_stocks = [
                s for s in stock_codes
                if correlation_matrix.loc[stock, s] > threshold and s not in used_stocks
            ]

            if len(high_corr_stocks) >= self.min_stocks_per_sector:
                # 限制板块规模
                selected_stocks = high_corr_stocks[:self.max_stocks_per_sector]
                sectors[f"sector_{sector_id}"] = [stock] + selected_stocks
                used_stocks.update([stock] + selected_stocks)
                sector_id += 1

        return sectors

    def _hierarchical_sector_detection(self, correlation_matrix, threshold):
        """
        分层板块检测：先找高相关性的核心板块
        """
        print("使用分层板块检测...")

        sectors = {}
        stock_codes = correlation_matrix.columns.tolist()
        used_stocks = set()
        sector_id = 0

        # 第一轮：寻找高相关性核心板块
        print("第一轮：寻找核心板块...")
        for stock in stock_codes:
            if stock in used_stocks:
                continue

            # 找到高度相关的股票（要求更高的相关性）
            high_corr_stocks = [
                s for s in stock_codes
                if (correlation_matrix.loc[stock, s] > threshold + 0.1)  # 更高的阈值
                   and s not in used_stocks
            ]

            if len(high_corr_stocks) >= self.min_stocks_per_sector:
                selected_stocks = high_corr_stocks[:self.max_stocks_per_sector]
                sectors[f"core_sector_{sector_id}"] = [stock] + selected_stocks
                used_stocks.update([stock] + selected_stocks)
                sector_id += 1
                print(f"  发现核心板块 {sector_id}: {len(selected_stocks) + 1} 只股票")

        # 第二轮：用较低阈值寻找次级板块
        print("第二轮：寻找次级板块...")
        secondary_threshold = threshold - 0.05
        for stock in stock_codes:
            if stock in used_stocks:
                continue

            high_corr_stocks = [
                s for s in stock_codes
                if correlation_matrix.loc[stock, s] > secondary_threshold
                   and s not in used_stocks
            ]

            if len(high_corr_stocks) >= self.min_stocks_per_sector:
                selected_stocks = high_corr_stocks[:self.max_stocks_per_sector]
                sectors[f"secondary_sector_{sector_id}"] = [stock] + selected_stocks
                used_stocks.update([stock] + selected_stocks)
                sector_id += 1
                print(f"  发现次级板块 {sector_id}: {len(selected_stocks) + 1} 只股票")

        # 第三轮：处理剩余股票
        print("第三轮：处理剩余股票...")
        remaining_stocks = [s for s in stock_codes if s not in used_stocks]
        if len(remaining_stocks) >= self.min_stocks_per_sector:
            # 将剩余股票分组
            for i in range(0, len(remaining_stocks), self.max_stocks_per_sector):
                group = remaining_stocks[i:i + self.max_stocks_per_sector]
                if len(group) >= self.min_stocks_per_sector:
                    sectors[f"residual_sector_{sector_id}"] = group
                    sector_id += 1
                    print(f"  创建剩余股票板块 {sector_id}: {len(group)} 只股票")

        return sectors

    def _analyze_sector_statistics(self):
        """分析板块统计"""
        print(f"\n板块统计:")
        print(f"- 总板块数: {len(self.sectors)}")

        sector_sizes = [len(stocks) for stocks in self.sectors.values()]
        print(f"- 板块规模: 最小{min(sector_sizes)}, 最大{max(sector_sizes)}, 平均{np.mean(sector_sizes):.1f}")

        total_stocks = sum(sector_sizes)
        unique_stocks = len(set(stock for stocks in self.sectors.values() for stock in stocks))
        print(f"- 总归属数: {total_stocks}, 唯一股票数: {unique_stocks}")

        if unique_stocks > 0:
            overlap_ratio = (total_stocks - unique_stocks) / unique_stocks
            print(f"- 平均重叠度: {overlap_ratio:.2f} (每只股票平均属于{total_stocks / unique_stocks:.2f}个板块)")

    def calculate_belonging_factors(self, returns_data):
        """计算归属因数"""
        if not self.sectors:
            return {}

        print("计算板块归属因数...")

        self.belonging_factors = {}

        # 为每个板块计算基准
        sector_benchmarks = {}
        for sector_id, stocks in self.sectors.items():
            try:
                sector_returns = returns_data[stocks].mean(axis=1)
                sector_benchmarks[sector_id] = sector_returns
            except:
                continue

        # 计算每只股票的归属因数
        for sector_id, stocks in self.sectors.items():
            if sector_id not in sector_benchmarks:
                continue

            self.belonging_factors[sector_id] = {}
            sector_benchmark = sector_benchmarks[sector_id]

            for stock in stocks:
                try:
                    stock_returns = returns_data[stock]
                    correlation = np.corrcoef(stock_returns, sector_benchmark)[0, 1]

                    if np.isnan(correlation):
                        factor = 0.5
                    else:
                        factor = 0.3 + 0.6 * (correlation + 1) / 2
                        factor = min(0.9, max(0.3, factor))
                except:
                    factor = 0.5

                self.belonging_factors[sector_id][stock] = factor

        return self.belonging_factors

    def generate_report(self):
        """生成分析报告"""
        if not self.sectors:
            return

        print("\n" + "=" * 60)
        print("改进的板块分析报告")
        print("=" * 60)

        # 按板块质量排序
        sector_quality = []
        for sector_id, factors in self.belonging_factors.items():
            if factors:
                avg_factor = np.mean(list(factors.values()))
                stock_count = len(factors)
                sector_quality.append((sector_id, avg_factor, stock_count))

        sector_quality.sort(key=lambda x: x[1], reverse=True)

        print(f"\n板块质量排名:")
        for i, (sector_id, avg_factor, stock_count) in enumerate(sector_quality[:10]):
            quality = "优秀" if avg_factor > 0.7 else "良好" if avg_factor > 0.6 else "一般"
            print(f"{i + 1:2d}. {sector_id:20s} 股票数:{stock_count:2d} 平均归属因数:{avg_factor:.3f} [{quality}]")

        # 显示核心股票
        print(f"\n各板块核心股票:")
        for sector_id, avg_factor, stock_count in sector_quality[:5]:
            factors = self.belonging_factors[sector_id]
            top_stocks = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"{sector_id}: {', '.join([f'{stock}({factor:.3f})' for stock, factor in top_stocks])}")


# ========== 主执行函数 ==========
def main_task2_analysis():
    """
    主分析函数
    """
    # ========== 运行参数设置 ==========
    data_folder = r"D:\stock data"  # 修改为你的数据文件夹路径

    # 更合理的参数设置
    ANALYSIS_YEARS = 1
    MIN_STOCKS_PER_SECTOR = 8  # 合理的板块最小规模
    MAX_STOCKS_PER_SECTOR = 25  # 限制板块最大规模
    CORRELATION_THRESHOLD = 0.4  # 较高的相关性阈值
    SECTOR_COUNT_TARGET = 10  # 目标板块数量

    print("=== 板块分析系统 ===")
    print(f"运行参数:")
    print(f"- 分析年限: {ANALYSIS_YEARS} 年")
    print(f"- 板块规模: {MIN_STOCKS_PER_SECTOR}-{MAX_STOCKS_PER_SECTOR} 只股票")
    print(f"- 相关性阈值: {CORRELATION_THRESHOLD}")
    print(f"- 目标板块数: {SECTOR_COUNT_TARGET}")

    try:
        # 1. 数据准备
        print("\n步骤1: 数据准备...")
        wide_data, corr_matrix, returns_data = main_efficient_analysis(
            data_folder,
            years=ANALYSIS_YEARS
        )

        # 2. 智能板块分析
        print("\n步骤2: 智能板块分析...")
        analyzer = ImprovedMultiSectorAnalyzer(
            min_stocks_per_sector=MIN_STOCKS_PER_SECTOR,
            max_stocks_per_sector=MAX_STOCKS_PER_SECTOR,
            correlation_threshold=CORRELATION_THRESHOLD,
            sector_count_target=SECTOR_COUNT_TARGET
        )

        sectors = analyzer.detect_sectors_intelligently(corr_matrix)

        if not sectors:
            print("未发现有效板块，分析结束")
            return

        # 3. 归属因数计算
        print("\n步骤3: 计算归属因数...")
        belonging_factors = analyzer.calculate_belonging_factors(returns_data)

        # 4. 生成报告
        print("\n步骤4: 生成报告...")
        analyzer.generate_report()

        print("\n✅ 分析完成！")

        # 保存结果
        save_results(analyzer)
        return analyzer, returns_data

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_results(analyzer):
    """保存结果"""
    if not analyzer.sectors:
        return

    # 保存板块信息
    sector_data = []
    for sector_id, stocks in analyzer.sectors.items():
        for stock in stocks:
            factor = analyzer.belonging_factors.get(sector_id, {}).get(stock, 0.5)
            sector_data.append({
                'sector_id': sector_id,
                'stock_code': stock,
                'belonging_factor': factor
            })

    sector_df = pd.DataFrame(sector_data)
    sector_df.to_csv('improved_sector_results.csv', index=False, encoding='utf-8-sig')
    print("结果已保存到 'improved_sector_results.csv'")
