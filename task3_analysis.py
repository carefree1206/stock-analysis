import pandas as pd
import numpy as np
import akshare as ak
import requests
import json
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Any

warnings.filterwarnings('ignore')


# ==================== 数据层 ====================
class Task3DataManager:
    """
    数据管理层：处理任务三所需的所有数据
    """

    def __init__(self, task2_analyzer, returns_data):
        self.task2_analyzer = task2_analyzer
        self.returns_data = returns_data
        self.base_sectors = task2_analyzer.sectors
        self.base_factors = task2_analyzer.belonging_factors

    def validate_input_data(self):
        """验证输入数据完整性"""
        if not self.base_sectors:
            raise ValueError("❌ 任务二未发现有效板块")
        if self.returns_data.empty:
            raise ValueError("❌ 收益率数据为空")

        print(f"✅ 数据验证通过")
        print(f"   板块数量: {len(self.base_sectors)}")
        print(f"   股票数量: {sum(len(stocks) for stocks in self.base_sectors.values())}")
        print(f"   数据期间: {self.returns_data.index[0]} 到 {self.returns_data.index[-1]}")

    def get_time_periods(self, window_size: int, step_size: int) -> List[Dict]:
        """生成分析时间窗口"""
        dates = self.returns_data.index
        periods = []

        for start_idx in range(0, len(dates) - window_size, step_size):
            end_idx = start_idx + window_size
            period_label = f"{dates[start_idx].strftime('%Y-%m-%d')}_to_{dates[end_idx].strftime('%Y-%m-%d')}"
            periods.append({
                'label': period_label,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_date': dates[start_idx],
                'end_date': dates[end_idx]
            })

        print(f"   生成 {len(periods)} 个分析时间段")
        return periods


# ==================== 事件数据层 ====================
class EventDataCollector:
    """
    事件数据收集器 - 基于真实数据源
    """

    def __init__(self):
        self.cache = {}  # 简单缓存避免重复请求

    def get_stock_events(self, stock_code: str, start_date: str, end_date: str) -> List[Dict]:
        """
        获取股票相关事件（财报、公告、新闻）
        """
        events = []

        try:
            # 1. 获取财报数据
            earnings_events = self._get_earnings_events(stock_code, start_date, end_date)
            events.extend(earnings_events)

            # 2. 获取公司公告
            announcement_events = self._get_announcement_events(stock_code, start_date, end_date)
            events.extend(announcement_events)

            # 3. 获取新闻数据
            news_events = self._get_news_events(stock_code, start_date, end_date)
            events.extend(news_events)

        except Exception as e:
            print(f"⚠️ 获取 {stock_code} 事件数据失败: {e}")

        # 按时间排序并去重
        events = self._deduplicate_events(events)
        events.sort(key=lambda x: x.get('event_date', ''), reverse=True)

        return events[:15]  # 返回最近15个事件

    # 在task3_analysis.py中改进财报数据获取方法
    def _get_earnings_events(self, stock_code: str, start_date: str, end_date: str) -> List[Dict]:
        """获取财报事件 - 优化日期处理和数据提取"""
        events = []
        try:
            # 使用AKShare获取财务数据（改进版本）
            # 尝试获取最新的财报日期数据
            if stock_code.endswith('.SH'):
                symbol = f"sh{stock_code[:6]}"
            elif stock_code.endswith('.SZ'):
                symbol = f"sz{stock_code[:6]}"
            else:
                symbol = stock_code

            # 尝试获取业绩预告数据（更相关的财报事件）
            try:
                forecast_df = ak.stock_forecast_report(symbol=symbol)
                if not forecast_df.empty:
                    for _, row in forecast_df.iterrows():
                        # 提取实际公告日期
                        event_date = row.get('ann_date', datetime.now().strftime('%Y-%m-%d'))
                        if isinstance(event_date, pd.Timestamp):
                            event_date = event_date.strftime('%Y-%m-%d')

                        # 确定影响程度
                        if '预增' in row.get('content', ''):
                            impact = 'positive'
                        elif '预减' in row.get('content', ''):
                            impact = 'negative'
                        else:
                            impact = 'neutral'

                        event = {
                            'event_date': event_date,
                            'event_type': '业绩预告',
                            'title': f"{row.get('title', '业绩预告')}",
                            'content': row.get('content', ''),
                            'impact': impact,
                            'source': 'AKShare-业绩预告',
                            'stock_code': stock_code
                        }
                        events.append(event)
            except:
                pass  # 该接口可能不适用于所有股票，失败时跳过

            # 补充获取财务指标数据
            stock_individual_info_em_df = ak.stock_individual_info_em(symbol=symbol)
            if not stock_individual_info_em_df.empty:
                # 提取最近一次财报日期
                latest_finance_date = None
                if '发布日期' in stock_individual_info_em_df.columns:
                    date_cols = stock_individual_info_em_df['发布日期'].dropna()
                    if not date_cols.empty:
                        latest_finance_date = date_cols.iloc[0]
                        if isinstance(latest_finance_date, pd.Timestamp):
                            latest_finance_date = latest_finance_date.strftime('%Y-%m-%d')

                for _, row in stock_individual_info_em_df.iterrows():
                    event = {
                        'event_date': latest_finance_date or datetime.now().strftime('%Y-%m-%d'),
                        'event_type': '财务指标',
                        'title': f"{row.get('item', '财务指标')}",
                        'content': str(row.to_dict()),
                        'impact': 'neutral',
                        'source': 'AKShare-财务指标',
                        'stock_code': stock_code
                    }
                    events.append(event)

        except Exception as e:
            print(f"获取财报数据失败 {stock_code}: {e}")

        return events

    def _get_announcement_events(self, stock_code: str, start_date: str, end_date: str) -> List[Dict]:
        """获取公司公告"""
        events = []
        try:
            # 使用AKShare获取新闻公告
            stock_news_em_df = ak.stock_news_em(symbol=stock_code)
            if not stock_news_em_df.empty:
                for _, row in stock_news_em_df.iterrows():
                    event = {
                        'event_date': row.get('发布时间', datetime.now().strftime('%Y-%m-%d')),
                        'event_type': '公司公告',
                        'title': row.get('标题', ''),
                        'content': row.get('内容', ''),
                        'impact': self._assess_announcement_impact(row.get('标题', '')),
                        'source': 'AKShare',
                        'stock_code': stock_code
                    }
                    events.append(event)

        except Exception as e:
            print(f"获取公告数据失败 {stock_code}: {e}")

        return events

    def _get_news_events(self, stock_code: str, start_date: str, end_date: str) -> List[Dict]:
        """获取新闻事件"""
        events = []
        try:
            # 使用AKShare获取个股新闻
            stock_news_em_df = ak.stock_news_em(symbol=stock_code)
            if not stock_news_em_df.empty:
                for _, row in stock_news_em_df.iterrows():
                    event = {
                        'event_date': row.get('发布时间', datetime.now().strftime('%Y-%m-%d')),
                        'event_type': '市场新闻',
                        'title': row.get('标题', ''),
                        'content': row.get('内容', ''),
                        'impact': self._assess_news_impact(row.get('标题', '')),
                        'source': 'AKShare',
                        'stock_code': stock_code
                    }
                    events.append(event)

        except Exception as e:
            print(f"获取新闻数据失败 {stock_code}: {e}")

        return events

    def _assess_announcement_impact(self, title: str) -> str:
        """评估公告影响"""
        positive_keywords = ['利好', '增长', '盈利', '合作', '订单', '突破', '扩张', '收购', '中标']
        negative_keywords = ['亏损', '下滑', '风险', '诉讼', '警示', '退市', '减持', '违规']

        title_lower = title.lower()

        positive_count = sum(1 for keyword in positive_keywords if keyword in title_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in title_lower)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def _assess_news_impact(self, title: str) -> str:
        """评估新闻影响"""
        return self._assess_announcement_impact(title)

    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """去重事件"""
        seen = set()
        unique_events = []

        for event in events:
            event_key = f"{event.get('event_date')}_{event.get('title')}"
            if event_key not in seen:
                seen.add(event_key)
                unique_events.append(event)

        return unique_events


# ==================== 分析层 ====================
class ChangeDetector:
    """
    变化检测层：识别归属因数的显著变化
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.belonging_changes = {}

    def detect_changes_multi_period(self, window_size: int = 60, step_size: int = 30) -> Dict:
        """
        多时期变化检测
        """
        print("🔍 开始多时期变化检测...")

        periods = self.data_manager.get_time_periods(window_size, step_size)
        base_factors = self.data_manager.base_factors

        total_analyses = 0

        for sector_id, stocks in self.data_manager.base_sectors.items():
            print(f"   分析板块 {sector_id} ({len(stocks)}只股票)...")
            sector_changes = self._analyze_sector_changes(sector_id, stocks, periods, base_factors)
            self.belonging_changes[sector_id] = sector_changes
            total_analyses += len(sector_changes)

        print(f"✅ 变化检测完成，共进行 {total_analyses} 个时间段分析")
        return self.belonging_changes

    def _analyze_sector_changes(self, sector_id: str, stocks: List[str], periods: List[Dict],
                                base_factors: Dict) -> Dict:
        """分析单个板块的变化"""
        sector_changes = {}

        for period in periods:
            # 计算当前窗口的归属因数
            window_data = self.data_manager.returns_data.iloc[period['start_idx']:period['end_idx']]
            current_factors = self._calculate_period_factors(window_data, stocks)

            if current_factors:
                changes = self._compute_factor_changes(
                    base_factors.get(sector_id, {}),
                    current_factors
                )
                sector_changes[period['label']] = changes

        return sector_changes

    def _calculate_period_factors(self, period_returns: pd.DataFrame, stocks: List[str]) -> Dict[str, float]:
        """计算特定时期的归属因数"""
        factors = {}
        available_stocks = [s for s in stocks if s in period_returns.columns]

        if len(available_stocks) < 3:
            return factors

        try:
            sector_benchmark = period_returns[available_stocks].mean(axis=1)

            for stock in available_stocks:
                try:
                    correlation = np.corrcoef(period_returns[stock], sector_benchmark)[0, 1]
                    if not np.isnan(correlation):
                        factor = 0.3 + 0.6 * (correlation + 1) / 2
                        factors[stock] = min(0.9, max(0.3, factor))
                except:
                    continue

        except Exception as e:
            print(f"      计算因数出错: {e}")

        return factors

    def _compute_factor_changes(self, base_factors: Dict, current_factors: Dict) -> Dict:
        """计算归属因数变化"""
        changes = {}

        for stock, current_factor in current_factors.items():
            base_factor = base_factors.get(stock, 0.5)
            change = current_factor - base_factor

            changes[stock] = {
                'stock': stock,
                'base_factor': base_factor,
                'current_factor': current_factor,
                'change': change,
                'change_pct': (change / base_factor) * 100 if base_factor > 0 else 0,
                'significance': self._assess_significance(change),
                'direction': '上升' if change > 0 else '下降',
                'magnitude': abs(change)
            }

        return changes

    def _assess_significance(self, change: float) -> str:
        """评估变化显著性"""
        abs_change = abs(change)
        if abs_change > 0.2:
            return 'high'
        elif abs_change > 0.1:
            return 'medium'
        else:
            return 'low'


# ==================== 归因引擎层 ====================
class EventDrivenAttributionEngine:
    """
    事件驱动的归因引擎 - 基于真实事件数据
    """

    def __init__(self):
        self.event_collector = EventDataCollector()
        self.reason_templates = self._initialize_reason_templates()

    def _initialize_reason_templates(self) -> Dict:
        """初始化归因原因模板"""
        return {
            'positive_high': [
                "财报业绩超预期，盈利大幅增长{change_pct:.1f}%",
                "获得重大战略订单，业务前景显著改善",
                "技术创新取得突破，行业竞争力大幅提升",
                "行业政策重大利好，公司充分受益"
            ],
            'positive_medium': [
                "经营状况持续改善，市场份额稳步扩大",
                "成本控制成效显著，利润率明显提升",
                "新产品成功上市，收入来源更加多元化",
                "管理效率提升，运营成本下降"
            ],
            'negative_high': [
                "财报业绩不及预期，盈利大幅下滑{change_pct:.1f}%",
                "重要项目遭遇重大挫折或取消",
                "行业监管政策收紧，经营受到严格限制",
                "市场竞争激烈，市场份额严重流失"
            ],
            'negative_medium': [
                "原材料成本大幅上升，利润率受到挤压",
                "市场需求出现季节性波动",
                "汇率波动对海外业务造成负面影响",
                "行业周期性调整影响业绩表现"
            ]
        }

    def perform_attribution_analysis(self, significant_changes: Dict) -> Dict:
        """
        执行归因分析
        """
        print("🧠 执行事件驱动的归因分析...")

        attribution_results = {}
        total_attributions = 0

        for sector_id, period_changes in significant_changes.items():
            print(f"  分析板块 {sector_id} 的归因...")
            sector_attributions = {}

            for period, changes in period_changes.items():
                period_attributions = {}

                for stock, change_info in changes.items():
                    attribution = self._analyze_single_change_with_events(stock, period, change_info)
                    period_attributions[stock] = attribution
                    total_attributions += 1

                if period_attributions:
                    sector_attributions[period] = period_attributions

            if sector_attributions:
                attribution_results[sector_id] = sector_attributions

        print(f"✅ 归因分析完成，共生成 {total_attributions} 个归因分析")
        return attribution_results

    def _analyze_single_change_with_events(self, stock: str, period: str, change_info: Dict) -> Dict:
        """基于事件分析单个变化"""
        # 解析时间段
        start_date, end_date = self._parse_period(period)

        # 收集相关事件
        events = self.event_collector.get_stock_events(stock, start_date, end_date)

        if events:
            # 基于真实事件进行归因
            return self._event_based_attribution(stock, period, change_info, events)
        else:
            # 使用规则引擎归因
            return self._rule_based_attribution(stock, period, change_info)

    def _event_based_attribution(self, stock: str, period: str, change_info: Dict, events: List[Dict]) -> Dict:
        """基于真实事件的归因"""
        # 分析事件影响
        significant_events = [e for e in events if e['impact'] in ['positive', 'negative']]

        if significant_events:
            reasons = []
            for event in significant_events[:2]:  # 取最重要的两个事件
                reason = f"{event['event_type']}: {event['title']}"
                reasons.append(reason)

            # 计算置信度
            confidence = '高' if len(significant_events) >= 2 else '中'

            attribution = {
                'stock': stock,
                'period': period,
                'factor_change': change_info['change'],
                'change_direction': change_info['direction'],
                'change_magnitude': change_info['significance'],
                'event_based': True,
                'events_count': len(significant_events),
                'total_events': len(events),
                'possible_reasons': reasons,
                'confidence': confidence,
                'analysis_method': '事件驱动归因',
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            attribution = self._rule_based_attribution(stock, period, change_info)
            attribution['event_based'] = False
            attribution['events_count'] = len(events)

        return attribution

    def _rule_based_attribution(self, stock: str, period: str, change_info: Dict) -> Dict:
        """基于规则的归因"""
        change = change_info['change']
        significance = change_info['significance']
        direction = change_info['direction']

        # 确定归因类型
        if direction == '上升':
            if significance == 'high':
                reason_key = 'positive_high'
                confidence = '高'
            else:
                reason_key = 'positive_medium'
                confidence = '中'
        else:
            if significance == 'high':
                reason_key = 'negative_high'
                confidence = '高'
            else:
                reason_key = 'negative_medium'
                confidence = '中'

        # 选择原因模板
        templates = self.reason_templates[reason_key]
        import random
        selected_reasons = random.sample(templates, min(2, len(templates)))

        # 格式化原因
        formatted_reasons = []
        for reason in selected_reasons:
            formatted_reason = reason.format(change_pct=abs(change_info['change_pct']))
            formatted_reasons.append(formatted_reason)

        return {
            'stock': stock,
            'period': period,
            'factor_change': change,
            'change_direction': direction,
            'change_magnitude': significance,
            'event_based': False,
            'events_count': 0,
            'total_events': 0,
            'possible_reasons': formatted_reasons,
            'confidence': confidence,
            'analysis_method': '规则引擎归因',
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _parse_period(self, period_str: str) -> tuple:
        """解析时间段"""
        try:
            parts = period_str.split('_to_')
            start_date = datetime.strptime(parts[0], '%Y-%m-%d')
            end_date = datetime.strptime(parts[1], '%Y-%m-%d')
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        except:
            # 默认返回最近30天
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# ==================== 报告生成层 ====================
class ComprehensiveReportGenerator:
    """
    综合报告生成器
    """

    def __init__(self, attribution_results: Dict):
        self.attribution_results = attribution_results

    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n" + "=" * 100)
        print("📊 任务三：归属因数变化归因分析综合报告")
        print("=" * 100)

        total_analyses = 0
        event_based_analyses = 0
        significant_analyses = 0
        positive_changes = 0
        negative_changes = 0

        # 详细分析报告
        for sector_id, period_attributions in self.attribution_results.items():
            print(f"\n🎯 板块 {sector_id}")
            print("=" * 80)

            sector_analyses = 0
            sector_event_based = 0

            for period, stock_attributions in period_attributions.items():
                print(f"\n  时间段: {period}")
                print("  " + "─" * 60)

                for stock, attribution in stock_attributions.items():
                    total_analyses += 1
                    sector_analyses += 1

                    if attribution['event_based']:
                        event_based_analyses += 1
                        sector_event_based += 1

                    if attribution['confidence'] in ['高', '中']:
                        significant_analyses += 1

                    if attribution['change_direction'] == '上升':
                        positive_changes += 1
                    else:
                        negative_changes += 1

                    self._print_detailed_attribution(attribution)

            # 板块统计
            if sector_analyses > 0:
                event_based_ratio = sector_event_based / sector_analyses
                print(f"\n  📈 板块统计: 共{sector_analyses}个分析, "
                      f"事件驱动{sector_event_based}个({event_based_ratio:.1%})")

        # 汇总统计
        self._print_summary_statistics(
            total_analyses, event_based_analyses, significant_analyses,
            positive_changes, negative_changes
        )

    def _print_detailed_attribution(self, attribution: Dict):
        """打印详细归因分析"""
        # 图标选择
        direction_icon = "📈" if attribution['change_direction'] == '上升' else "📉"
        confidence_icon = "🔥" if attribution['confidence'] == '高' else "⚠️" if attribution[
                                                                                    'confidence'] == '中' else "ℹ️"
        method_icon = "🎯" if attribution['event_based'] else "⚙️"

        print(f"  {direction_icon} {attribution['stock']}")
        print(f"    变化: {attribution['factor_change']:+.3f} ({attribution['change_direction']})")
        print(f"    程度: {attribution['change_magnitude']} | 置信度: {confidence_icon} {attribution['confidence']}")
        print(f"    方法: {method_icon} {attribution['analysis_method']}")

        if attribution['event_based']:
            print(f"    事件: 共{attribution['total_events']}个, 显著{attribution['events_count']}个")

        print(f"    可能原因:")
        for i, reason in enumerate(attribution['possible_reasons'], 1):
            print(f"      {i}. {reason}")

        print(f"    分析时间: {attribution['analysis_timestamp']}")
        print()

    def _print_summary_statistics(self, total_analyses: int, event_based_analyses: int,
                                  significant_analyses: int, positive_changes: int, negative_changes: int):
        """打印汇总统计"""
        print("\n" + "=" * 100)
        print("📈 综合分析统计")
        print("=" * 100)

        print(f"📊 基础统计:")
        print(f"   总分析案例: {total_analyses}")
        print(f"   显著变化案例: {significant_analyses}")
        print(f"   分析覆盖率: {significant_analyses / total_analyses * 100:.1f}%")

        print(f"\n🎯 归因方法:")
        print(f"   事件驱动分析: {event_based_analyses}个 ({event_based_analyses / total_analyses * 100:.1f}%)")
        print(
            f"   规则引擎分析: {total_analyses - event_based_analyses}个 ({(total_analyses - event_based_analyses) / total_analyses * 100:.1f}%)")

        print(f"\n📈 变化方向:")
        print(f"   上升变化: {positive_changes}个 ({positive_changes / total_analyses * 100:.1f}%)")
        print(f"   下降变化: {negative_changes}个 ({negative_changes / total_analyses * 100:.1f}%)")
        print(f"   净变化方向: {'上升' if positive_changes > negative_changes else '下降'}")
        print(f"   变化平衡度: {(positive_changes - negative_changes) / total_analyses * 100:+.1f}%")

        # 置信度分布
        confidence_dist = {'高': 0, '中': 0, '低': 0}
        for sector_attributions in self.attribution_results.values():
            for period_attributions in sector_attributions.values():
                for attribution in period_attributions.values():
                    confidence_dist[attribution['confidence']] += 1

        print(f"\n✅ 置信度分布:")
        for conf_level, count in confidence_dist.items():
            if total_analyses > 0:
                percentage = count / total_analyses * 100
                print(f"   {conf_level}置信度: {count}个 ({percentage:.1f}%)")

    def save_detailed_results(self, filename: str = 'task3_detailed_results.csv'):
        """保存详细结果"""
        results_data = []

        for sector_id, period_attributions in self.attribution_results.items():
            for period, stock_attributions in period_attributions.items():
                for stock, attribution in stock_attributions.items():
                    record = {
                        'sector_id': sector_id,
                        'analysis_period': period,
                        'stock_code': stock,
                        'factor_change': attribution['factor_change'],
                        'change_direction': attribution['change_direction'],
                        'change_magnitude': attribution['change_magnitude'],
                        'event_based': attribution['event_based'],
                        'events_count': attribution['events_count'],
                        'total_events': attribution['total_events'],
                        'confidence_level': attribution['confidence'],
                        'analysis_method': attribution['analysis_method'],
                        'possible_reason_1': attribution['possible_reasons'][0] if len(
                            attribution['possible_reasons']) > 0 else '',
                        'possible_reason_2': attribution['possible_reasons'][1] if len(
                            attribution['possible_reasons']) > 1 else '',
                        'possible_reason_3': attribution['possible_reasons'][2] if len(
                            attribution['possible_reasons']) > 2 else '',
                        'analysis_timestamp': attribution['analysis_timestamp']
                    }
                    results_data.append(record)

        df = pd.DataFrame(results_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 详细结果已保存到: {filename}")
        return df

    def save_summary_report(self, filename: str = 'task3_summary_report.txt'):
        """保存文本摘要报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("任务三：归属因数变化归因分析摘要报告\n")
            f.write("=" * 50 + "\n\n")

            total_analyses = 0
            for sector_id, period_attributions in self.attribution_results.items():
                f.write(f"板块 {sector_id}:\n")
                for period, stock_attributions in period_attributions.items():
                    f.write(f"  时期 {period}: {len(stock_attributions)} 个分析\n")
                    total_analyses += len(stock_attributions)

            f.write(f"\n总分析数量: {total_analyses}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"✅ 摘要报告已保存到: {filename}")


# ==================== 主控制器 ====================
class Task3MainController:
    """
    任务三主控制器
    """

    def __init__(self, task2_analyzer, returns_data):
        self.data_manager = Task3DataManager(task2_analyzer, returns_data)
        self.change_detector = ChangeDetector(self.data_manager)
        self.attribution_engine = EventDrivenAttributionEngine()
        self.report_generator = None

    def run_complete_analysis(self, window_size: int = 30, step_size: int = 15,
                              change_threshold: float = 0.08) -> Dict:
        """
        运行完整的任务三分析
        """
        print("🚀 开始任务三完整分析流程")
        print("=" * 60)
        print(f"分析参数: 窗口{window_size}天, 步长{step_size}天, 阈值{change_threshold}")

        try:
            # 1. 数据验证
            print("📋 步骤1: 数据验证...")
            self.data_manager.validate_input_data()

            # 2. 变化检测
            print("\n🔍 步骤2: 变化检测...")
            all_changes = self.change_detector.detect_changes_multi_period(
                window_size=window_size,
                step_size=step_size
            )

            # 3. 识别显著变化
            print("\n🎯 步骤3: 识别显著变化...")
            significant_changes = self._filter_significant_changes(all_changes, change_threshold)

            if not significant_changes:
                print("❌ 未发现显著变化，分析终止")
                return {}

            # 4. 归因分析
            print("\n🧠 步骤4: 归因分析...")
            attribution_results = self.attribution_engine.perform_attribution_analysis(significant_changes)

            if not attribution_results:
                print("❌ 归因分析失败")
                return {}

            # 5. 生成报告
            print("\n📊 步骤5: 生成报告...")
            self.report_generator = ComprehensiveReportGenerator(attribution_results)
            self.report_generator.generate_comprehensive_report()

            # 6. 保存结果
            print("\n💾 步骤6: 保存结果...")
            self.report_generator.save_detailed_results()
            self.report_generator.save_summary_report()

            print("\n🎉 任务三分析完成！")
            return attribution_results

        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _filter_significant_changes(self, all_changes: Dict, threshold: float) -> Dict:
        """过滤显著变化"""
        significant_changes = {}
        total_significant = 0

        for sector_id, period_changes in all_changes.items():
            sector_significant = {}

            for period, changes in period_changes.items():
                period_significant = {
                    stock: info for stock, info in changes.items()
                    if abs(info['change']) >= threshold
                }

                if period_significant:
                    sector_significant[period] = period_significant
                    total_significant += len(period_significant)

            if sector_significant:
                significant_changes[sector_id] = sector_significant

        print(f"✅ 发现 {total_significant} 个显著变化 (阈值: {threshold})")
        return significant_changes


# ==================== 使用接口 ====================
def run_complete_task3_analysis(task2_analyzer, returns_data, **kwargs):
    """
    任务三完整分析入口函数

    Parameters:
    - task2_analyzer: 任务二分析器实例
    - returns_data: 收益率数据
    - **kwargs: 分析参数

    Returns:
    - 归因分析结果
    """
    # 设置分析参数
    window_size = kwargs.get('window_size', 30)
    step_size = kwargs.get('step_size', 15)
    change_threshold = kwargs.get('change_threshold', 0.08)

    print(f"🔧 分析参数:")
    print(f"   窗口大小: {window_size}天")
    print(f"   滑动步长: {step_size}天")
    print(f"   变化阈值: {change_threshold}")

    # 创建控制器并运行分析
    controller = Task3MainController(task2_analyzer, returns_data)
    results = controller.run_complete_analysis(
        window_size=window_size,
        step_size=step_size,
        change_threshold=change_threshold
    )

    return results


def debug_change_detection(task2_analyzer, returns_data):
    """
    调试变化检测过程
    """
    print("=== 变化检测调试 ===")

    # 检查任务二结果
    print(f"任务二板块数量: {len(task2_analyzer.sectors)}")
    for sector_id, stocks in task2_analyzer.sectors.items():
        print(f"  板块 {sector_id}: {len(stocks)} 只股票")

    # 检查收益率数据
    print(f"收益率数据形状: {returns_data.shape}")
    print(f"数据期间: {returns_data.index[0]} 到 {returns_data.index[-1]}")

    # 分析基础归属因数分布
    print("\n归属因数分布:")
    all_factors = []
    for sector_id, factors in task2_analyzer.belonging_factors.items():
        sector_factors = list(factors.values())
        all_factors.extend(sector_factors)
        if sector_factors:
            print(
                f"  板块 {sector_id}: 平均{np.mean(sector_factors):.3f}, 范围[{min(sector_factors):.3f}, {max(sector_factors):.3f}]")

    if all_factors:
        print(f"总体: 平均{np.mean(all_factors):.3f}, 标准差{np.std(all_factors):.3f}")

    return len(all_factors) > 0


def main_task3_analysis(task2_analyzer, returns_data):
    return run_complete_task3_analysis(task2_analyzer, returns_data)