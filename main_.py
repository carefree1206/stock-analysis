# main.py - 整合所有任务的主程序
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from task1_analysis import main as task1_main
from task2_analysis import main_task2_analysis
from task3_analysis import main_task3_analysis
from task4_analysis import main_task4_analysis


def run_all_tasks():
    """运行所有任务"""
    print("🚀 开始运行大湾区杯股票分析所有任务")
    print("=" * 60)

    try:
        # 任务一：板块联动分析
        print("\n📊 任务一：军工板块联动分析")
        print("-" * 40)
        task1_main()

        # 任务二：板块检测和归属因数计算
        print("\n📊 任务二：智能板块检测")
        print("-" * 40)
        task2_analyzer, returns_data = main_task2_analysis()

        # 任务三：归属因数变化归因分析
        print("\n📊 任务三：归属因数变化归因分析")
        print("-" * 40)
        if task2_analyzer:
            task3_results = main_task3_analysis(task2_analyzer, returns_data)
        else:
            print("❌ 任务二分析失败，跳过任务三")
            task3_results = None

        # 任务四：板块评价基准与归因分析
        print("\n📊 任务四：板块评价基准分析")
        print("-" * 40)
        if task2_analyzer:
            task4_results = main_task4_analysis(
                task2_analyzer=task2_analyzer,
                max_sectors=3,
                max_stocks_per_sector=10
            )
        else:
            print("❌ 任务二分析失败，使用示例数据运行任务四")
            task4_results = main_task4_analysis()

        print("\n🎉 所有任务完成！")
        return {
            'task2': task2_analyzer,
            'task3': task3_results,
            'task4': task4_results
        }

    except Exception as e:
        print(f"❌ 运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_all_tasks()