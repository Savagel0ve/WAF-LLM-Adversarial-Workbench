"""
实验结果分析与可视化
====================

生成论文风格的图表和分析报告。
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np

# 尝试导入绑图库
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，可视化功能将不可用")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ==================== 样式配置 ====================

# 论文风格配置
PAPER_STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}

# 颜色方案
COLORS = {
    "gptfuzzer": "#2E86AB",     # 蓝色
    "random_fuzzer": "#A23B72", # 紫红色
    "grammar_rl": "#F18F01",    # 橙色
    "reward_model": "#C73E1D",  # 红色
    "waf_feedback": "#3B1F2B",  # 深紫色
}

# 标记样式
MARKERS = {
    "gptfuzzer": "o",
    "random_fuzzer": "s",
    "grammar_rl": "^",
    "reward_model": "D",
    "waf_feedback": "v",
}


# ==================== 工具函数 ====================

def load_json(path: str) -> Dict:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_figure(fig, path: str, dpi: int = 300):
    """保存图表"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存: {path}")


def format_number(n: int) -> str:
    """格式化大数字"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ==================== RQ1: 有效性对比图 ====================

def plot_rq1_tp_curves(
    results_dir: str,
    attack_type: str,
    waf_type: str,
    output_path: Optional[str] = None
):
    """
    绘制RQ1的TP曲线对比图 (论文Figure 6风格)
    
    Args:
        results_dir: 结果目录
        attack_type: 攻击类型
        waf_type: WAF类型
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib未安装，跳过绑图")
        return
    
    # 应用论文样式
    plt.rcParams.update(PAPER_STYLE)
    
    fig, ax = plt.subplots()
    
    methods = ["gptfuzzer", "random_fuzzer", "grammar_rl"]
    
    for method in methods:
        # 加载数据
        result_path = Path(results_dir) / f"{attack_type}_{waf_type}_{method}.json"
        
        if not result_path.exists():
            print(f"跳过: {result_path} 不存在")
            continue
        
        data = load_json(str(result_path))
        
        # 提取检查点数据
        checkpoints = data.get("checkpoints", [])
        if not checkpoints:
            continue
        
        requests = [c["requests"] for c in checkpoints]
        tps = [c["tp"] for c in checkpoints]
        
        # 绑图
        ax.plot(
            requests, tps,
            label=method.replace("_", " ").title(),
            color=COLORS.get(method, "gray"),
            marker=MARKERS.get(method, "o"),
            markevery=max(1, len(requests) // 8),
        )
    
    # 设置标签
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("True Positives (TP)")
    ax.set_title(f"{attack_type.upper()} on {waf_type.title()}")
    
    # 格式化x轴
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_number(int(x))))
    
    # 图例
    ax.legend(loc="lower right")
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


def plot_rq1_comparison_bar(
    results_dir: str,
    output_path: Optional[str] = None
):
    """
    绘制RQ1的方法对比柱状图
    
    Args:
        results_dir: 结果目录
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update(PAPER_STYLE)
    
    # 加载汇总数据
    summary_path = Path(results_dir) / "summary.json"
    if not summary_path.exists():
        print(f"汇总文件不存在: {summary_path}")
        return
    
    summary = load_json(str(summary_path))
    results = summary.get("results", {})
    
    # 准备数据
    groups = list(results.keys())
    methods = ["gptfuzzer", "random_fuzzer", "grammar_rl"]
    
    x = np.arange(len(groups))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        tps = []
        for group in groups:
            tp = results.get(group, {}).get(method, {}).get("tp", 0)
            tps.append(tp)
        
        ax.bar(
            x + i * width, tps, width,
            label=method.replace("_", " ").title(),
            color=COLORS.get(method, "gray"),
        )
    
    ax.set_xlabel("Attack Type / WAF")
    ax.set_ylabel("True Positives (TP)")
    ax.set_title("RQ1: Method Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels([g.replace("_", " / ").upper() for g in groups], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


# ==================== RQ3: 数据规模影响图 ====================

def plot_rq3_data_scale(
    results_dir: str,
    attack_type: str,
    output_path: Optional[str] = None
):
    """
    绘制RQ3的数据规模影响图
    
    Args:
        results_dir: 结果目录
        attack_type: 攻击类型
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update(PAPER_STYLE)
    
    # 收集数据
    scales = [0, 20000, 256000, 512000]
    tps = []
    valid_rates = []
    
    for scale in scales:
        result_path = Path(results_dir) / f"{attack_type}_scale_{scale}.json"
        if result_path.exists():
            data = load_json(str(result_path))
            evaluation = data.get("evaluation", {})
            tps.append(evaluation.get("tp", 0))
            valid_rates.append(evaluation.get("bypass_rate", 0) * 100)
        else:
            tps.append(0)
            valid_rates.append(0)
    
    # 创建双Y轴图
    fig, ax1 = plt.subplots()
    
    # TP曲线
    color1 = COLORS["gptfuzzer"]
    ax1.set_xlabel("Pre-training Data Size")
    ax1.set_ylabel("True Positives (TP)", color=color1)
    ax1.plot(scales, tps, color=color1, marker='o', label="TP")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_number(int(x))))
    
    # Valid Rate曲线
    ax2 = ax1.twinx()
    color2 = COLORS["random_fuzzer"]
    ax2.set_ylabel("Bypass Rate (%)", color=color2)
    ax2.plot(scales, valid_rates, color=color2, marker='s', linestyle='--', label="Bypass Rate")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title(f"RQ3: Pre-training Data Scale Effect ({attack_type.upper()})")
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


# ==================== RQ5: 超参数热力图 ====================

def plot_rq5_heatmap(
    results_dir: str,
    attack_type: str,
    metric: str = "er",
    output_path: Optional[str] = None
):
    """
    绘制RQ5的超参数热力图
    
    Args:
        results_dir: 结果目录
        attack_type: 攻击类型
        metric: 指标 (er/nrr/ter)
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update(PAPER_STYLE)
    
    # 加载数据
    result_path = Path(results_dir) / f"{attack_type}_hyperparams.json"
    if not result_path.exists():
        print(f"文件不存在: {result_path}")
        return
    
    data = load_json(str(result_path))
    grid_results = data.get("grid_search", [])
    
    if not grid_results:
        print("无网格搜索结果")
        return
    
    # 提取唯一的KL系数和数据量
    kl_coefs = sorted(set(r["kl_coef"] for r in grid_results))
    data_sizes = sorted(set(r["reward_data_size"] for r in grid_results))
    
    # 构建矩阵
    matrix = np.zeros((len(kl_coefs), len(data_sizes)))
    
    for result in grid_results:
        i = kl_coefs.index(result["kl_coef"])
        j = data_sizes.index(result["reward_data_size"])
        matrix[i, j] = result.get(metric, 0)
    
    # 绘制热力图
    fig, ax = plt.subplots()
    
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    
    # 设置刻度
    ax.set_xticks(np.arange(len(data_sizes)))
    ax.set_yticks(np.arange(len(kl_coefs)))
    ax.set_xticklabels([format_number(s) for s in data_sizes])
    ax.set_yticklabels([f"{k:.1f}" for k in kl_coefs])
    
    ax.set_xlabel("Reward Model Data Size")
    ax.set_ylabel("KL Coefficient (β)")
    ax.set_title(f"RQ5: {metric.upper()} Heatmap ({attack_type.upper()})")
    
    # 添加数值标注
    for i in range(len(kl_coefs)):
        for j in range(len(data_sizes)):
            text = ax.text(j, i, f"{matrix[i, j]:.3f}",
                          ha="center", va="center", color="black", fontsize=10)
    
    # 颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.upper(), rotation=-90, va="bottom")
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


# ==================== RQ4: 奖励方法对比 ====================

def plot_rq4_comparison(
    results_dir: str,
    attack_type: str,
    output_path: Optional[str] = None
):
    """
    绘制RQ4的奖励方法对比图
    
    Args:
        results_dir: 结果目录
        attack_type: 攻击类型
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update(PAPER_STYLE)
    
    fig, ax = plt.subplots()
    
    methods = ["reward_model", "waf_feedback"]
    
    for method in methods:
        result_path = Path(results_dir) / f"{attack_type}_{method}.json"
        
        if not result_path.exists():
            print(f"跳过: {result_path}")
            continue
        
        data = load_json(str(result_path))
        epoch_results = data.get("epoch_results", [])
        
        if not epoch_results:
            continue
        
        requests = [e["requests"] for e in epoch_results]
        tps = [e["tp"] for e in epoch_results]
        
        ax.plot(
            requests, tps,
            label=method.replace("_", " ").title(),
            color=COLORS.get(method, "gray"),
            marker=MARKERS.get(method, "o"),
            markevery=max(1, len(requests) // 8),
        )
    
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Cumulative True Positives (TP)")
    ax.set_title(f"RQ4: Reward Model vs WAF Feedback ({attack_type.upper()})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


# ==================== 汇总报告生成 ====================

def generate_summary_report(
    results_base_dir: str,
    output_path: str = "experiment_report.md"
):
    """
    生成实验汇总报告
    
    Args:
        results_base_dir: 结果基础目录
        output_path: 输出文件路径
    """
    report_lines = [
        "# GPTFuzzer 实验结果报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # RQ1结果
    rq1_dir = Path(results_base_dir) / "rq1_effectiveness"
    if rq1_dir.exists():
        summary_path = rq1_dir / "summary.json"
        if summary_path.exists():
            summary = load_json(str(summary_path))
            
            report_lines.extend([
                "## RQ1: 有效性与效率",
                "",
                "| 配置 | GPTFuzzer TP | Random Fuzzer TP | Grammar-RL TP |",
                "|------|-------------|------------------|---------------|",
            ])
            
            for group, methods in summary.get("results", {}).items():
                gpt_tp = methods.get("gptfuzzer", {}).get("tp", "-")
                random_tp = methods.get("random_fuzzer", {}).get("tp", "-")
                grammar_tp = methods.get("grammar_rl", {}).get("tp", "-")
                report_lines.append(f"| {group} | {gpt_tp} | {random_tp} | {grammar_tp} |")
            
            report_lines.append("")
    
    # RQ5结果
    rq5_dir = Path(results_base_dir) / "rq5_hyperparams"
    if rq5_dir.exists():
        summary_path = rq5_dir / "summary.json"
        if summary_path.exists():
            summary = load_json(str(summary_path))
            
            report_lines.extend([
                "## RQ5: 最佳超参数配置",
                "",
                "| 攻击类型 | 最佳KL系数 | 最佳数据量 | ER | NRR |",
                "|----------|-----------|----------|-----|-----|",
            ])
            
            for attack, config in summary.get("best_configs", {}).items():
                report_lines.append(
                    f"| {attack.upper()} | {config.get('best_kl_coef', '-')} | "
                    f"{config.get('best_data_size', '-')} | "
                    f"{config.get('er', 0):.4f} | {config.get('nrr', 0):.4f} |"
                )
            
            report_lines.append("")
    
    # 写入报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"报告已生成: {output_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="实验结果分析与可视化")
    
    parser.add_argument("--results_dir", type=str, default="results",
                        help="结果目录")
    parser.add_argument("--rq", type=str, choices=["rq1", "rq3", "rq4", "rq5", "all"],
                        default="all", help="要分析的RQ")
    parser.add_argument("--attack_type", type=str, default="sqli",
                        help="攻击类型")
    parser.add_argument("--waf_type", type=str, default="modsecurity",
                        help="WAF类型")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="图表输出目录")
    parser.add_argument("--report", action="store_true",
                        help="生成汇总报告")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.report:
        generate_summary_report(args.results_dir, str(output_dir / "experiment_report.md"))
        return
    
    if args.rq in ["rq1", "all"]:
        rq1_dir = Path(args.results_dir) / "rq1_effectiveness"
        if rq1_dir.exists():
            plot_rq1_tp_curves(
                str(rq1_dir),
                args.attack_type,
                args.waf_type,
                str(output_dir / f"rq1_tp_curves_{args.attack_type}_{args.waf_type}.png")
            )
            plot_rq1_comparison_bar(
                str(rq1_dir),
                str(output_dir / "rq1_comparison_bar.png")
            )
    
    if args.rq in ["rq3", "all"]:
        rq3_dir = Path(args.results_dir) / "rq3_data_scale"
        if rq3_dir.exists():
            plot_rq3_data_scale(
                str(rq3_dir),
                args.attack_type,
                str(output_dir / f"rq3_data_scale_{args.attack_type}.png")
            )
    
    if args.rq in ["rq4", "all"]:
        rq4_dir = Path(args.results_dir) / "rq4_reward_vs_waf"
        if rq4_dir.exists():
            plot_rq4_comparison(
                str(rq4_dir),
                args.attack_type,
                str(output_dir / f"rq4_comparison_{args.attack_type}.png")
            )
    
    if args.rq in ["rq5", "all"]:
        rq5_dir = Path(args.results_dir) / "rq5_hyperparams"
        if rq5_dir.exists():
            for metric in ["er", "nrr"]:
                plot_rq5_heatmap(
                    str(rq5_dir),
                    args.attack_type,
                    metric,
                    str(output_dir / f"rq5_heatmap_{args.attack_type}_{metric}.png")
                )
    
    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
