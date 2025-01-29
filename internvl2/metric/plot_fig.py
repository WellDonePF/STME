import os
import re
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 融合不同的analysis结果文件
def plot_figure():
    scores = dict()
    formats = {
        "easy": {"color":'#d62728', "linestyle":'--', "marker":'o'},
        "medium": {"color":'#ff7f0e', "linestyle":'-', "marker":'s'}
    }
    task_map = {"easy": "Easy", "medium": "Hard"}

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 12))
    plt.rcParams.update({"font.size":20})
    for task_type in ["easy", "medium"]:
        accs = []
        attn_accs = []
        for model_name in args.models:
            scores = json.loads(
                open(os.path.join(args.score_path,model_name,task_type,"result.json"),'r',encoding="utf-8").read())
            accs.append(scores["Acc"])
            attn_accs.append(scores["Max Attn Acc"])
        ax1.plot(args.scales, accs, linewidth=3.5, markersize=10, label=task_map[task_type], **formats[task_type])
        ax2.plot(args.scales, attn_accs, linewidth=3.5, markersize=10, label=task_map[task_type], **formats[task_type])

    ax1.set_ylabel(f"Attention Acc.", fontsize=26, fontproperties=args.fontprop)
    ax2.set_ylabel(f"Answer Acc.", fontsize=26, fontproperties=args.fontprop)
    ax2.set_xlabel(args.model_series, fontsize=26, fontproperties=args.fontprop)

    for ax in (ax1, ax2):
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xticks(range(len(args.scales)))
        ax.set_xticklabels(args.scales, fontsize=16, fontproperties=args.fontprop)
        ax.grid(axis='y', linestyle='-', color='gray', alpha=0.15)
        ax.legend(loc="lower right", prop={'family': args.fontprop, 'size': 19})

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.metric_path, f"5_different_scale_{args.model_series}.pdf"),
        format='pdf',dpi=3000, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--metric_path", type=str, default="/cpfs/user/wangpengfei2/project/InternVL/metric")
    # parser.add_argument("--model_series", type=str, default="InternVL")
    # parser.add_argument("--fontprop", type=str, default="Arial")
    # parser.add_argument("--scales", type=list, default=["1B", "2B", "4B", "8B", "26B"])
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_path", type=str, default="/cpfs/user/wangpengfei2/project/Qwen2VL/metric")
    parser.add_argument("--model_series", type=str, default="Qwen2VL")
    parser.add_argument("--fontprop", type=str, default="Arial")
    parser.add_argument("--scales", type=list, default=["2B", "7B"])
    args = parser.parse_args()

    args.models = []
    for sca in args.scales:
        args.models.append(f"{args.model_series.lower()}-{sca}")
    args.score_path = os.path.join(args.metric_path, "res")
    plot_figure()
