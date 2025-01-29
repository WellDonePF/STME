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
    # models = ["internvl2-1B","internvl2-2B","internvl2-4B","internvl2-8B","internvl2-26B"]
    models = ["internvl2-1B","internvl2-2B","internvl2-8B","internvl2-26B"]
    score_path = "/cpfs/user/wangpengfei2/project/InternVL/metric/res"
    # scales = ["1B", "2B", "4B", "8B", "26B"]
    scales = ["1B", "2B", "8B", "26B"]
    formats = {
        "Acc": {"color":'#d62728', "linestyle":'--', "marker":'o'},
        "Attn Acc": {"color":'#ff7f0e', "linestyle":'-', "marker":'s'}
    }
    task_map = {"easy": "Easy", "medium": "Hard"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    axs = {"easy": ax1, "medium":ax2}
    plt.rcParams.update({"font.size":20})
    for task_type in ["easy", "medium"]:
        accs = []
        attn_accs = []
        for model_name in models:
            scores = json.loads(
                open(os.path.join(score_path,model_name,task_type,"result.json"),'r',encoding="utf-8").read())
            accs.append(scores["Acc"])
            attn_accs.append(scores["Max Attn Acc"])

        if task_type == "medium":
            axs[task_type].set_xlabel(f"InternVL2", fontsize=26, fontproperties=args.fontprop)
            axs[task_type].set_ylabel(f"Attention Acc", fontsize=26, fontproperties=args.fontprop)
        else:
            axs[task_type].set_ylabel(f"Acc", fontsize=26, fontproperties=args.fontprop)
            axs[task_type].plot(scales, accs, linewidth=3.5, markersize=7, label="Acc", **formats["Acc"])
            axs[task_type].plot(scales, attn_accs, linewidth=3.5, markersize=7, label="Attn Acc", **formats["Attn Acc"])

        axs[task_type].set_xticklabels(scales, fontsize=16, fontproperties=args.fontprop)

        axs[task_type].grid(axis='y', linestyle='-', color='gray', alpha=0.15)
        axs[task_type].legend(loc="lower right", prop={'family': args.fontprop, 'size': 19})

    plt.tight_layout()
    plt.savefig(f"/cpfs/user/wangpengfei2/project/InternVL/metric/5_different_scale_{args.model_name}.pdf",format='pdf',dpi=3000, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/Qwen2VL/attn_score_analysis")
    parser.add_argument("--save_path", type=str, default="/cpfs/user/wangpengfei2/project/Qwen2VL/figures")
    parser.add_argument("--model_name", type=str, default="qwen2vl-7B")
    parser.add_argument("--fontprop", type=str, default="Arial")
    args = parser.parse_args()

    plot_figure()
