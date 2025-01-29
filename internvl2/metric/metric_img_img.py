import os
import re
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.comments import Comment

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# 融合不同的analysis结果文件
def merge_analysis_file():
    marged_info = f"/cpfs/user/wangpengfei2/project/usage/attn_score_analysis/make_data/merged_result_{args.task_type}.json"
    marged_info = json.loads(open(marged_info,'r',encoding="utf-8").read())
    marged_info = {(item['dataset_name'], item['sample_id']): item for item in marged_info}

    pred = os.path.join(args.root_path, "info/pred.json")
    pred = json.loads(open(pred,'r',encoding="utf-8").read())
    pred = {(item['dataset_name'], item['sample_id']): item for item in pred}

    is_right = os.path.join(args.root_path, "info/eval_score.json")
    is_right = json.loads(open(is_right,'r',encoding="utf-8").read())
    is_right = {(item['dataset_name'], item['sample_id']): item for item in is_right}

    scores_path = os.path.join(args.root_path, "scores")
    scores = sorted(os.listdir(scores_path))
    real_scores = [os.path.join(scores_path, x) for x in scores]

    used = []
    all_res = []
    for i in range(len(real_scores)):
        # CharacterOrder_sample107_imgnum9_target9_scores.xlsx
        file_name = scores[i].split("_")
        dataset_name = file_name[0]
        if dataset_name != "GPR1200":
            continue
        sample_id = int(file_name[1].strip("sample"))
        if (dataset_name, sample_id) not in pred or (dataset_name, sample_id) not in marged_info:
            print(f"lost data: {dataset_name, sample_id}")
            continue
        if (dataset_name, sample_id) in used:
            print(f"repeat data: {dataset_name, sample_id}")
            break

        used.append((dataset_name, sample_id))
        item = pred[(dataset_name, sample_id)]
        item["gpt_answer"] = marged_info[(dataset_name, sample_id)]["answer"] if "answer" in marged_info[(dataset_name, sample_id)] else ""
        item["is_right"] = is_right[(dataset_name, sample_id)]["score"]
        score_data = pd.read_excel(real_scores[i], sheet_name="total_info")
        score_data = score_data.iloc[:args.total_layer, 1:].to_numpy()

        for last_N in args.N:
            save_key = args.metric.replace("N",str(last_N))
            last_N_rows = score_data[-last_N:]  # 取出最后N行decoder
            if args.metric == "LND":
                max_value_index = np.argmax(last_N_rows)  # 在展平的数组中找到最大值的索引
                _, max_value_col = np.unravel_index(max_value_index, last_N_rows.shape)  # 将展平索引转换为行、列索引
                max_value_col = int(max_value_col)+1  # 转化为int，并+1
            elif args.metric == "M-LND":
                mean_last_N_rows = last_N_rows.mean(axis=0)  # 按照decoder维度求均值
                max_value_col = int(np.argmax(mean_last_N_rows)) + 1  # 求完均值后找最大索引，并+1
            elif args.metric == "MC-LND":
                max_value_indices = np.argmax(last_N_rows, axis=1)  # 每一个decoder求最大值索引
                img_counts = Counter(max_value_indices)  # 计算每个元素的出现次数
                max_count = max(img_counts.values())  # 找到最大出现次数
                all_common_elem = [elem for elem, count in img_counts.items() if count == max_count]  # 找出所有出现次数等于 max_count 的元素
                if len(all_common_elem) > 1:
                    element_indices = {elem: np.where(max_value_indices == elem)[0] for elem in all_common_elem}
                    accu_values = np.zeros(last_N_rows.shape[1])
                    for img_idx in element_indices:
                        decoder_indices = element_indices[img_idx]  # MAF image索引=img_idx的 所有deocder索引
                        for decoder_idx in decoder_indices:
                            accu_values[img_idx] = accu_values[img_idx] + last_N_rows[decoder_idx][img_idx]
                    max_value_col = int(np.argmax(accu_values)) + 1  # 找最大索引，并+1
                else:
                    max_value_col = int(all_common_elem[0]) + 1  # 求完均值后找最大索引，并+1
            item[save_key] = max_value_col
        all_res.append(item)

    all_res.sort(key=lambda x: (x['dataset_name'], x['sample_id']))
    open(os.path.join(args.save_file_path, f"img_img_analysis_{args.metric}.json"),'w',encoding="utf-8"
        ).write(json.dumps(all_res,indent=4))


# 计算混淆矩阵
def cal_matrix():
    analysis_file_path = os.path.join(args.save_file_path, f"img_img_analysis_{args.metric}.json")
    analysis = json.loads(open(analysis_file_path,'r',encoding="utf-8").read())
    metric_value = dict()

    for last_N in args.N:
        save_key = args.metric.replace("N",str(last_N))
        is_right = []
        attn_index = []
        for item in analysis:
            is_right.append(item["is_right"])
            attn_index.append(int(item[save_key]==item["target_id"]))
            # if is_right[-1] == 0 and attn_index[-1] == 0:
            #     print(item["dataset_name"], item["sample_id"])

        cm = confusion_matrix(is_right, attn_index)
        precision_0 = cm[0, 1] / (cm[0, 0] + cm[0, 1])  # 类别 0 的精度
        precision_1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # answer_true & attn_true 在 answer_true 中的占比
        recall_0 = cm[1, 0] / (cm[0, 0] + cm[1, 0])     # 类别 0 的召回率
        recall_1 = cm[1, 1] / (cm[1, 1] + cm[0, 1])     # answer_true & attn_true 在 attn_true 中的占比
        total_num = cm[0, 0] + cm[0, 1] + cm[1, 1] + cm[1, 0]

        metric_value[save_key] = [precision_0,precision_1,recall_0,recall_1]
    # print(f"acc: {sum(is_right)/len(is_right)}")
    return metric_value, sum(is_right)/len(is_right)


def plot_figure(metric_values, acc):
    formats = {
        "LND": {"label":"LND", "color":'#96C37D', "linestyle":'-', "marker":'o'},
        "M-LND": {"label":"M-LND", "color":'blue', "linestyle":'--', "marker":'s'},
        "MC-LND": {"label":"MC-LND", "color":'#C32423', "linestyle":'-.', "marker":'^'},
    }
    all_res = {"Acc": acc, "Attn Acc": dict()}

    plt.figure(figsize=(12, 4))
    max_value = -1
    min_max_value_num = float("inf")
    for key in metric_values:
        metric_value = metric_values[key]
        x = list(metric_value.keys())
        x = [int(key.split("L")[-1].strip("D")) for key in x]
        y = list(metric_value.values())
        y = [value[1] for value in y]
        if max(y) > max_value:
            max_value = max(y)
            min_max_value_num = y.index(max_value)+1
        elif max(y) == max_value:
            min_max_value_num = min(min_max_value_num, y.index(max_value)+1)
        # print(f"method: {key}\tmax_attn_acc: {max(y)}\t")
        all_res["Attn Acc"][key] = float(max(y))
        plt.plot(x, y,linewidth=2, markersize=3, **formats[key])
    all_res["Max Attn Acc"] = max_value
    all_res["Max Attn Acc N"] = min_max_value_num

    # 添加水平和垂直线
    plt.axhline(y=max_value, color='#555555', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.axvline(x=min_max_value_num, color='#555555', linestyle='--', linewidth=1.2, alpha=0.3)

    plt.title("Hard Tasks", fontweight="bold", fontsize=14)
    plt.xlabel("Last $N$ Decoders", fontsize=13, fontweight="bold", fontproperties=args.fontprop)    # 加粗X轴标签
    ylabel = plt.ylabel('Attention Acc.', labelpad=-40, rotation=0, fontsize=13, fontweight="bold", fontproperties=args.fontprop)    # 加粗X轴标签
    ylabel.set_y(1.01)
    plt.legend(prop={'family': args.fontprop, 'size': 13})

    x_ticks = x
    plt.xticks(ticks=x, labels=x_ticks, fontweight='bold', fontproperties=args.fontprop)
    plt.grid(axis='y', linestyle='-', color='gray', alpha=0.15)
    plt.text(22, 0.948, "Maximum Acc.", fontsize=13, color='#555555', alpha=0.8)

    plt.savefig(
        os.path.join(args.save_figure_path, f"img_img_metric_{args.task_type}_{args.model_name}.pdf"),
        format='pdf', dpi=3000, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    open(os.path.join(args.save_file_path, f"img_img_result.json"),'w',encoding="utf-8"
        ).write(json.dumps(all_res, indent=4))
    print(all_res)


def main():
    metric_value = dict()
    for metric in ["LND", "M-LND", "MC-LND"]:
        args.metric = metric
        merge_analysis_file()
        metric_value[metric], acc = cal_matrix()
    plot_figure(metric_value, acc)


if __name__ == "__main__":
    num_map = {
        "internvl2-1B": {
            "total_layer": 14,
            "total_head": 24
        },
        "internvl2-2B": {
            "total_layer": 16,
            "total_head": 24
        },
        "internvl2-4B": {
            "total_layer": 32,
            "total_head": 32
        },
        "internvl2-8B": {
            "total_layer": 32,
            "total_head": 32
        },
        "internvl2-26B": {
            "total_layer": 48,
            "total_head": 48
        },
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/InternVL/attn_score_analysis")
    parser.add_argument("--metric_path", type=str, default="/cpfs/user/wangpengfei2/project/InternVL/metric")
    parser.add_argument("--model_name", type=str, default="internvl2-2B")
    parser.add_argument("--task_type", type=str, default="medium", choices=["easy", "medium"])
    parser.add_argument("--fontprop", type=str, default="Arial")
    args = parser.parse_args()

    args.root_path = os.path.join(args.root_path, args.model_name, args.task_type)
    args.save_file_path = os.path.join(args.metric_path, "res", args.model_name, args.task_type)
    args.save_figure_path = os.path.join(args.metric_path, "figs")
    os.makedirs(args.save_figure_path, exist_ok=True)
    os.makedirs(args.save_file_path, exist_ok=True)

    args.total_layer = num_map[args.model_name]["total_layer"]
    args.total_head = num_map[args.model_name]["total_head"]
    args.N = list(range(1,args.total_layer+1))

    main()
