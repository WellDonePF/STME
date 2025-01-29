import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import copy
import time
import argparse
import xlsxwriter

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def get_image_info(model, tokenizer, image_processor, item, args):
    args.dataset_name = item["dataset_name"]
    context = item["context"]
    context = context.replace("<ImageHere>", DEFAULT_IMAGE_TOKEN)
    if args.dataset_name != "CM":
        context = context.rstrip("Your answer is: ")
        context = context + "\nRespond with the answer and your thought process for reasoning."

    images = item["raw_img_list"]
    images = [Image.open(img).convert("RGB") for img in images]
    images = process_images(images, image_processor, model.config)
    images = [img.to(dtype=torch.float16) for img in images]
    image_sizes = [img.size for img in images]

    args.imgnum = len(images)
    args.sample_id = item["sample_id"]
    args.target_id = item["target_id"]
    args.gt = item["response"]

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], context)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
    tokens, image_ranges = model.get_token_num(
        input_ids,
        images=images,
        image_sizes=image_sizes,
    )

    token_num = len(tokens)
    return image_ranges, token_num


def write_score_to_sheet(worksheet, bold_format, col_names, row_name, gt, metadata, other_data=None):
    last_row_idx = metadata.shape[0] + 2
    # 插入第一行表头，gt加粗
    for col_num, col_name in enumerate(col_names):
        if gt >= 0 and col_num == gt+1:  # 最大列加粗
            worksheet.write(0, col_num, col_name, bold_format)
        else:
            worksheet.write(0, col_num, col_name)

    # 插入递增的 [1, 2, 3, ...] 到第一列，并从第二行开始
    for row_num in range(metadata.shape[0]):
        worksheet.write(row_num + 1, 0, f"{row_name} {row_num + 1}")  # 第一列写入递增的数字，从第二行开始

    # 将矩阵写入到相应的工作表中，从第二列、第二行开始写
    for row_num, row_data in enumerate(metadata):
        max_index_text = np.argmax(metadata[row_num])
        for col_num, cell_data in enumerate(row_data):
            if col_num == max_index_text:  # 值最大的列加粗
                worksheet.write(row_num + 1, col_num + 1, cell_data, bold_format)  # 数据写入第+1列，行号偏移1
            else:
                worksheet.write(row_num + 1, col_num + 1, cell_data)  # 数据从第2列开始写入，行号偏移1

    # 写入其他数据
    if other_data is not None:
        for i, data in enumerate(other_data):
            value = data["value"]
            worksheet.write(last_row_idx+i, 0, data["name"])
            if data["bold"]:
                max_index = np.argmax(value)
            for col_num, cell_data in enumerate(value):
                if data["bold"] and col_num == max_index:  # 最大列加粗
                    worksheet.write(last_row_idx+i, col_num + 1, cell_data, bold_format)
                else:
                    worksheet.write(last_row_idx+i, col_num + 1, cell_data)


def cal_head_weights(args):
    llava_model_args = {"multimodal": True}
    overwrite_config = {"image_aspect_ratio": "pad"}
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.pretrained_model_name_or_path,
        model_base=None,
        model_name=args.model_name,
        device_map="cuda",
        torch_dtype="bfloat16",
        attn_implementation="eager",
        **llava_model_args
    )
    info = json.loads(open(args.data_path,'r',encoding="utf-8").read())
    new_info = []
    for item in info:
        score_path_qwenvl = os.path.join(
            args.root_path, args.task_type, "scores",
            f'{item["dataset_name"]}_sample{item["sample_id"]}_imgnum{len(item["raw_img_list"])}_target{item["target_id"]}_scores.xlsx'
        )
        if not os.path.exists(score_path_qwenvl):
            new_info.append(item)
    info = new_info
    if args.num_worker == 1:  # 如果有多个进程，则取消进度条
        args.decoder_ids = tqdm(args.decoder_ids)

    while True:
        sleep_cnt = 0
        while len(os.listdir(args.tensor_path)) == 0:
            time.sleep(10)
            sleep_cnt += 1
            if sleep_cnt == 160:
                print(f"finish: task {args.task_type}\trank {args.rank}")
                return

        cnt = -1
        for item in info:
            # if item["dataset_name"] != "SlideVQA" or item["sample_id"] not in [25]:
            #     continue
            cnt += 1
            if cnt % args.num_worker != args.rank:
                continue
            attn_path = os.path.join(args.tensor_path, f'{item["dataset_name"]}_sample{item["sample_id"]}')
            if not os.path.exists(attn_path) or len(os.listdir(attn_path)) != total_layer:
                continue
            image_ranges, token_num = get_image_info(model, tokenizer, image_processor, item, args)
            other_data = []

            # 保存excel文件的路径
            score_path = os.path.join(
                args.score_path,
                f"{args.dataset_name}_sample{args.sample_id}_imgnum{args.imgnum}_target{args.target_id}_scores.xlsx"
            )
            workbook = xlsxwriter.Workbook(score_path)
            bold_format = workbook.add_format({'bold': True})
            first_worksheet = workbook.add_worksheet('total_info') # 先生成第一个sheet

            if args.dataset_name != "GPR1200":
                col_names = [""] + [f"image{j+1}" for j in range(args.imgnum)]
                img_value_text_total = np.zeros((total_layer, args.imgnum))

                for layer_index in args.decoder_ids:
                    attn = torch.load(
                        os.path.join(attn_path, f"sample{args.sample_id}_layer{layer_index}.pt"),
                        weights_only=True,
                        map_location="cpu"
                    ).float() * args.scale

                    # 提取输入文本和输出文本对应的矩阵
                    output_text_pos = list(range(token_num, attn.shape[-1]))
                    input_text_pos = list(range(image_ranges[-1][1], token_num-5))
                    iutput_text_matrix = attn[0, :, input_text_pos].cpu().numpy()
                    output_text_matrix = attn[0, :, output_text_pos].cpu().numpy()
                    text_matrix = np.concatenate([iutput_text_matrix, output_text_matrix], axis=1)  # heads * text_token_num * img_token_num

                    img_value_text = np.zeros(args.imgnum)  # 存储text和所有image的decoder的attn值
                    img_value_text_with_head = np.zeros((total_head, args.imgnum))  # 存储text和所有image的head的attn值
                    for i in range(args.imgnum):
                        img_matrix = text_matrix[:, :, image_ranges[i][0]:image_ranges[i][1]]  # heads * img_token_num * img_token_num
                        img_matrix = img_matrix.mean(axis=(1,2))  # heads
                        img_value_text_with_head[:, i] = img_matrix
                        img_value_text[i] = img_matrix.mean()
                    img_value_text_total[layer_index] = img_value_text

                    worksheet = workbook.add_worksheet(f'decoder_{layer_index+1}')
                    write_score_to_sheet(
                        worksheet=worksheet,
                        bold_format=bold_format,
                        col_names=col_names,
                        row_name="head",
                        gt=args.target_id-1,
                        metadata=img_value_text_with_head,
                        other_data=[{"name": "avg", "value": img_value_text, "bold": True}] + other_data
                    )
            else:
                col_names = [""] + [f"image{j+1}" for j in range(args.imgnum-1)]
                img_value_anchor_total = np.zeros((total_layer, args.imgnum-1))

                for layer_index in args.decoder_ids:
                    attn = torch.load(
                        os.path.join(attn_path, f"sample{args.sample_id}_layer{layer_index}.pt"),
                        weights_only=True,
                        map_location="cpu",
                    ).float() * args.scale

                    # 提取输入文本和输出文本对应的矩阵
                    anchor_img_pos = list(range(image_ranges[-1][0], image_ranges[-1][1]))
                    anchor_matrix = attn[0, :, anchor_img_pos].cpu().numpy()  # heads * tar_img_token_num * img_token_num

                    img_value_anchor = np.zeros(args.imgnum-1)
                    img_value_anchor_with_head = np.zeros((total_head, args.imgnum-1))
                    for i in range(args.imgnum-1):
                        img_matrix = anchor_matrix[:, :, image_ranges[i][0]:image_ranges[i][1]]  # heads * img_token_num * img_token_num
                        img_matrix = img_matrix.mean(axis=(1,2))  # heads
                        img_value_anchor_with_head[:, i] = img_matrix
                        img_value_anchor[i] = img_matrix.mean()
                    img_value_anchor_total[layer_index] = img_value_anchor

                    worksheet = workbook.add_worksheet(f'decoder_{layer_index+1}')
                    write_score_to_sheet(
                        worksheet=worksheet,
                        bold_format=bold_format,
                        col_names=col_names,
                        row_name="head",
                        gt=args.target_id-1,
                        metadata=img_value_anchor_with_head,
                        other_data=[{"name": "avg", "value": img_value_anchor, "bold": True}] + other_data
                    )

            final_value = img_value_anchor_total if args.dataset_name == "GPR1200" else img_value_text_total
            max_index = np.argmax(final_value[-1])
            decoder_image_value = final_value[args.decoder_ids].mean(axis=0)
            write_score_to_sheet(
                worksheet=first_worksheet,
                bold_format=bold_format,
                col_names=col_names,
                row_name="decoder",
                gt=args.target_id-1,
                metadata=final_value,
                other_data=[{"name": "avg", "value": decoder_image_value, "bold": True}] + other_data
            )
            workbook.close()

            if args.delete_tensor:
                shutil.rmtree(attn_path)
            print("\n" + "*"*80)
            print(f"cnt: {cnt}\tdata_name: {args.dataset_name}\tsample_index: {args.sample_id}")
            print(f"target_id: {args.target_id}\tattn_index: {int(max_index)+1}")
            print(f"image_num={args.imgnum}")
            print("*"*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/LLaVA-NeXT/attn_score_analysis/llava-ov-0.5B")
    parser.add_argument("--task_type", type=str, default="easy", choices=["easy", "medium", "hard"])

    parser.add_argument("--num_worker", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1e6)
    parser.add_argument("--decoder_ids", type=list, default=list(range(0,total_layer)))
    parser.add_argument("--delete_tensor", action="store_true")

    args = parser.parse_args()
    args.data_path = os.path.join(
        "/cpfs/user/wangpengfei2/project/usage/attn_score_analysis/make_data",
        f"merged_result_{args.task_type}.json"
    )
    args.tensor_path = os.path.join(args.root_path, args.task_type, "tensors")
    args.score_path = os.path.join(args.root_path, args.task_type, "scores")
    os.makedirs(args.score_path, exist_ok=True)

    return args


if __name__ == "__main__":
    total_head = 14
    total_layer = 24

    cal_head_weights(parse_args())
