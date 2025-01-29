import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import shutil
import copy
import time
import argparse
import xlsxwriter
import matplotlib.pyplot as plt

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

    args.sample_id = item["sample_id"]
    args.target_id = item["target_id"]
    args.gt = item["response"]

    images = item["raw_img_list"]
    images = [Image.open(img).convert("RGB") for img in images]
    args.tar_img = images[args.target_id-1]

    images = process_images(images, image_processor, model.config)
    images = [img.to(dtype=torch.float16) for img in images]
    image_sizes = [img.size for img in images]
    args.imgnum = len(images)

    conv = copy.deepcopy(conv_templates[args.conv_template])
    conv.append_message(conv.roles[0], context)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
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
        device_map=args.device,
        **llava_model_args
    )
    info = json.loads(open(args.data_path,'r',encoding="utf-8").read())

    cnt = -1
    for item in info:
        cnt += 1
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

            for layer_index in tqdm(args.decoder_ids):
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

                img_value_text = np.zeros(args.imgnum)  # img_num
                img_value_text_with_head = np.zeros((total_head, args.imgnum))  # heads * img_num
                for i in range(args.imgnum):
                    img_matrix = text_matrix[:, :, image_ranges[i][0]:image_ranges[i][1]]  # heads * text_token_num * tar_img_token_num
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

            for layer_index in tqdm(args.decoder_ids):
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
                    img_matrix = anchor_matrix[:, :, image_ranges[i][0]:image_ranges[i][1]]  # heads * text_token_num * tar_img_token_num
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
        attn_index = int(np.argmax(final_value[-1]))
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

        # 如果attn正确，那么进行token level分析
        if attn_index+1 == args.target_id:
            save_token_img_pth = os.path.join(
                args.root_path, args.task_type, "patch_imgs",
                f"{args.dataset_name}_sample{args.sample_id}"
            )
            os.makedirs(save_token_img_pth, exist_ok=True)
            args.tar_img.save(os.path.join(save_token_img_pth, "org_img.jpg"))

            # 获取patch信息
            original_width, original_height = args.tar_img.size
            patch_size = 14
            target_size = (384, 384)  # 宽，高
            scale_w = target_size[0] / original_width
            scale_h = target_size[1] / original_height
            num_patches_w = target_size[0] // patch_size
            num_patches_h = target_size[1] // patch_size
            resized_img = args.tar_img.resize(target_size)

            # 将patch和原图对应起来
            patch_to_original_mapping = []
            for i in range(num_patches_h):  # 遍历每个patch的高度索引
                for j in range(num_patches_w):  # 遍历每个patch的宽度索引
                    # 计算resize图像中patch的左上角和右下角
                    x_start_resized = j * patch_size
                    y_start_resized = i * patch_size
                    x_end_resized = x_start_resized + patch_size
                    y_end_resized = y_start_resized + patch_size

                    # 对应到原图的区域 (反向缩放)
                    x_start_original = int(x_start_resized / scale_w)
                    x_end_original = int(x_end_resized / scale_w)
                    y_start_original = int(y_start_resized / scale_h)
                    y_end_original = int(y_end_resized / scale_h)

                    # 记录这个patch对应原图的区域
                    patch_to_original_mapping.append({
                        'patch_index': (i, j),
                        'resized_patch_coords': (x_start_resized, y_start_resized, x_end_resized, y_end_resized),
                        'original_patch_coords': (x_start_original, y_start_original, x_end_original, y_end_original)
                    })

            # 在原图上添加网格
            img_with_grid = args.tar_img.copy()
            draw = ImageDraw.Draw(img_with_grid)

            # 遍历每个patch对应的原图区域，绘制网格并保存
            for mapping in patch_to_original_mapping:
                x_start, y_start, x_end, y_end = mapping['original_patch_coords']
                draw.rectangle([x_start, y_start, x_end, y_end], outline="red", width=2)
            img_with_grid.save(
                os.path.join(
                    save_token_img_pth,
                    f"target{args.target_id}.jpg"
                )
            )

            # 遍历每个层进行分析
            for layer_index in tqdm(args.decoder_ids):
                attn = torch.load(
                    os.path.join(attn_path, f"sample{args.sample_id}_layer{layer_index}.pt"),
                    weights_only=True,
                    map_location="cpu"
                ).float() * args.scale
                if args.dataset_name != "GPR1200":
                    output_text_pos = list(range(token_num, attn.shape[-1]))
                    input_text_pos = list(range(image_ranges[-1][1], token_num-5))
                    iutput_text_matrix = attn[0, :, input_text_pos].cpu().numpy()
                    output_text_matrix = attn[0, :, output_text_pos].cpu().numpy()
                    text_matrix = np.concatenate([iutput_text_matrix, output_text_matrix], axis=1)  # dim: heads * text_token_num * img_token_num

                    # token 级别的分析
                    img_matrix = text_matrix[:, :, image_ranges[attn_index][0]:image_ranges[attn_index][1]]  # dim: heads * text_token_num * tar_img_token_num
                    img_matrix = img_matrix.mean(axis=(0,1))[:-1]  # dim: tar_img_token_num-1  去除末尾的pad_token
                else:
                    anchor_img_pos = list(range(image_ranges[-1][0], image_ranges[-1][1]-1))
                    anchor_matrix = attn[0, :, anchor_img_pos].cpu().numpy()

                    # token 级别的分析
                    img_matrix = anchor_matrix[:, :, image_ranges[attn_index][0]:image_ranges[attn_index][1]]  # dim: heads * anchor_token_num * tar_img_token_num
                    img_matrix = img_matrix.mean(axis=(0,1))[:-1]  # dim: tar_img_token_num-1  去除末尾的pad_token

                mask_img = args.tar_img.convert("RGBA").copy()
                draw = ImageDraw.Draw(mask_img)
                mask_resized_img = resized_img.convert("RGBA").copy()
                resized_draw = ImageDraw.Draw(mask_resized_img)

                # 扁平化patch值矩阵，进行排序并获取阈值
                median_value = np.median(img_matrix)
                top10_per_value = np.percentile(img_matrix, 90)

                # 遍历每个patch并处理
                img_matrix = img_matrix.reshape((num_patches_h, num_patches_w))
                for mapping in patch_to_original_mapping:
                    i, j = mapping['patch_index']
                    # 如果patch值小于或等于中位数，变成黑色
                    # if img_matrix[i, j] <= median_value:
                    #     draw.rectangle(mapping['original_patch_coords'], fill=(0, 0, 0, 255))
                    #     resized_draw.rectangle(mapping['resized_patch_coords'], fill=(0, 0, 0, 255))
                    if img_matrix[i, j] >= top10_per_value:
                        draw.rectangle(mapping['original_patch_coords'], outline="orange", width=4)
                        resized_draw.rectangle(mapping['resized_patch_coords'], outline="orange", width=4)
                mask_img.convert("RGB").save(
                    os.path.join(save_token_img_pth, f"target{args.target_id}_layer{layer_index}.jpg")
                )
                mask_resized_img.convert("RGB").save(
                    os.path.join(save_token_img_pth, f"resized_target{args.target_id}_layer{layer_index}.jpg")
                )

        print("\n" + "*"*80)
        print(f"sample_id={args.sample_id}")
        print(f"attn_index={int(attn_index)+1}")
        print(f"gt={args.gt}\ttarget_id={args.target_id}")
        print(f"image_num={args.imgnum}")
        print("*"*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--task_type", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/LLaVA-NeXT/attn_score_analysis/llava-ov-7B")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1e6)
    parser.add_argument("--decoder_ids", type=list, default=list(range(0,total_layer)))

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
    total_head = 28
    total_layer = 28

    cal_head_weights(parse_args())
