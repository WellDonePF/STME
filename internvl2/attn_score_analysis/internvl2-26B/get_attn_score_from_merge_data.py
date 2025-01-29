import sys
sys.path.append('/cpfs/user/wangpengfei2/project/InternVL/internvl_chat')

import os
import gc
import json
import time
import argparse
from PIL import Image

import torch
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        max_length=args.max_input_length
    )
    model = InternVLChatModel.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).eval()
    # model = model.to(torch.device(f"cuda:{args.local_process_index}"))
    model = model.to("cuda")
    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=1,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
    )
    args.image_size = model.config.force_image_size or model.config.vision_config.image_size
    args.use_thumbnail = model.config.use_thumbnail
    image_transform = build_transform(is_train=False, input_size=args.image_size)
    badid = []

    # Load Data
    info = json.loads(open(args.data_path,'r',encoding="utf-8").read())
    new_info = []
    for item in info:
        score_path = os.path.join(
            args.root_path, args.task_type, "scores",
            f'{item["dataset_name"]}_sample{item["sample_id"]}_imgnum{len(item["raw_img_list"])}_target{item["target_id"]}_scores.xlsx'
        )
        if not os.path.exists(score_path):
            new_info.append(item)
    info = new_info
    all_res = []
    output_dir = os.path.join(args.save_result_path, f"pred_{args.process_index}.json")
    max_tensors_num = args.num_processes + 2

    print(f"max_tensors_num: {max_tensors_num}")
    print(f"num_processes: {args.num_processes}")
    print(f"process_index: {args.process_index}")
    print(f"dataset number: {len(info)}")
    cnt = -1
    for item in info:
        gc.collect(); torch.cuda.empty_cache()
        # if item["dataset_name"] != "DocVQA" or item["sample_id"] not in [69]:
        #     continue
        cnt += 1
        if cnt % args.num_processes != args.process_index:
            continue

        sleep_cnt = 0
        while len(os.listdir(args.save_tensor_path)) >= max_tensors_num:
            time.sleep(10)
            sleep_cnt += 1
            if sleep_cnt == 160:
                return

        sample_id = item["sample_id"]
        dataset_name = item["dataset_name"]
        result = None
        context = item["context"]
        context = context.replace("<ImageHere>", "<image>")
        if dataset_name != "CM":
            context = context.rstrip("Your answer is: ")
            context = context + "\nRespond with the answer and your thought process for reasoning."
        images = item["raw_img_list"]
        images = [Image.open(img) for img in images]
        imgnum = len(images)
        if imgnum > 8:
            args.max_num = 1
            args.dynamic = False
        elif imgnum > 5:
            args.max_num = 2
        elif imgnum > 4:
            args.max_num = 3
        elif imgnum > 3:
            args.max_num = 4

        # 动态分辨率，将图片切成子图
        num_patches_list = []
        pixel_values = []
        for img in images:
            if args.dynamic:
                patches = dynamic_preprocess(
                    image=img,
                    max_num=args.max_num,
                    image_size=args.image_size,
                    use_thumbnail=args.use_thumbnail
                )
            else:
                patches = [img]
            num_patches_list.append(len(patches))
            pixel_values.extend([image_transform(patch) for patch in patches])
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(device=model.device, dtype=torch.bfloat16)

        for st in args.save_attn:
            if st:
                tensor_path_mantis = os.path.join(
                    args.save_tensor_path,
                    f'{item["dataset_name"]}_sample{item["sample_id"]}',
                    f'sample{item["sample_id"]}_layer{st[-1]}.pt'
                )
                if os.path.exists(tensor_path_mantis):
                    continue
            model.language_model.model.config.save_attn = st

            try:
                result = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    question=context,
                    generation_config=generation_config,
                    verbose=True
                )
            except Exception as e:
                print("\n" + "*"*80)
                print(f"fail: cnt: {cnt}\tdata_name: {dataset_name}\tsample_index: {sample_id}")
                print("ERROR:", e)
                print("*"*80 + "\n")
                badid.append(str(sample_id))
                break

            if st:
                save_tensor_path_new = os.path.join(args.save_tensor_path, f"{dataset_name}_sample{sample_id}")
                os.makedirs(save_tensor_path_new, exist_ok=True)
                for layer_index in st:
                    if "InternVL2-2B" in args.model_name_or_path or "InternVL2-8B" in args.model_name_or_path or "InternVL2-26B" in args.model_name_or_path:
                        attn_score_matrix = model.language_model.model.layers[layer_index].attention
                    else:
                        attn_score_matrix = model.language_model.model.layers[layer_index].self_attn
                    torch.save(
                        attn_score_matrix.full_attention_score.to(torch.float16),
                        os.path.join(save_tensor_path_new, f"sample{sample_id}_layer{layer_index}.pt")
                    )
                    del attn_score_matrix.full_attention_score
                    attn_score_matrix.full_attention_score=None
        item.update({"answer": result})
        all_res.append(item)
        open(output_dir,'w',encoding="utf-8").write(json.dumps(all_res,indent=4))
        del pixel_values, num_patches_list, context

        if result is not None:
            print("\n" + "*"*80)
            print(f"cnt: {cnt}\tdata_name: {dataset_name}\tsample_index: {sample_id}")
            print(f'gt: {item["response"]}')
            print(f"image_num: {imgnum}")
            print(f'model answer: {result}')
            print("*"*80 + "\n")
    print(f"finish: rank {args.process_index}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/cpfs/user/wangpengfei2/project/InternVL/pretrained/InternVL2-26B")
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/InternVL/attn_score_analysis/internvl2-26B")
    parser.add_argument("--task_type", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument('--dynamic', action='store_true')

    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--local_process_index", type=int, default=0)
    parser.add_argument("--process_index", type=int, default=0)
    parser.add_argument("--attn_mode", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max_input_length", type=int, default=32*1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_num", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument('--num_beams', type=int, default=1)
    args = parser.parse_args()

    args.dynamic = True
    args.data_path = os.path.join(
        "/cpfs/user/wangpengfei2/project/usage/attn_score_analysis/make_data",
        f"merged_result_{args.task_type}.json"
    )
    args.save_tensor_path = os.path.join(args.root_path, args.task_type, "tensors")
    os.makedirs(args.save_tensor_path, exist_ok=True)
    args.save_result_path = os.path.join(args.root_path, args.task_type, "bench_res")
    pth_cnt = 0
    while os.path.exists(args.save_result_path + str(pth_cnt)):
        pth_cnt += 1
    args.save_result_path = args.save_result_path + str(pth_cnt)
    os.makedirs(args.save_result_path, exist_ok=True)

    if args.attn_mode == 1:
        args.save_attn = [[]]
    elif args.attn_mode == 2:
        args.save_attn = [
            list(range(0, int(total_layer//3))),
            list(range(total_layer//3, total_layer*2//3)),
            list(range(total_layer*2//3, total_layer))
        ]
    elif args.attn_mode == 3:
        args.save_attn = [list(range(0, total_layer))]

    return args


if __name__ == "__main__":
    total_layer = 48
    main(parse_args())
