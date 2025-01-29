import os
import gc
import json
import time
import argparse

import torch
import accelerate
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def main(args):
    accelerator = accelerate.Accelerator()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        # device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    model.eval()
    badid = []

    # Load Data
    info = json.loads(open(args.data_path,'r',encoding="utf-8").read())
    all_res = []
    output_dir = os.path.join(args.save_result_path, f"pred_{accelerator.process_index}.json")
    model = accelerator.prepare(model)
    if accelerator.num_processes > 1:
        model = model.module

    print(f"accelerator.num_processes: {accelerator.num_processes}")
    print(f"accelerator.process_index: {accelerator.process_index}")
    cnt = -1
    for item in info:
        gc.collect(); torch.cuda.empty_cache()
        # if item["dataset_name"] != "CM" or item["sample_id"] not in [1189]:  # SlideVQA
        #     continue
        cnt += 1
        if cnt % accelerator.num_processes != accelerator.process_index:
            continue

        # 收集各种信息
        sample_id = item["sample_id"]
        context = item["context"]
        context = context.split("<ImageHere>")
        images = item["raw_img_list"]
        imgnum = len(images)
        dataset_name = item["dataset_name"]

        # make message
        messages = [ {"role": "user", "content": []} ]
        for ind in range(imgnum):
            if context[ind] != "":
                messages[0]["content"].extend([
                    {
                        "type": "text",
                        "text": context[ind]
                    }
                ])
            messages[0]["content"].extend([
                {
                    "type": "image",
                    "image": f"file://{images[ind]}"
                }
            ])
        if context[-1] != "":
            if dataset_name == "CM":
                question = context[-1]
            else:
                question = context[-1].rstrip("Your answer is: ")
                question = question + "Respond with the numbers of all images related to the correct answer in the format of list: [related image1 number, related image2 number,...]"
            messages[0]["content"].extend([
                {
                    "type": "text",
                    "text": question
                }
            ])

        # Preparation for inference
        if imgnum >= 15:
            processor.image_processor.max_pixels = 1280*28*28//5
        elif imgnum >= 8:
            processor.image_processor.max_pixels = 1280*28*28//4
        else:
            processor.image_processor.max_pixels = 1280*28*28//3
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(accelerator.device)

        # 推理并保存权重
        for st in args.save_attn:
            if st:
                tensor_path_qwen2vl = os.path.join(
                    args.save_tensor_path,
                    f'{item["dataset_name"]}_sample{item["sample_id"]}',
                    f'sample{item["sample_id"]}_layer{st[-1]}.pt'
                )
                if os.path.exists(tensor_path_qwen2vl):
                    continue
            model.config.save_attn = st
            try:
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print("\n" + "*"*80)
                print(f"fail: cnt: {cnt}\tdata_name: {dataset_name}\tsample_index: {sample_id}")
                print("ERROR:", e)
                print("*"*80 + "\n")
                result = None
                badid.append(str(sample_id))
                break

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        item.update({"answer": result})
        all_res.append(item)
        open(output_dir,'w',encoding="utf-8").write(json.dumps(all_res,indent=4))
        if result is not None:
            print("\n" + "*"*80)
            print(f"cnt: {cnt}\tdata_name: {dataset_name}\tsample_index: {sample_id}")
            print(f'gt: {item["response"]}')
            print(f'model answer: {result}')
            print(f"target_num: {item['target_id']}")
            print(f"image_num: {imgnum}")
            print("*"*80 + "\n")
    print(f"finish: rank {accelerator.process_index}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/Qwen2VL/attn_score_analysis/qwen2vl-7B")
    parser.add_argument("--task_type", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "flash_attention_2", "sdpa"])

    parser.add_argument("--attn_mode", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--min_pixels", type=int, default=256*28*28)
    parser.add_argument("--max_pixels", type=int, default=1280*28*28//5)

    args = parser.parse_args()
    args.data_path = os.path.join(
        "/cpfs/user/wangpengfei2/project/usage/attn_score_analysis/make_data",
        f"merged_result_{args.task_type}.json"
    )
    args.save_result_path = os.path.join(args.root_path, args.task_type, "answer_target")
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
    total_layer = 28
    main(parse_args())
