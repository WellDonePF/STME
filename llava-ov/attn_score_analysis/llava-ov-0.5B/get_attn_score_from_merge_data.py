import os
import gc
import json
import time
import argparse
from PIL import Image
import copy
import torch
import accelerate
import torch.distributed

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def main(args):
    accelerator = accelerate.Accelerator()
    llava_model_args = {"multimodal": True}
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.pretrained_model_name_or_path,
        model_base=None,
        model_name=args.model_name,
        device_map=None,
        torch_dtype="bfloat16",
        attn_implementation=args.attn_implementation,
        **llava_model_args
    )
    model.eval()
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
    output_dir = os.path.join(args.save_result_path, f"pred_{accelerator.process_index}.json")
    model = accelerator.prepare(model)
    if accelerator.num_processes > 1:
        model = model.module
        max_tensors_num = torch.distributed.get_world_size() + 2
    else:
        max_tensors_num = 100

    print(f"max_tensors_num: {max_tensors_num}")
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

        sleep_cnt = 0
        while len(os.listdir(args.save_tensor_path)) >= max_tensors_num:
            time.sleep(10)
            sleep_cnt += 1
            if sleep_cnt == 160:
                return

        # 收集各种信息
        sample_id = item["sample_id"]
        dataset_name = item["dataset_name"]
        result = None
        context = item["context"]
        context = context.replace("<ImageHere>", DEFAULT_IMAGE_TOKEN)
        if dataset_name != "CM":
            context = context.rstrip("Your answer is: ")
            context = context + "\nRespond with the answer and your thought process for reasoning."
        images = item["raw_img_list"]
        images = [Image.open(img).convert("RGB") for img in images]
        images = process_images(images, image_processor, model.config)
        images = [img.to(dtype=torch.float16, device=accelerator.device) for img in images]
        image_sizes = [img.size for img in images]
        imgnum = len(images)

        for st in args.save_attn:
            if st:
                tensor_path = os.path.join(
                    args.save_tensor_path,
                    f'{item["dataset_name"]}_sample{item["sample_id"]}',
                    f'sample{item["sample_id"]}_layer{st[-1]}.pt'
                )
                if os.path.exists(tensor_path):
                    continue
            model.config.save_attn = st

            conv = copy.deepcopy(conv_templates[args.conv_template])
            conv.append_message(conv.roles[0], context)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(accelerator.device)

            # Generate response
            try:
                cont = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    # do_sample=True if args.temperature > 0 else False,
                    do_sample=False,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                result = tokenizer.batch_decode(cont, skip_special_tokens=True)
                result = result[0]
            except Exception as e:
                print("\n" + "*"*80)
                print(f"fail: cnt: {cnt}\tdata_name: {dataset_name}\tsample_index: {sample_id}")
                print("ERROR:", e)
                print("*"*80 + "\n")
                badid.append(str(sample_id))
                continue

            if st:
                save_tensor_path_new = os.path.join(args.save_tensor_path, f"{dataset_name}_sample{sample_id}")
                os.makedirs(save_tensor_path_new, exist_ok=True)
                for layer_index in st:
                    torch.save(
                        model.model.layers[layer_index].self_attn.full_attention_score.to(torch.float16),
                        os.path.join(save_tensor_path_new, f"sample{sample_id}_layer{layer_index}.pt")
                    )
                    del model.model.layers[layer_index].self_attn.full_attention_score
                    model.model.layers[layer_index].self_attn.full_attention_score=None
        item.update({"answer": result})
        all_res.append(item)
        open(output_dir,'w',encoding="utf-8").write(json.dumps(all_res,indent=4))
        del input_ids, conv

        if result is not None:
            print("\n" + "*"*80)
            print(f"cnt: {cnt}\tdata_name: {dataset_name}\tsample_index: {sample_id}")
            print(f'gt: {item["response"]}')
            print(f"image_num: {imgnum}")
            print(f'model answer: {result}')
            print("*"*80 + "\n")
    print(f"finish: rank {accelerator.process_index}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--task_type", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--root_path", type=str, default="/cpfs/user/wangpengfei2/project/LLaVA-NeXT/attn_score_analysis/llava-ov-0.5B")
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "flash_attention_2", "sdpa"])

    parser.add_argument("--attn_mode", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

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
    total_layer = 24
    main(parse_args())
