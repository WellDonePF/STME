import os
import json
import re

def main():
    easy_res_path = "/cpfs/user/wangpengfei2/project/Qwen2VL/attn_score_analysis/qwen2vl-7B/easy/answer_target/pred_0.json"
    easy_res = json.loads(open(easy_res_path,'r',encoding="utf-8").read())
    rig = 0
    for item in easy_res:
        sample_id = item["sample_id"]
        answer = item["answer"]
        target_id = str(item["target_id"])
        dataset_name = item["dataset_name"]
        if target_id in answer:
            rig += 1
    print(rig / len(easy_res))

    med_res_path = "/cpfs/user/wangpengfei2/project/Qwen2VL/attn_score_analysis/qwen2vl-7B/medium/answer_target/pred_0.json"
    med_res = json.loads(open(med_res_path,'r',encoding="utf-8").read())
    rig = 0
    for item in med_res:
        sample_id = item["sample_id"]
        answer = item["answer"]
        target_id = str(item["target_id"])
        dataset_name = item["dataset_name"]
        if target_id in answer:
            rig += 1
    print(rig / len(med_res))



if __name__ == "__main__":
    main()
