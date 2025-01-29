import re
import gc
from rouge import Rouge
import argparse
import os
import json
import numpy as np


class Eval:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index):
        if index < 26:
            return chr(index+65)
        elif index < 52:
            return 'A'+chr(index+65-26)
        else:
            return 'B'+chr(index+65-26-26)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip().lower()
        return answer

    def get_image_quantity_level(self, sample):
        # 2-5 6-31 32-109
        image_num = len(sample['image'])
        if image_num < 6:
            return 'Few'
        elif image_num > 31:
            return 'Many'
        else:
            return 'Medium'

    def evaluate_rouge(self, predictions, core_json):
        # get image_quantity_level
        if len(predictions) != len(core_json['data']):
            raise ValueError(f'There is prediction absent.')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
            new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'image_quantity_level' in pre.keys()

        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, res in enumerate(predictions):
            sample_id = res['sample_id']
            gt_ans = self.process(res["response"])
            pred_ans = self.process(res["answer"])
            assert gt_ans != ''
            if pred_ans == '':
                score = 0
            else:
                score = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(score)
            image_quantity_level_cnt[self.get_image_quantity_level(res)].append(score)
            eval_list.append({'id':str(sample_id),'score':str(round(score,3))})
        return {
            'Rouge-L f': np.mean(acc['f']),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list

    def match_choice(self, text, option):
        '''Return: A B C D...'''

        def preprocess_option_string(option_string):
            # First, preprocess the option text to normalize it
            processed_option = self.process(option_string)

            # Then, escape any special regex characters in the processed option text
            # List of regex special characters that need to be escaped
            special_chars = ["\\", ".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "|", "(", ")"]
            # Escape the special characters by prefixing them with a backslash
            for char in special_chars:
                if char in processed_option:
                    processed_option = processed_option.replace(char, "\\" + char)
            # escaped_option = escape_special_chars(processed_option)
            return processed_option
            
        if text == "":
            return 'C'
        try:
            # Maybe start from the head
            # 1. Char+Choice: `A. Blastomycosis`
            option_str = "|".join([preprocess_option_string(f"{k} {v}")for k,v in option.items()])
            option_pattern = rf'({option_str})'
            option_res = re.search(option_pattern, text, re.S)   # NOTE we dont use match_all
            if option_res:
                return (option_res.group(0)[0]).upper()

            # 2. Choice: `Blastomycosis`
            option_str = "|".join([preprocess_option_string(v).replace(' ', '') for k,v in option.items()])
            option_pattern = rf'({option_str})'
            option_res = re.search(option_pattern, text.replace(' ', ''), re.S)   # NOTE we dont use match_all
            if option_res:
                for k, v in option.items():
                    if option_res[0].strip() == preprocess_option_string(v).replace(' ', ''):
                        return k.upper()
            
            # 3. Char: `A` `AB`
            if len(text) in [1,2] and text.upper() in option.keys():
                return text.upper()

            # use gpt extract

        except Exception as e:
            print(f"something wrong during match_choice {text}: {e}")
            return text
        return "".join([i.upper() for i in text if i.upper() in option])

    def judge_multi_choice(self, sample):
        sample_id = sample['sample_id']
        gt_ans = sample["response"]
        pred_ans = sample["answer"]
        choice_list = sample['choice_list']
        assert gt_ans in choice_list
        # Convert choice_list to a dictionary format expected by match_choice
        option_dict = {self.char(i): choice for i, choice in enumerate(choice_list)}

        # Use match_choice to determine the selected answer from pred_ans
        selected_answer = self.match_choice(pred_ans, option_dict)

        # Check if the selected answer matches the ground truth
        gt_ans_chr = self.char(choice_list.index(sample["response"]))
        if selected_answer == gt_ans_chr:
            return 1, selected_answer
        else:
            return 0, selected_answer

    def process_sample(self, sample):
        sample["answer"] = self.process(sample["answer"])
        sample["response"] = self.process(sample["response"])
        for i in range(len(sample['choice_list'])):
            sample["choice_list"][i] = self.process(sample["choice_list"][i])

    def evaluate_multichoice(self, predictions, core_json):
        '''
        predictions: raw prediction file output by models
        '''
        # get choice_list & image_quantity_level
        # if len(predictions) != len(core_json['data']):
        #     raise ValueError(f'There is prediction absent. {len(predictions)}!={len(core_json["data"])}')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            if int(sample['sample_id']) in new_pres:
                new_pres[int(sample['sample_id'])]['choice_list'] = sample['task_instance']['choice_list']
                new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
                new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'choice_list' in pre.keys()
            assert 'image_quantity_level' in pre.keys()
        
        correct = 0
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, sample in enumerate(predictions):
            # Process string
            self.process_sample(sample)
            # Score
            
            score, extracted_answer = self.judge_multi_choice(sample)
            sample['extracted'] = extracted_answer
            sample['result'] = score
            eval_list.append({'id':str(sample['sample_id']), 'score': str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(score)
        return predictions, {
            'Accuracy': correct/len(predictions),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list

    def evaluate_needle(self, predictions, core_json, needle=True):
        # get choice_list & image_quantity_level
        # if len(predictions) != len(core_json['data']):
        #     raise ValueError(f'There is prediction absent. {len(predictions)}!={len(core_json["data"])}')
        new_pres = {d['sample_id']: d for d in predictions}
        for sample in core_json['data']:
            if int(sample['sample_id']) in new_pres:
                new_pres[int(sample['sample_id'])]['image_quantity_level'] = sample['image_quantity_level']
                new_pres[int(sample['sample_id'])]['image'] = sample['task_instance']['images_path']
        for pre in new_pres.values():
            assert 'image_quantity_level' in pre.keys()
        
        correct = 0
        eval_list = []
        image_quantity_level_cnt = {'Few': [], 'Medium': [], 'Many': []}
        for i, sample in enumerate(predictions):
            # Process string
            sample_id = sample['sample_id']
            gt_ans = self.process(sample["response"])
            pred_ans = self.process(sample["answer"])
            
            # Score
            if needle:
                score = 1 if gt_ans in pred_ans.split() else 0
            else:
                score = 1 if gt_ans in pred_ans else 0

            sample['result'] = score
            eval_list.append({'id':str(sample['sample_id']), 'score': str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(score)
        return {
            'Accuracy': correct/len(predictions),
            'image_quantity_level-Accuracy': {k: np.mean(v) if len(v)!=0 else 0 for k, v in image_quantity_level_cnt.items()},
            'image_quantity_level-Result': {k: [sum(v), len(v)] for k, v in image_quantity_level_cnt.items()}}, eval_list


def campare_result(model_answer: str, groundtruth):
    if "no match" in model_answer.lower():
        return 0
    matches = re.findall(r'image\s+(\d+)', model_answer.lower())
    if len(matches) > 0:
        print("muti choice image number 1")
        return int(int(matches[0]) == int(groundtruth))

    numbers = re.findall(r'-?\d+\.?\d+', model_answer)
    if len(numbers) > 0:
        print("muti choice image number 2")
        return int(int(numbers[0]) == int(groundtruth))
    return 0


def main(args):
    output_dir = args.output_dir
    all_preds = json.loads(open(args.result_dir,'r',encoding="utf-8").read())
    all_eval_result = []
    all_eval_list = []
    assert all_preds and len(all_preds) != 0

    # eval milebench
    for dataset in ["CharacterOrder", "DocVQA", "GPR1200", "ImageNeedleInAHaystack", "SlideVQA", "TQA", "nuscenes"]:
        # Get scores
        preds = [x for x in all_preds if x["dataset_name"] == dataset]
        if len(preds) == 0:
            continue
        core_annotation = json.load(open(os.path.join(args.data_dir, dataset, f'{dataset}-adv.json' if args.adv else f'{dataset}.json')))
        question_type = core_annotation['meta_data']['question_type']
        args.dataset = dataset
        scorer = Eval()
        if 'NeedleInAHaystack' in dataset or 'MMCoQA' in dataset:
            eval_result, eval_list = \
                scorer.evaluate_needle(preds, core_annotation, needle='NeedleInAHaystack' in dataset)
        elif question_type == 'open-ended':
            eval_result,eval_list = scorer.evaluate_rouge(preds, core_annotation)
        elif question_type == 'multi-choice':
            predictions_with_extracted_answers, eval_result, eval_list = scorer.evaluate_multichoice(preds, core_annotation)
            json.dump(predictions_with_extracted_answers, open(os.path.join(output_dir, 'pred_with_extracted.json'),'w'), indent=4)
        else:
            raise ValueError('Dataset not supported')

        print(f"{dataset}:  {eval_result}")
        new_eval_list = []
        for item in eval_list:
            new_eval_list.append({"dataset_name": dataset, "sample_id": int(item["id"]), "score": int(item["score"])})
        eval_list = new_eval_list
        all_eval_result += eval_result
        all_eval_list += eval_list

        gc.collect()
    
    for dataset in ["CM"]:
        preds = [x for x in all_preds if x["dataset_name"] == dataset]
        if len(preds) == 0:
            continue
        eval_list = []
        for item in preds:
            is_right = campare_result(model_answer=item["answer"], groundtruth=item["response"])
            eval_list.append({"dataset_name": dataset, "sample_id": int(item["sample_id"]), "score": int(is_right)})
        all_eval_list += eval_list
    json.dump(all_eval_list, open(os.path.join(output_dir, 'eval_score.json'),'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/cpfs/user/wangpengfei2/data/vlmbenchmark/MileBench")
    parser.add_argument("--task_type", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument('--root_path', type=str, default="/cpfs/user/wangpengfei2/project/LLaVA-NeXT/attn_score_analysis/llava-ov-0.5B")
    args = parser.parse_args()

    args.result_dir = os.path.join(args.root_path, args.task_type, "info/pred.json")
    args.output_dir = os.path.dirname(args.result_dir)
    args.adv = False

    main(args)
