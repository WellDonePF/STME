from datasets import load_dataset

configs = [
    'action_sequence', 'moving_count', 'action_prediction', 'episodic_reasoning', 'action_antonym',
    'action_count', 'scene_transition', 'object_shuffle', 'object_existence', 'fine_grained_pose',
    'unexpected_action', 'moving_direction', 'state_change', 'object_interaction', 'character_order',
    'action_localization', 'counterfactual_inference', 'fine_grained_action', 'moving_attribute',
    'egocentric_navigation'
]
for conf in configs:
    dataset = load_dataset(
        "OpenGVLab/MVBench",
        conf,
        cache_dir="/data/nas/wangpengfei2/project/InternVL/internvl_chat/data/MVBench",
        split="train"
    )

