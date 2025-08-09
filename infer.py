import json
import os

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration)


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        raise ValueError()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if fps > 0:
        return int(frame_count / fps)
    else:
        print("无法获取帧率")
        raise ValueError()


def inference(
    video_path,
    prompt,
    max_new_tokens=256,
    total_pixels=20480 * 28 * 28,
    min_pixels=48 * 28 * 28,
):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "video": video_path,
                    "fps": 1.0,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                },
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], return_video_kwargs=True
    )
    fps_inputs = video_kwargs["fps"]
    duration = get_video_duration(video_path)
    print("video duration: ", duration)
    print("video input: ", video_inputs[0].shape)
    print("video fps input: ", fps_inputs)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print(
        "num of video tokens: ",
        int(num_frames / 2 * resized_height / 28 * resized_width / 28),
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


PROMPT = """
You are provided with a creative advertisement video. Please watch the video carefully and answer the given question based on its visual content and inferential reasoning. Limit your response for each question to no more than 30 words.

Question: {}
"""

model_dir = "../../VG-RS/models/Qwen--Qwen25-VL-72B-Instruct/"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir)

video_dir = "../data/adsqa_video_collection/"
video_list = sorted(os.listdir(video_dir))

with open("../data/adsqa_question_file.json", "r") as f:
    ads_qa_list = json.load(f)

results = []
for i, ads_qa in enumerate(ads_qa_list):
    print("==================================================")
    print(f"Processing {i}/{len(ads_qa_list)} item ...")

    video_id = ads_qa["video"]
    question = ads_qa["question"]
    question_id = ads_qa["question_id"]

    video_path = os.path.join(video_dir, video_id + ".mp4")
    assert os.path.isfile(video_path)

    text_prompt = PROMPT.format(question).strip()
    answer = inference(video_path, text_prompt)
    results.append({"question_id": question_id, "question": question, "answer": answer})
    print(f"answer: ", answer)
    print("=============================================")
    torch.cuda.empty_cache()

with open("submission.json", "w") as f:
    json.dump(results, f)
