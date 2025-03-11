import os
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import json
import argparse

def load_janus():
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vlm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vlm = vlm.to(torch.bfloat16).cuda().eval()
    return vlm, tokenizer, vl_chat_processor

def load_slideVQA_images(slideVQA_base_dir:str = "SlideVQA", split:str = "test"):
    split_dir = os.path.join(slideVQA_base_dir, "images", split)
    decks = []
        
    # 숫자 기준 정렬 함수
    def extract_number(file_name):
        return int(file_name.split("-")[-2])

    for deck in os.listdir(split_dir):
        deck_dir = os.path.join(split_dir, deck)
        image_paths = []
        image_names = os.listdir(deck_dir)
        
        # 정렬 실행
        image_names = sorted(image_names, key=extract_number)
        
        for image_name in image_names:
            # image_names.append(image_name)
            image_paths.append(os.path.join(deck_dir, image_name))
        decks.append({
            "deck_name": deck,
            "deck_dir": deck_dir,
            "image_names": image_names,
            "image_paths": image_paths,
        })
    return decks

def load_slideVQA_annotations(decks, slideVQA_base_dir:str = "SlideVQA", split = "test"):
    jsonl_path = os.path.join(slideVQA_base_dir, "annotations", "qa", f"{split}_filtered.jsonl")

    # jsonl에서 json을 하나씩 읽어오면서 deck list에 있으면 추가
    queries = []
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            cnt+=1
            data = json.loads(line)
            if data["deck_name"] in decks:
                queries.append(data)
                
    return queries
          
def load_index(index_path, skip_1st=False):
    deck_jsons = os.listdir(index_path)
    tag_dict = {} # key: tag 이름, value: tag에 해당하는 deck의 index들 list
    deck_dict = {} # key: deck 이름, value: deck에 포함된 모든 tag들 list
    decks = [] # deck name list -> tag_dict와의 매칭을 위해 필요함. 여기서의 순서가 deck의 index
    deck_indices = {} # key: deck 이름, value: deck 번호 -> tag_dict와의 매칭을 위해 필요함
    deck_1st_names = []
    
    for deck_idx, deck in enumerate(deck_jsons):    
        deck_path = os.path.join(index_path, deck)
        
        # 1. deck name 저장
        deck_name = os.path.splitext(deck)[0]
        decks.append(deck_name)
        
        # 2. deck data 로드
        with open(deck_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if skip_1st and deck_idx == 0:
            deck_1st_names.append(data["slide_datas"][0]["image_name"])
            data["slide_datas"] = data["slide_datas"][1:]
        
        tags = []
        
        # 3. 모든 tag를 tag_dict에 추가
        def add_tag(tag):
            tag = str(tag)
            tag = f'{tag}'
            tag = tag.lower()
            if tag not in tag_dict:
                tag_dict[tag] = [] 
            if deck_idx not in tag_dict[tag]:
                tag_dict[tag].append(deck_idx)
            if tag not in tags:
                tags.append(tag)
        
        for tag in data["deck_data"]["metadata"]["tags"]:
            add_tag(tag)
        
        for slide_data in data["slide_datas"]:
            for tag in slide_data["metadata"]["tags"]:
                add_tag(tag)  
                
        deck_dict[deck_name] = tags  
        deck_indices[deck_name] = deck_idx
        
    return tag_dict, deck_dict, decks, deck_indices

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")