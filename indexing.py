import os
import torch
from  tqdm import tqdm
from janus.utils.io import load_pil_images
import json
from transformers.generation import GenerationConfig
from utils import load_janus, load_slideVQA_images, str2bool
from json_repair import repair_json
import argparse

json_prefix = '```json\n{'

json_schema_slide = """```json
{
  "title": "PLACEHOLDER_TITLE",
  "content": "PLACEHOLDER_CONTENT",
  "metadata": {
    "tags": ["PLACEHOLDER_TAG"],  # Provide as many relevant tags as possible
    "tl;dr": "PLACEHOLDER_SUMMARY",
    "has_figures": PLACEHOLDER_BOOLEAN,  # True or False
    "has_equation": PLACEHOLDER_BOOLEAN  # True or False
  },
  "figures": [
    {
      "figure_description": "PLACEHOLDER_FIGURE_DESCRIPTION",
      "figure_tags": ["PLACEHOLDER_FIGURE_TAG"],  # Provide as many relevant tags as possible
      "tl;dr": "PLACEHOLDER_FIGURE_SUMMARY"
    }
  ],
  "equations": [
    {
      "equation_description": "PLACEHOLDER_EQUATION_DESCRIPTION",
      "latex": "PLACEHOLDER_LATEX"
    }
  ]
}
```"""

json_schema_deck = """```json
{
  "title": "PLACEHOLDER_TITLE",
  "content": "PLACEHOLDER_CONTENT",
  "metadata": {
    "tags": ["PLACEHOLDER_TAG"],  # Provide as many relevant tags as possible
    "tl;dr": "PLACEHOLDER_SUMMARY"
    "has_figures": PLACEHOLDER_BOOLEAN,  # True or False
    "has_equation": PLACEHOLDER_BOOLEAN  # True or False
  }
}
```"""

prompt_slide = """Fill in the following JSON structure based on the given context.
- The `"title"` should be a concise and descriptive heading.
- The `"content"` should be a well-structured, informative explanation.
- The `"metadata.tags"` should include relevant keywords.
- The `"metadata.tl;dr"` should provide a short summary of the content.
- The `"has_figures"` and `"has_equation"` should be `true` if applicable, otherwise `false`.
- `"figures"` and `"equations"` should contain detailed descriptions if relevant.
JSON Structure:
"""

prompt_deck = """Summarize the content based on the provided JSON schema.  
- `"title"` should be the main topic of the document.
- `"content"` should include an overview of the topic.
- `"metadata.tags"` should be relevant to the subject.
- `"metadata.tl;dr"` should provide a concise summary.
JSON Structure:
"""

prompt_deck_pre = """I am about to show you 20 JSON objects representing individual slides from a PowerPoint presentation.  
Each JSON object follows the structure defined in the schema below.  

Please analyze the provided JSON data carefully, as it contains structured information about each slide, including its title, content, metadata, figures, and equations.
"""

def check_slide_json(data):
    # print(data)
    # json 무결성 검사
    if "title" not in data:
        data["title"] = "Err"
    if "content" not in data:
        data["content"] = "Err"
    if "metadata" not in data:
        data["metadata"] = {}
    if "figures" not in data:
        data["figures"] = []
    if "equations" not in data:
        data["equations"] = []
    # metadata 내부 확인
    if "tags" not in data["metadata"]:
        data["metadata"]["tags"] = []
    if "tl;dr" not in data["metadata"]:
        data["metadata"]["tl;dr"] = "Err"
    if "has_figures" not in data["metadata"]:
        data["metadata"]["has_figures"] = False
    if "has_equation" not in data["metadata"]:
        data["metadata"]["has_equation"] = False 
    return data

def check_deck_json(data):
    # json 무결성 검사
    # print(data)
    if "title" not in data:
        data["title"] = "Err"
    if "content" not in data:
        data["content"] = "Err"
    if "metadata" not in data:
        data["metadata"] = {}
    # metadata 내부 확인
    if "tags" not in data["metadata"]:
        data["metadata"]["tags"] = []
    if "tl;dr" not in data["metadata"]:
        data["metadata"]["tl;dr"] = "Err"
        
    return data

def extract_data_from_slide_image(image_path, vlm, tokenizer, vl_chat_processor):
    # 1. Conversation 생성
    conversation = []
    # 1.1. 첫 번째 conversation은 그냥 이미지를 넘겨준다.
    conversation.append(
    {
        "role": "<|User|>",
        "content": prompt_deck_pre,
    })
    conversation.append(
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n",
        "images": [image_path],
    })
    # 1.2. 두 번째 conversation은 json 스키마를 채워달라는 instruction과 json 스키마를 전달한다.
    conversation.append(
    {
        "role": "<|User|>",
        "content": f"{prompt_slide}\n{json_schema_slide}",
    })
    # 1.3. 세 번째 conversation은 assistant가 답변하는 양식의 앞부분을 넣어줌 -> 이래야 잘됨.
    conversation.append({"role": "<|Assistant|>", "content": f'{json_prefix}'},)        

    # 2 Input 전처리
    # 2.1. conversation 내에 이미지가 있으면, pil 이미지 리스트로 로드해줌. 직접 로드해도 상관없음.
    pil_images = load_pil_images(conversation) 
    # 2.2. 이미지가 있는 경우, vl_chat_procesdsor로 전처리 필요
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vlm.device) 
    # 2.3. input에 맞는 embedding으로 변환
    inputs_embeds = vlm.prepare_inputs_embeds(**prepare_inputs)

    # 3. VLM inference -> 여기는 쉬움    
    outputs = vlm.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    # 3.2. decoding
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # 4. Json 파싱
    # 4.1. Markdown json prefix를 붙여줌. -> 사실 없어도 되는데 디버깅 때 보기 편하려고 넣어둠
    answer = json_prefix + answer
    # 4.2. Markdown json prefix와 뒤의 \n``` 제거 -> json string만 남김
    answer = answer[8:-4]
    # 4.3. Json 포맷을 어기는 경우가 종종 발생함. 따라서, json_repair 패키지를 이용해서 json 틀린 부분 수정
    try:
        fixed_json = repair_json(answer)
        # 4.4.json string으로부터 dictionary 생성
        data = json.loads(fixed_json) 
    except Exception as e:
        print("failed to parse json")
        print(answer)
        data = {}
        
    # 4.5. 4.3을 해도 json 스키마를 벗어나는 경우는 해결 못함. 특히, json이 딕셔너리가 아니라 리스트를 생성하는 경우 오류 발생. 이 경우는 그냥 기본 json format으로 변환
    # TODO: 현재는 오류가 슬라이드를 포기하는 전략 -> 수정해야함
    try:
        data = check_slide_json(data)
    except Exception as e:
        print("failed to parse json")
        print(answer)
        data = {}
        data = check_slide_json(data)
    
    # 5. 생성한 dictionary 데이터 
    return data

json_schema_tag_refine = """```json
{
  "title": @,
  "content": @,
  "metadata": {
    "tags": [@],  # As many as possible
    "tl;dr": @
}
```"""

def refine_tag():
    # Image와 Tag를 보여주고, tag가 잘 나왔는지 체크    
    return    

# 20장의 slide 데이터로부터 전체를 요약하는 json을 생성함 -> 아직까지는 크게 유용하지는 않음.
def extract_data_from_overall_data(datas, vlm, tokenizer, vl_chat_processor, refine=True):
    # 1. 20장의 slide 데이터를 하나의 json으로 연결
    deck_info = []
    for data in datas:
        deck_info.append(
            {
                "title": data["title"],
                "content": data["content"],
                "metadata": data["metadata"]
            }
        )
    deck_info = json.dumps(deck_info)

    # 2. Conversation 생성
    conversation = []
    # 2.1. 첫 번째 conversation은 markdown json 형태로 json을 전달한다.
    conversation.append(
    {
        "role": "<|User|>",
        "content": "```json\n "+ deck_info + "\n```",
    })
    # 2.2 두 번째 conversation은 앞에서 전달받은 json을 바탕으로 overall data json 스키마에 따라 요약을 생성하도록 함.
    # TODO: 지금 프롬프는 요약해달라는 내용이 없음. 내용 추가 필요 
    conversation.append(
    {
        "role": "<|User|>",
        "content": f"{prompt_deck}\n{json_schema_deck}",
    })
    conversation.append({"role": "<|Assistant|>", "content": f'{json_prefix}'},)        

    # 3. Input 전처리
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vlm.device)
    inputs_embeds = vlm.prepare_inputs_embeds(**prepare_inputs)

    # 4. VLM inference
    outputs = vlm.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        return_legacy_cache=True,
        do_sample=False,
        use_cache=True,
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # 5. Json 파싱
    # 5.1. json 형식으로 변환
    answer = json_prefix + answer
    answer = answer[8:-4]
    # 5.2. json 오류 보정 (extract_data_from_overall_data 함수에서와 동일)
    try:
        fixed_json = repair_json(answer)
        data = json.loads(fixed_json)
    except Exception as e:
        print("failed to parse json")
        data = {}
    # TODO: 현재는 오류가 슬라이드를 포기하는 전략 -> 수정해야함
    try:
        data = check_deck_json(data)
    except Exception as e:
        print("failed to parse json")
        data = {}
        data = check_deck_json(data)
    return data
 
# 생성된 
def indexing(decks, vlm, tokenizer, vl_chat_processor, out_dir="output/indexing", skip_1st_page=False, subset=-1, overwrite=True):
    os.makedirs(out_dir, exist_ok=True)
    
    for deck_id, deck in tqdm(enumerate(decks), total=len(decks)):  
        deck_name = deck["deck_name"]  
        output_json_path = os.path.join(out_dir, f"{deck_name}.json")
        if not overwrite and os.path.exists(output_json_path):
            # print("pass")
            continue
        
        # 1. 각 슬라이드 파싱해서 dict 생성
        slide_datas = []
        for i in range(20):
            if skip_1st_page and i==0:
                continue
            data = extract_data_from_slide_image(deck["image_paths"][i], vlm, tokenizer, vl_chat_processor)
            data["image_name"] = deck["image_names"][i]
            slide_datas.append(data)
            
        # 2. deck에 대한 데이터 생성
        # deck 데이터 내부에 1에서 생성한 dict들 포함
        deck_data = extract_data_from_overall_data(slide_datas, vlm, tokenizer, vl_chat_processor)
        data = {
            "deck_name": deck["deck_name"],
            "deck_data": deck_data,
            "slide_datas": slide_datas,
        }
        
        # 3. 각 deck별로 하나의 json으로 저장
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        # 4. 사전에 지정한 subset 까지만 indexing -> 디버깅용
        if deck_id + 1 == subset:
            break
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_1st_page", type=str2bool, default=False)
    parser.add_argument("--out_dir", type=str, default="output/indexing")
    parser.add_argument("--dataset_base_dir", type=str, default="SlideVQA") 
    args = parser.parse_args()
    
    vlm, tokenizer, vl_chat_processor = load_janus() # 1. janus 모델 로드
    decks = load_slideVQA_images() # 2.slideVQA 데이터셋 로드
    indexing(decks, vlm, tokenizer, vl_chat_processor, out_dir=args.out_dir, skip_1st_page=args.skip_1st_page) # 3. 인덱싱 -> 덱별로 json 저장
    