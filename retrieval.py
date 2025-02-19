import os
import json
from trie import Trie
from utils import load_janus, load_slideVQA_annotations, load_index
import torch
from tqdm import tqdm
from collections import Counter

# trie 만들기
def build_trie(vlm, tokenizer, vl_chat_processor):
    # Keyword들을 encoding
    input_ids = []
    deck_ids = []
    for tag, deck_idx in tag_dict.items():
        input_id = vl_chat_processor.tokenizer.encode(tag)[1:]
        input_ids.append(input_id)
        deck_ids.append(deck_idx)
        
    trie = Trie(input_ids, deck_ids) # input_ids로 trie를 만들고, leaf에 deck_ids를 저장함
    return trie

def get_user_prompot(prompt):
    return {
        "role": "<|User|>",
        "content": f"{prompt}",
    }
    
def get_assistant_prompot(prompt):
    return {
        "role": "<|Assistant|>",
        "content": f"{prompt}",
    }

# Retrieval을 위한 few shot samples
def get_few_shot_conversations():    
    conversation = []
    conversation.append(get_user_prompot("From now on, you are an assistant that helps with retrieval."))
    conversation.append(get_user_prompot("Query: How many online shopping orders are made per month in CY2013 in India?"))
    conversation.append(get_assistant_prompot("Tag: india, online shopping, CY2013"))
    conversation.append(get_user_prompot("Query: Which country is the researcher with the Twitter handle @doqtu84 giving a Group-buying Market Overview of?"))
    conversation.append(get_assistant_prompot("Tag: @doqtu84, Group-buying Market Overview"))
    conversation.append(get_user_prompot("Query: Which is most expensive among iOS, Android, and Windows?"))
    conversation.append(get_assistant_prompot("Tag: iOS, Android, Windows"))
    conversation.append(get_user_prompot("Query: What mobile game is shown in the presentation?"))
    conversation.append(get_assistant_prompot("Tag: mobile game"))
    conversation.append(get_user_prompot("Query: What is Fonny Schenck's phone number?"))
    conversation.append(get_assistant_prompot("Tag: Fonny Schenck"))
    conversation.append(get_user_prompot("Query: What is the structure into which glomeruler filtrate filters through the glomeruli part of?"))
    conversation.append(get_assistant_prompot("Tag: glomeruler, filtrate, filters"))
    return conversation

def generate_tags(
    query, 
    trie,
    vlm, 
    tokenizer, 
    vl_chat_processor,
    beam_size=8,
    max_tag_len=128,
    temperature=1,
    ):
    # 1. Conversation 생성
    # 1.1. Few-shot 프롬프트 생성
    conversation = get_few_shot_conversations()
    # 1.2. query 추가
    conversation.append(get_user_prompot(f"Query: {query}"))
    # 1.3. Assistant prompt 추가
    conversation.append(get_assistant_prompot("Tag: "))

    # 2 Input 전처리
    # 이제 이미지가 없기 때문에, images에는 빈 list를 넘겨줌
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[], force_batchify=True
    ).to(vlm.device)
    inputs_embeds = vlm.prepare_inputs_embeds(**prepare_inputs)
    
    # 3. Inference 중 결과를 보관한 buffer들
    # 3.1. 현재까지 generated된 토큰들 보관
    generated_tokens = -torch.ones((beam_size, max_tag_len), dtype=torch.int).cuda()
    # 3.2. beam search를 위한 log lickelihood 버퍼
    log_likelihoods = torch.zeros(beam_size, device=vlm.device)
    # 3.3. 모든 beam이 동시에 search가 끝나는 게 아님 -> 아직 생성이 끝나지 않은 beam의 수 관리
    beam_left = beam_size
    # 3.4. 생성이 끝난 beam의 결과를 보관하기 위한 buffer
    generated_tags = []
    # 3.5. 모든 beam을 사용하기 위해 inputs embeds를 beam size만큼 복사
    # TODO 하지만 실제로는 처음에 가장 첫 beam만을 사용하기 때문에 사실 불필요한 연산량임. 개선 필요
    inputs_embeds = inputs_embeds.repeat((beam_size, 1, 1))
    
    # 4. Trie를 따라서 tag generation -> retrieval
    with torch.no_grad():
        for i in tqdm(range(max_tag_len)):
            # 4.1. Vlm inference
            # past_key_values: KV caching을 위해 필요
            outputs = vlm.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            
            # 4.2.log_prob 계산
            hidden_states = outputs.last_hidden_state            
            logits = vlm.language_model.lm_head(hidden_states[:, -1, :])
            probs = torch.softmax(logits / temperature, dim=-1)
            log_probs = torch.log(probs)
            
            # 4.3. log-likelihoods 계산
            log_probs = log_likelihoods.unsqueeze(-1) + log_probs            
            log_probs = log_probs.clamp_min(-1e10)
            if i == 0:
                log_probs[1:] = -torch.inf  # 가장 처음에는 가장 첫 sample만 사용 -> 나머지 beam_size - 1개만큼은 버리는거라서 처음부터 계산 안 해도 됨 -> 대신 코드 복잡해져서 일단 둠       
            log_probs = log_probs.flatten() # flatten
            
            # 4.4. Trie를 고려하여, trie에 없는 next token 선택지는 log_prob을 -torch.inf로 설정해서, 뽑히지 않게 한다.
            # TODO 하지만, trie의 root 근처에서 beam_size보다 작은 개수가 존재하면, 문제가 생김. 예외처리나 다른 방법 적용 필요
            dict_size = probs.shape[-1]
            trie_mask = torch.ones((beam_left, dict_size), dtype=torch.bool, device= vlm.device) # True면 trie에 없는 것
            for j in range(beam_left):                
                next = trie.trie                  
                for k in range(i):
                    next = next.children[int(generated_tokens[j][k])]
                trie_mask[j][list(next.children.keys())] = False
            trie_mask = trie_mask.flatten()
            log_probs[trie_mask] = -torch.inf            
            
            # 4.5. top-k log-likelihoods를 고른다.
            log_probs_topk = log_probs.topk(beam_left)
            log_likelihoods = log_probs_topk.values
            indices = log_probs_topk.indices
            
            # 4.6. 몇 번째 beam의 몇 번째 token이 뽑혔는지 계산
            beam_indices = indices // dict_size
            next_token = indices % dict_size
            
            # 4.7. generated_tokens 업데이트
            # 4.7.1. 기존의 generated_tokens의 순서도 변경해줘야함
            generated_tokens = torch.stack([generated_tokens[beam_idx] for beam_idx in beam_indices])
            # 4.7.2. 그 뒤에 이번에 생성한 token 추가
            generated_tokens[:, i] = next_token.squeeze(dim=-1).to(torch.int32)
                        
            # 4.8. Trie의 끝에 도착한 beam은 끝내고, 더 이상 search를 하지 않게 빼줘야함
            _beam_left = beam_left
            next_beam_ids = []
            for beam_idx in range(_beam_left):
                beam_finished = False
                next = trie.trie
                for k in range(i+1):
                    next = next.children[int(generated_tokens[beam_idx][k])]
                    # 현재 진행된 depth까지 왔을 때, next children이 없다면, 해당 beam은 Trie의 끝에 도착한 것임.
                    if len(next.children) == 0:
                        beam_finished = True
                        break
                
                # 해당 beam이 끝났으면, beam_left를 1 감소시키고, generated_tags에 결과 저장
                if beam_finished:
                    beam_left -= 1
                    generated_tags.append(generated_tokens[beam_idx, :i+1])
                # 해당 beam이 끝나지 않았으면, 다음 stpe 준비하기 위해 beam idx를 next_beam_ids에 저장
                else:
                    next_beam_ids.append(beam_idx)
            
            # 4.9. 남은 beam이 있으면, 다음 Step을 위한 준빔를 해야함           
            if beam_left > 0:                    
                # 4.9.1 next_token 업데이트
                next_token = torch.stack([next_token[next_beam_id] for next_beam_id in next_beam_ids])
                # 4.9.2 log_likelihoods 업데이트
                log_likelihoods = torch.stack([log_likelihoods[next_beam_id] for next_beam_id in next_beam_ids])
                # 4.9.3 generated tokens업데이트
                generated_tokens = torch.stack([generated_tokens[next_beam_id] for next_beam_id in next_beam_ids])
                # 4.9.4 이번에 생성한 token의 임베딩을 다시 계산해서, inputs_embeds 업데이트
                inputs_embeds = vlm.language_model.get_input_embeddings()(next_token).unsqueeze(dim=1)
                
                # KV cache 업데이트
                for l, (key, value) in enumerate(zip(outputs.past_key_values.key_cache, outputs.past_key_values.value_cache)):
                    key_buffer = torch.zeros((beam_left, key.shape[1], key.shape[2], key.shape[3]), dtype=key.dtype, device=key.device)
                    value_buffer = torch.zeros((beam_left, value.shape[1], value.shape[2], value.shape[3]), dtype=value.dtype, device=value.device)                    
                    for j, next_beam_id in enumerate(next_beam_ids):
                        key_buffer[j] = key[beam_indices[next_beam_id]]        
                        value_buffer[j] = value[beam_indices[next_beam_id]]                    
                    outputs.past_key_values.key_cache[l] = key_buffer
                    outputs.past_key_values.value_cache[l] = value_buffer
            else:
                break
            
    return generated_tags

def retrieval(
    queries,
    trie,
    vlm, 
    tokenizer, 
    vl_chat_processor,
    ):
    top_1 = 0
    sampled = 0
    
    # Query들에 대한 평가
    for cnt, query in enumerate(queries):
        # 1. Query로부터 태그 생성
        generated_tags = generate_tags(
            query["question"],
            trie,    
            vlm, 
            tokenizer, 
            vl_chat_processor,        
        )
        
        # 2. 채점을 위해 이번 deck의 deck idx를 가져옴
        deck_name = query['deck_name']
        deck_idx = deck_indices[deck_name]    
        
        # 3. 생성된 tag들의 trie leaf에서 data를 가져옴 -> tag에 해당하는 deck list
        # TODO: tag_dict에서 바로 뽑아다 써도 될거같은데, 완전히 동일한지 확인 필요
        tags = []
        for generated_ids in generated_tags:
            next = trie.trie
            for id in generated_ids:
                next = next.children[int(id)]    
            print(next.data)
            tags += next.data
        
        # 4. 가장 많이 나온 deck이 정답 deck과 일치하는지 확인 -> Top 1 recall
        count_dict = Counter(tags)  
        if deck_idx == max(count_dict, key=count_dict.get):
            top_1+=1
        # 6. 정답 deck이 candidates에 있는지 확인 -> In candidates
        if deck_idx in count_dict:
            sampled+=1
        
        print(f"Top 1: {top_1/(cnt+1)*100:0.2f}")
        print(f"In candidates: {sampled/(cnt+1)*100:0.2f}")
        print(f"-------------------------------------")

    # 결과 표시
    total = len(queries)
    print(f"Total: {total}")
    print(f"Top 1: {top_1/total*100:0.2f}")
    print(f"In candidates: {sampled/total*100:0.2f}")

if __name__ == "__main__":
    # 1. 인덱싱해둔 데이터셋 로드
    tag_dict, deck_dict, decks, deck_indices = load_index("output/indexing")
    
    # 2. Trie 생성
    # 2.1 Janus 로드
    vlm, tokenizer, vl_chat_processor = load_janus()
    trie = build_trie(vlm, tokenizer, vl_chat_processor)
    
    # 3. Annotation 데이터셋 로드
    queries = load_slideVQA_annotations(decks, "SlideVQA", "test")
    
    # 4. query들에 대해 tag 생성하고, tag들을 포함하는 deck들을 이용하여 성능 집계
    retrieval(queries, trie, vlm, tokenizer, vl_chat_processor)