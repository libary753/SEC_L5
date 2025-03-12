from typing import List
 
# trie 만들기
def build_trie(tag_dict, vl_chat_processor):
    # Keyword들을 encoding
    input_ids = []
    deck_ids = []
    for tag, deck_idx in tag_dict.items():
        input_id = vl_chat_processor.tokenizer.encode(tag)[1:]
        input_ids.append(input_id)
        deck_ids.append(deck_idx)
    trie = Trie(input_ids, deck_ids) # input_ids로 trie를 만들고, leaf에 deck_ids를 저장함
    return trie
        
class TrieNode:
    def __init__(self):
        self.children = {} # 자식 index들 -> 다음 순서의 token index
        self.size = 0 # 아래에 딸린 자식의 수
        self.data = None # leaf에 저장하는 data. leaf가 아니면 None, leaf이면 저장된 객체

class Trie(object):
    def __init__(self, sequences: List[List[int]] = [], datas: List[object] = []):
        self.trie = TrieNode()
        self.len = 0
        if sequences:
            for sequence, data in zip(sequences, datas):
                Trie._add_to_trie(sequence, data, self.trie)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    @staticmethod
    def _add_to_trie(sequence: List[int], data: object, trie_node: TrieNode):
        if sequence:
            if sequence[0] not in trie_node.children:
                trie_node.children[sequence[0]] = TrieNode()            
            trie_node.size += 1
            Trie._add_to_trie(sequence[1:], data, trie_node.children[sequence[0]])
        else:
            trie_node.data = data
