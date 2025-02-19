from typing import List
 
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
