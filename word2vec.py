from collections import Counter
import re
from typing import List


class Vocabulary:
    def __init__(self,min_count:int=5):
        self.min_count=min_count
        self.word2idx={}
        self.idx2word={} # Bidirectional lookup
        self.word_counts=Counter() # to filter rare ones 
        self.vocab_size=0
    def build_vocab(self, sentences: List[List[str]]):

        for sentence in sentences:
            self.word_counts.update(sentence)

        # Filter out rare words
        filtered_words=[word for word,count in self.word_counts.items() if count>=self.min_count]
        
        self.word2idx={'<PAD>':0,'<UNK>':1}
        self.idx2word={0:'<PAD>',1:'<UNK>'}

        for idx,word in enumerate(filtered_words, start=2):
            self.word2idx[word]=idx
            self.idx2word[idx]=word
        self.vocab_size=len(self.word2idx)

        print(f"Size is {self.vocab_size} words (min count={self.min_count})")
        
        
    def get_idx(self,word:str)->int:
            return self.word2idx.get(word,self.word2idx['<UNK>'])
    def get_word(self , idx:int )->str:
            return self.idx2word.get(idx,'<UNK>')




def preprocess_text(text:str)->List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    #Tokenize
    tokens=text.split()
    return tokens

def tokenize_corpus(corpus: str) -> List[List[str]]:
    #Tokenize a corpus into sentences
    sentences = corpus.split('.')
    tokenized = [preprocess_text(sent) for sent in sentences if sent.strip()]
    return [sent for sent in tokenized if len(sent) > 0]
