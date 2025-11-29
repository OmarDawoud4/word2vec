from collections import Counter
import re
from typing import List, Tuple
from torch.utils.data import Dataset


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


class Word2VecDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Vocabulary,
                 window_size: int = 2, mode: str = 'skipgram'):
        self.vocab = vocab
        self.window_size = window_size
        self.mode = mode
        self.pairs = self._generate_pairs(sentences)
        
    def _generate_pairs(self, sentences: List[List[str]]) -> List[Tuple]:
        pairs = []

        for sentence in sentences:
            indices = [self.vocab.get_idx(word) for word in sentence]
            
            # Context pairs
            for center_idx in range(len(indices)):
                center_word = indices[center_idx]

                # Context window
                start = max(0, center_idx - self.window_size)
                end = min(len(indices), center_idx + self.window_size + 1)
                
                context_words = []
                for ctx_idx in range(start, end):
                    if ctx_idx != center_idx:
                        context_words.append(indices[ctx_idx])
                
                if self.mode == 'skipgram':
                    # (center, context) pairs
                    for context_word in context_words:
                        pairs.append((center_word, context_word))
                else:
                    # (context, center) pairs
                    if context_words:
                        pairs.append((context_words, center_word))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]



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
