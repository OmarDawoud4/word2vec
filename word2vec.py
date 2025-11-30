from collections import Counter
import re
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
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


class CBOW(nn.Module):
    #CBOW model: predicts center word from context words
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input embeddings (context words)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (center word)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        init_range = 0.5 / self.embedding_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, context_words, center_word, negative_samples):
        # Get context embeddings and average them
        context_embeds = self.in_embeddings(context_words)  # [batch_size, context_size, embed_dim]
        context_embeds = torch.mean(context_embeds, dim=1)  # [batch_size, embed_dim]
        
        # Get center and negative embeddings
        center_embeds = self.out_embeddings(center_word)  # [batch_size, embed_dim]
        neg_embeds = self.out_embeddings(negative_samples)  # [batch_size, num_neg, embed_dim]
        
        # Positive score
        pos_score = torch.sum(context_embeds * center_embeds, dim=1)  # [batch_size]
        pos_loss = -torch.log(torch.sigmoid(pos_score))
        
        # Negative scores
        neg_score = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, num_neg]
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)  # [batch_size]
        
        # Total loss
        loss = torch.mean(pos_loss + neg_loss)
        return loss
    
    def get_embeddings(self):
        return self.in_embeddings.weight.data.cpu().numpy()


class NegativeSampler:    
    def __init__(self, vocab: Vocabulary, power: float = 0.75):
        self.vocab = vocab
        self.power = power
        self.sampling_probs = self._compute_sampling_probs()
        
    def _compute_sampling_probs(self):
        # Get word frequencies (excluding special tokens)
        word_freqs = np.zeros(self.vocab.vocab_size)
        for word, idx in self.vocab.word2idx.items():
            if word not in ['<PAD>', '<UNK>']:
                word_freqs[idx] = self.vocab.word_counts.get(word, 0)
        
        # Apply 0.75power transformation(as in the paper)
        word_freqs = np.power(word_freqs, self.power)
        
        # Normalize to probabilities
        sampling_probs = word_freqs / np.sum(word_freqs)
        return sampling_probs
    
    def sample(self, num_samples: int, exclude: Optional[List[int]] = None) -> np.ndarray:
        #Sample negative examples
        samples = np.random.choice(
            self.vocab.vocab_size,
            size=num_samples,
            p=self.sampling_probs,
            replace=True
        )
        return samples

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


