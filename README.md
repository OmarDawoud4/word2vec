# Word2Vec 
A complete implementation of Word2Vec (Skip-gram and CBOW) with negative sampling, built using PyTorch.

## Overview
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
- **[Read the detailed blog post explaining Word2Vec](https://omardawoud.netlify.app/blog/word2vec-explained)**

# Features

- **Two Model Architectures**:
  - Skip-gram: Predicts context words from center word
  - CBOW (Continuous Bag of Words): Predicts center word from context

- **Negative Sampling**: Efficient training using negative sampling with 0.75 power distribution

- **Complete Pipeline**:
  - Text preprocessing and tokenization
  - Vocabulary building with frequency filtering
  - Training pair generation
  - Word similarity search
  - Word analogies 
  - t-SNE visualization of embeddings
  - 
## Training Details

- **Corpus**: Music lyrics from Linkin Park, Pink Floyd, The Beatles, Nirvana, Metallica, and The Doors
- **Vocabulary Size**: 3,885 unique words
- **Embedding Dimension**: 100
- **Window Size**: 3
- **Negative Samples**: 5
- **Learning Rate**: 0.005
- **Epochs**: 60
- **Device**: CUDA (GPU) if available, else CPU

##  TODO
- Implement subsampling of frequent words to improve training quality and speed
