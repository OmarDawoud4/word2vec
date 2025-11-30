from word2vec import Vocabulary, Word2VecDataset, tokenize_corpus

corpus = "i love machine learning"
sentences = tokenize_corpus(corpus)

vocab = Vocabulary(min_count=1)
vocab.build_vocab(sentences)

dataset = Word2VecDataset(sentences, vocab, window_size=1, mode='cbow')

print("\npairs generated:")
for i in range(len(dataset)):
    context_list, center = dataset[i]
    context_words = [vocab.get_word(idx) for idx in context_list]
    print(f"{context_words} -> {vocab.get_word(center)}")
