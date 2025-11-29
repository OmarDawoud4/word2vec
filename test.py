from word2vec import Vocabulary, preprocess_text, tokenize_corpus

# test preprocessing
text = "Hello world this is a test!"
tokens = preprocess_text(text)
print("tokens:", tokens)

# test corpus tokenization
corpus = "Add a README file .and start coding in a secure env , github talkin !"
sentences = tokenize_corpus(corpus)
print("sentences:", sentences)

vocab = Vocabulary(min_count=1)
vocab.build_vocab(sentences)

# test word to index
word = "readme"
idx = vocab.get_idx(word)
print(f"'{word}' -> {idx}")

# test index to word
print(f"{idx} -> '{vocab.get_word(idx)}'")

#unknown word
unknown = "xyz"
print(f"unknown '{unknown}' -> {vocab.get_idx(unknown)}")

# show some words
print("\nfirst few words:")
for i in range(8):
    print(f"{i}: {vocab.get_word(i)}")

print("\ndone!")
