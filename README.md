# Gaussian_LDA
Implementation of the paper - <a href="http://rajarshd.github.io/papers/acl2015.pdf">"Gaussian LDA for Topic Models with Word Embeddings"</a>

### Data Format

* Embedding file: Each line of the file should contain the embedding (dense vector representation) for a word. Each vector should have the same dimension and each dimension should be space separated. The experiments were carried with word2vec embeddings trained on an English Wikipedia dataset. However our model is agnostic to the choice of word embedding procedure and you could use any embeddings (for example trained <a href="http://nlp.stanford.edu/projects/glove/">GLove vectors</a>)
* Corpus train/test files: Here each line is a document with the words mapped as integers. The integer index of the word in your vocabulary should be equal to the position (line_number) of the word embedding in the embedding file. Please take care of 0-indexing.


### Running the script
Checkout run_gaussian_lda.sh. It should be self-explanatory.

Contact: Rajarshi Das (rajarshi@cs.umass.edu)

Citation
```
@InProceedings{das-zaheer-dyer:2015,
  author    = {Das, Rajarshi  and  Zaheer, Manzil  and  Dyer, Chris},
  title     = {Gaussian LDA for Topic Models with Word Embeddings},
  booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  publisher = {Association for Computational Linguistics},
  url       = {http://www.aclweb.org/anthology/P15-1077}
}
```





