### Contents

estimate_degree_of_grammaticization_by_classifier.py: program to estimate degree of grammaticalization

content_word_pos.dict: POS tags for English content words

content_word_pos_jp.dict: POS tags for Japanese content words

function_word_pos.dict: POS tags for English function words

function_word_pos_jp.dict: POS tags for Japanese function words


The program requires a corpus file and a word embedding file, which are not included in this repository. The corpus file is the text you want to analyze (to quantify the degree of grammaticalization of the words in it). It follows the one-sentence-per-line format. Each sentence consists of token-POS pair separated by a TAB and all token-POS pairs by a blank. The word embedding file can be obtained from the same text (the tokens) as in the corpus file by using gensim word2vec.

### How to use
Estimate degree of grammaticalization by CV-learning.     
    `python estimate_degree_of_grammaticization_by_classifier.py -f FUNCTION_WORD_POS_FILE -c CONTENT_WORD_POS_FILE -e EMBEDDING_FILE CORPUS_FILE`

     INPUT_FILE: FUNCTION_WORD_POS_FILE, file consisting of POS tags for function words (one POS per line)
                 CONTENT_WORD_POS_FILE, file consisting of POS tags for content words (one POS per line)
                 EMBEDDING_FILE, gensim word2vec embedding file
                 CORPUS_FILE, corpus file where each line consists of a sentence (token and POS tag separated by a TAB and token\tPOS separated by a blank)
                  
     OUTPUT: token, degree of gramaticalization separated by a blank.
   
    e.g.,
     `python estimate_degree_of_grammaticization_by_classifier.py -f function_word_pos.dict -c content_word_pos.dict -e EMBEDDING_FILE CORPUS_FILE` (for English) 

     `python estimate_degree_of_grammaticization_by_classifier.py -f function_word_pos_jp.dict -c content_word_pos_jp.dict -e EMBEDDING_FILE CORPUS_FILE` (for Japanese) 


Add -p option to estimate degree of grammaticalization by PU-learning. 
