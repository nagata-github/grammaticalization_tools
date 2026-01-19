### Contents
estimate_degree_of_grammaticization_by_classifier.py: program to estimate degree of grammaticalization.

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
