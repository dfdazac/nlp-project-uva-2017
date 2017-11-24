def perplexity(corpusfname, scoringfcn, base=10):
    """ Evaluates the perplexity of a language model
    on a given corpus.
    Args:
        - corpusfname (str): the corpus file name
        - scoringfcn (function): it should take a sentence (str).
                      as an argument and return its log-base probability (float).
        - base (int): the base of the log-probability returned by the model
    Returns:
        - float: the perplexity of the model on the corpus.
    """
    n_words = 0
    log_p_sum = 0
    with open(corpusfname) as corpus:        
        for line in corpus:            
            log_p_sum += scoringfcn(line)
            n_words += len(line.split())
    return base ** (-1/n_words * log_p_sum)
