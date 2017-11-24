def perplexity(corpusfname, scoringfcn, base=10):
    """ Evaluates the perplexity of a language model
    on a given corpus. Assumes there is one sentence per line.
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
            # Add 1 for the end of sentence symbol
            n_words += len(line.split()) + 1
    
    return base ** (-log_p_sum/n_words)
