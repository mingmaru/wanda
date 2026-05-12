# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    traindata = load_dataset('allenai/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz', 'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='train', verification_mode='no_checks')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    valdata = load_dataset('allenai/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', verification_mode='no_checks')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Load and process Korean MC4 dataset (multilingual C4, Korean subset).
# Same web-text domain as English C4, differs in language only.
def get_mc4_ko(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'allenai/c4', 'multilingual',
        data_files={'train': 'multilingual/c4-ko.tfrecord-00000-of-01024.json.gz'},
        split='train',
        verification_mode='no_checks',
    )

    random.seed(seed)
    trainloader = []
    attempts = 0
    while len(trainloader) < nsamples:
        attempts += 1
        if attempts > nsamples * 50:
            raise RuntimeError(f"Could not collect {nsamples} long-enough Korean samples after {attempts} attempts")
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
        if trainenc.input_ids.shape[1] <= seqlen:
            continue
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, start:start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, None


# Load and process The Stack (code) dataset.
# Same English language as C4 but a non-natural-language domain.
def get_the_stack(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'bigcode/the-stack-smol',
        split='train',
        verification_mode='no_checks',
    )

    random.seed(seed)
    trainloader = []
    attempts = 0
    while len(trainloader) < nsamples:
        attempts += 1
        if attempts > nsamples * 50:
            raise RuntimeError(f"Could not collect {nsamples} long-enough code samples after {attempts} attempts")
        i = random.randint(0, len(traindata) - 1)
        text = traindata[i].get('content') or traindata[i].get('text', '')
        if not text:
            continue
        trainenc = tokenizer(text, return_tensors='pt')
        if trainenc.input_ids.shape[1] <= seqlen:
            continue
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, start:start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, None


# Held-out Korean text for perplexity evaluation.
# Uses a different shard than get_mc4_ko (which loads shard 00000), so the
# test text is disjoint from any calibration draw. Returns (None, testenc)
# where testenc has the same TokenizerWrapper.input_ids interface that
# eval_ppl_wikitext expects.
def get_mc4_ko_test(nsamples, seed, seqlen, tokenizer):
    valdata = load_dataset(
        'allenai/c4', 'multilingual',
        data_files={'validation': 'multilingual/c4-ko.tfrecord-00001-of-01024.json.gz'},
        split='validation',
        verification_mode='no_checks',
    )
    # Tokenize a single concatenated blob and truncate to 256 * seqlen tokens.
    # Matches the get_c4 valenc construction so eval_ppl_wikitext can chunk it
    # the same way it chunks WikiText-2.
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return None, valenc


# Generate random-token "calibration" — structureless control.
# Uniformly samples token IDs from the model's vocabulary.
def get_random(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    torch.manual_seed(seed)
    vocab_size = tokenizer.vocab_size

    trainloader = []
    for _ in range(nsamples):
        inp = torch.randint(0, vocab_size, (1, seqlen), dtype=torch.long)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, None


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name == "mc4_ko" or name == "mc4-ko":
        return get_mc4_ko(nsamples, seed, seqlen, tokenizer)
    if name == "mc4_ko_test":
        return get_mc4_ko_test(nsamples, seed, seqlen, tokenizer)
    if name == "the_stack" or name == "stack":
        return get_the_stack(nsamples, seed, seqlen, tokenizer)
    if name == "random":
        return get_random(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown calibration dataset: {name}")