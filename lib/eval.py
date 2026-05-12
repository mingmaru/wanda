# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


# Perplexity on held-out MC4-ko text. Parallels eval_ppl (WikiText-2) but on
# Korean. Continuous metric and large effective sample size at the token
# level -- meaningfully more sensitive to weight-level changes than Korean
# MCQ tasks (KoBEST-HS, KMMLU) where logit-magnitude shifts must cross the
# option-margin to register.
def eval_ppl_korean(args, model, tokenizer, device=torch.device("cuda:0")):
    dataset = "mc4_ko_test"
    print(f"evaluating perplexity on {dataset}")
    _, testenc = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
    )
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testenc, 1, device)
    return ppl

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


# Run downstream evaluation using lm-evaluation-harness v0.4+ API.
# task_configs is a list of (task_name, num_fewshot) tuples so each task
# can use its canonical few-shot setting (e.g. MMLU 5-shot, GSM8K 8-shot,
# KoBEST HellaSwag 0-shot).
def eval_downstream(model, tokenizer, task_configs, batch_size="auto", limit=None):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    all_results = {}
    for task_name, num_fewshot in task_configs:
        print(f"\n=== Evaluating {task_name} ({num_fewshot}-shot) ===")
        out = simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )
        all_results[task_name] = out["results"]
    return all_results


# Backwards-compatible wrapper for the old eval_zero_shot signature.
# Defaults to the proposal's three downstream tasks at their canonical few-shot
# counts. The model_name argument is ignored (the in-memory model is used).
def eval_zero_shot(model_name, model, tokenizer, task_list=None, num_fewshot=0,
                   use_accelerate=False, add_special_tokens=False):
    if task_list is None:
        task_configs = [
            ("mmlu", 5),
            ("gsm8k", 8),
            ("kobest_hellaswag", 0),
        ]
    else:
        task_configs = [(t, num_fewshot) for t in task_list]
    return eval_downstream(model, tokenizer, task_configs)