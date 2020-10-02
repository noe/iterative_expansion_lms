import argparse
import io
import math
import sys
import torch
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.tokenization_gpt2 import GPT2Tokenizer


MODEL_ID = 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)


def score(sentence):
    # Based on:
    # https://github.com/huggingface/transformers/issues/1009#issuecomment-521588881
    tokenize_input = tokenizer.tokenize(sentence)
    eos_id = tokenizer.eos_token_id
    token_idxs = tokenizer.convert_tokens_to_ids(tokenize_input)
    token_idxs = torch.tensor([[eos_id] + token_idxs])

    with torch.no_grad():
        outputs = model(token_idxs, labels=token_idxs)
        loss, logits = outputs[:2]
        num_tokens = len(tokenize_input)
        total_logprob = -loss * num_tokens
        return total_logprob.cpu().item(), num_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, type=str)
    parser.add_argument('--ppl', action='store_true')
    parser.add_argument('--also-ppl', type=str, required=False)
    args = parser.parse_args()
    input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
                   if args.input is None else open(args.input, encoding='utf-8'))

    accumulated_logprob = 0.
    total_num_tokens = 0

    for line in input_lines:
        sentence = line.strip()
        logprob, num_tokens = score(sentence)
        accumulated_logprob += logprob
        total_num_tokens += num_tokens
        if not args.ppl:
            print(logprob)

    log2_prob = accumulated_logprob / math.log(2)
    ppl = math.pow(2., - log2_prob / total_num_tokens)

    if args.ppl:
        print(ppl)
    elif args.also_ppl:
        with open(args.also_ppl, 'w', encoding='utf-8') as f:
            print(ppl, file=f)


if __name__ == '__main__':
    main()
