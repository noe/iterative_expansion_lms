import torch
import torch.nn.functional as F


def filter_top_k_(logits, k):
    # Based on https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


def filter_top_p_(logits, p):
    # Based on https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    # with the batch extension proposed at
    # https://github.com/huggingface/transformers/pull/1333/commits/a9f24a16bc2965d2990b90127ed4b5a1f47344b9

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1,
                                                         sorted_indices,
                                                         sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits
