"""Exact command-content token masks for the calibrated Teacher objective."""
from __future__ import annotations

from typing import Optional

import torch


def compute_action_mask(
    tokenizer,
    responses: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mask original tokens overlapping the environment's first command.

    Decode the actual generated IDs, never substitute re-tokenized responses.
    Tags can merge with newlines or neighboring tokens. Whitespace-only/tag/
    reasoning tokens are excluded; an indivisible boundary token containing
    command characters is included. Incomplete blocks have no command span.
    The existing row-validity mask still determines training eligibility.
    """
    from agent_system.action_text import first_action_span

    if responses.ndim != 2:
        raise ValueError("responses must be a 2-D token tensor")
    if response_mask is None:
        response_mask = torch.ones_like(responses, dtype=torch.bool)
    else:
        response_mask = response_mask.to(device=responses.device, dtype=torch.bool)
    if response_mask.shape != responses.shape:
        raise ValueError("response_mask must have the same shape as responses")

    # One host transfer per batch, not one CUDA synchronization per row.
    ids_cpu = responses.detach().cpu()
    valid_cpu = response_mask.detach().cpu()
    output = torch.zeros_like(ids_cpu, dtype=torch.bool)
    piece_cache = {}
    decode_options = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
    byte_decoder = None
    for row in range(ids_cpu.size(0)):
        positions = torch.nonzero(valid_cpu[row], as_tuple=False).flatten()
        tokens = ids_cpu[row, positions].tolist()
        text = tokenizer.decode(tokens, **decode_options)
        span = first_action_span(text)
        if span is None:
            continue
        start, stop = span
        while start < stop and text[start].isspace():
            start += 1
        while stop > start and text[stop - 1].isspace():
            stop -= 1
        if stop <= start:
            continue
        for token in tokens:
            if token not in piece_cache:
                piece_cache[token] = tokenizer.decode([token], **decode_options)
        pieces = [piece_cache[token] for token in tokens]
        if "".join(pieces) == text:
            lengths = [len(piece) for piece in pieces]
        else:
            # Qwen/GPT byte-level BPE can split one UTF-8 character across IDs.
            # Recover original bytes rather than losing these token boundaries.
            if byte_decoder is None:
                alphabet = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
                chars = alphabet.copy()
                extra = 0
                for value in range(256):
                    if value not in alphabet:
                        alphabet.append(value)
                        chars.append(256 + extra)
                        extra += 1
                byte_decoder = dict(zip(map(chr, chars), alphabet))
            try:
                special_ids = set(tokenizer.all_special_ids)
                raw_pieces = [
                    piece_cache[token].encode("utf-8") if token in special_ids else
                    bytes(byte_decoder[c] for c in tokenizer.convert_ids_to_tokens(token))
                    for token in tokens
                ]
                raw = b"".join(raw_pieces)
                if raw.decode("utf-8", errors="replace") != text:
                    raise ValueError("decoded bytes differ")
                lengths = [len(piece) for piece in raw_pieces]
                # Locate ASCII tags in the original bytes. This also handles
                # sampled invalid UTF-8 elsewhere, whose replacement glyphs
                # must not shift the command's byte offsets.
                byte_start, byte_stop = first_action_span(raw.decode("latin-1"))
                content = raw[byte_start:byte_stop].decode("utf-8", errors="replace")
                leading = len(content) - len(content.lstrip())
                trailing = len(content.rstrip())
                start = byte_start + len(content[:leading].encode("utf-8"))
                stop = byte_stop - len(content[trailing:].encode("utf-8"))
            except (AttributeError, KeyError, TypeError, ValueError):
                # Other fast tokenizers: offsets are usable only if every ID
                # matches. Never score a canonical re-encoding instead of the
                # original sampled token sequence.
                encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                if list(encoded["input_ids"]) != tokens:
                    raise ValueError(f"Cannot align original action tokens in response row {row}")
                offsets = encoded["offset_mapping"]
                selected = [j for j, (lo, hi) in enumerate(offsets) if max(lo, start) < min(hi, stop)]
                output[row, positions[selected]] = True
                continue
        cursor = 0
        selected = []
        for j, length in enumerate(lengths):
            end = cursor + length
            if max(cursor, start) < min(end, stop):
                selected.append(j)
            cursor = end
        output[row, positions[selected]] = True
    return output.to(device=responses.device)
