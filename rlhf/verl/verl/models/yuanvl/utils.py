from typing import (Callable, Dict, Iterable, List, Literal, Mapping, Optional, Protocol, Set, Tuple, Union, overload)
import torch
from torch.func import functional_call

@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor:
    ...


@overload
def flatten_bn(x: List[torch.Tensor]) -> List[torch.Tensor]:
    ...


@overload
def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: Literal[True],
) -> torch.Tensor:
    ...


@overload
def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    ...


def flatten_bn(
    x: Union[List[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

    The input tensor should have shape ``(B, N, ...)```.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]

def _flatten_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))

def _embedding_count_expression(embeddings: torch.Tensor) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    Tensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)

def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)
    # [total_patches, text_config.hidden_size]
    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds

def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: torch.Tensor,
    placeholder_token_id: Union[int, List[int]],
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    ``placeholder_token_id`` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the ``input_ids`` MUST MATCH the order of
    their embeddings in ``multimodal_embeddings`` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = torch.tensor(placeholder_token_id,
                                            device=input_ids.device)
        return _merge_multimodal_embeddings(
            inputs_embeds,
            torch.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )
    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )
