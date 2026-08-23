"""The Phase-0 instrument: what a decode token actually attends to, per block.

This is a reference implementation of LazyAttention's cache, in HF, built so
that the objects the router will eventually score are the objects measured
here. Two properties matter and are asserted rather than assumed
(`check_equivalence`):

1. **Documents are encoded standalone**, from position 0, so the cached keys
   are in each document's *local frame* -- the same tensors a cross-request
   cache would hold, and the ones descriptors have to be computed from.
2. **The decode query is de-rotated per document.** RoPE is relative, so
   attending a query at global position `p` to a local-frame key at position
   `j` of a document that starts at `g` gives the same score whether the key is
   rotated forward by `g` or the query is rotated back by it. The engine does
   the latter in the kernel; the scorers here do the same, which is what makes
   "score the de-rotated query against stored keys" exactly request-local Quest
   math inside the document's frame (PROJECT.md §1.4).

What is logged per decode step, per layer:

* the true attention distribution over the cache (from the model's own eager
  attention, not a reimplementation), and
* the pre-RoPE query, from which any scorer's view can be reconstructed.

Everything downstream -- oracle recall, the Quest box, centroids -- is computed
from those two, so a scorer can never be measured against a differently-built
oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

PAGE = 16  # the engine's block size; pages are the descriptor granularity (R2)


def _cache_keys(cache: DynamicCache) -> list[torch.Tensor]:
    """`DynamicCache` layout moved between transformers releases."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache
    return [layer.keys for layer in cache.layers]


def _cache_values(cache: DynamicCache) -> list[torch.Tensor]:
    if hasattr(cache, "value_cache"):
        return cache.value_cache
    return [layer.values for layer in cache.layers]


@dataclass
class DocumentCache:
    """One document's KV, in its own frame, exactly as the cache would hold it."""
    token_ids: torch.Tensor  # [len]
    keys: list[torch.Tensor]  # per layer, [kv_heads, len, head_dim]
    values: list[torch.Tensor]

    @property
    def length(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def num_pages(self) -> int:
        return (self.length + PAGE - 1) // PAGE


@dataclass
class Layout:
    """Where every cached token sits, in documents and in pages.

    The engine pads each document to a whole number of blocks and masks the
    padding, so a document's last page is short. The layout mirrors that: pages
    never straddle a document boundary, and a page's token count is what the
    budget arithmetic charges for it.
    """
    doc_of_token: torch.Tensor  # [L] int64
    page_of_token: torch.Tensor  # [L]
    page_doc: torch.Tensor  # [num_pages]
    page_len: torch.Tensor  # [num_pages]
    doc_len: torch.Tensor  # [num_docs]
    doc_start: torch.Tensor  # [num_docs] global position of each document
    # How far the decode query is rotated back to enter each document's frame.
    # For a block cache this is `doc_start`, because the keys were stored at
    # local positions. For the request-local control the keys are already at
    # their committed global positions, so it is zero and the query is scored
    # where it actually sits -- ordinary Quest.
    frame_offset: torch.Tensor = None  # [num_docs]

    def __post_init__(self):
        if self.frame_offset is None:
            self.frame_offset = self.doc_start

    @property
    def num_docs(self) -> int:
        return int(self.doc_len.shape[0])

    @property
    def num_pages(self) -> int:
        return int(self.page_len.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.doc_of_token.shape[0])


def build_layout(documents: Sequence[DocumentCache], device) -> Layout:
    doc_of_token, page_of_token, page_doc, page_len = [], [], [], []
    doc_len, doc_start = [], []
    cursor = page_cursor = 0
    for doc_idx, document in enumerate(documents):
        doc_start.append(cursor)
        doc_len.append(document.length)
        for offset in range(0, document.length, PAGE):
            size = min(PAGE, document.length - offset)
            page_doc.append(doc_idx)
            page_len.append(size)
            page_of_token.extend([page_cursor] * size)
            page_cursor += 1
        doc_of_token.extend([doc_idx] * document.length)
        cursor += document.length

    tensor = lambda values: torch.tensor(values, dtype=torch.int64, device=device)
    return Layout(doc_of_token=tensor(doc_of_token),
                  page_of_token=tensor(page_of_token),
                  page_doc=tensor(page_doc),
                  page_len=tensor(page_len),
                  doc_len=tensor(doc_len),
                  doc_start=tensor(doc_start))


@dataclass
class DecodeTrace:
    """One teacher-forced decode step, everything a scorer study needs.

    `attention` is over the *cache region only* -- the reusable blocks. The
    query and generated tail are always kept dense by the router, so they are
    excluded here and their share of the mass is recorded separately.
    """
    step: int
    position: int  # global position of the query token
    token_id: int
    attention: torch.Tensor  # [layers, q_heads, L] fp32, cache region
    tail_mass: torch.Tensor  # [layers, q_heads] mass outside the cache region
    query: torch.Tensor  # [layers, q_heads, head_dim] fp32, PRE-RoPE


@dataclass
class ProbeResult:
    documents: list[DocumentCache]
    layout: Layout
    traces: list[DecodeTrace] = field(default_factory=list)
    forced_text: str = ""

    def stacked_attention(self) -> torch.Tensor:
        return torch.stack([trace.attention for trace in self.traces])

    def stacked_query(self) -> torch.Tensor:
        return torch.stack([trace.query for trace in self.traces])


class BlockProbe:
    """Runs the reference block cache and logs what decode attends to."""

    def __init__(self,
                 model_name: str,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            attn_implementation="eager").to(device).eval()
        self.device = device
        self.dtype = dtype
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // self.num_q_heads)
        self.group = self.num_q_heads // self.num_kv_heads
        self._rotary = self.model.model.rotary_emb
        self._captured_q: list[torch.Tensor] = []
        self._hooks = []

    # ---------------------------------------------------------------- rope --

    def _rotate(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE at `positions` to [heads, seq, head_dim]."""
        batched = tensor[None]
        cos, sin = self._rotary(batched, positions[None])
        rotated, _ = apply_rotary_pos_emb(batched, batched, cos, sin)
        return rotated[0]

    def rotate_query(self, query: torch.Tensor, position: int) -> torch.Tensor:
        """Pre-RoPE query [heads, head_dim] -> rotated at one position."""
        positions = torch.tensor([position], device=query.device)
        return self._rotate(query[:, None, :], positions)[:, 0, :]

    # ------------------------------------------------------------ encoding --

    @torch.no_grad()
    def encode_document(self, text: str) -> DocumentCache:
        """Prefill one document standalone, from position 0."""
        token_ids = self.tokenizer(text, add_special_tokens=False,
                                   return_tensors="pt").input_ids.to(self.device)
        cache = DynamicCache()
        self.model(input_ids=token_ids, past_key_values=cache, use_cache=True)
        return DocumentCache(
            token_ids=token_ids[0],
            keys=[key[0].clone() for key in _cache_keys(cache)],
            values=[value[0].clone() for value in _cache_values(cache)])

    @torch.no_grad()
    def merge(self, documents: Sequence[DocumentCache],
              layout: Layout) -> DynamicCache:
        """Rotate each document's keys forward to where it sits, and concatenate.

        The alternative -- keeping local-frame keys and de-rotating the query --
        is what the engine kernel does; here the keys move instead, because HF
        attention has one query rotation for the whole cache. The scores are the
        same either way, which `check_equivalence` verifies.
        """
        merged = DynamicCache()
        for layer in range(self.num_layers):
            keys, values = [], []
            for doc_idx, document in enumerate(documents):
                start = int(layout.doc_start[doc_idx])
                positions = torch.full((document.length, ), start,
                                       device=self.device, dtype=torch.int64)
                keys.append(self._rotate(document.keys[layer], positions))
                values.append(document.values[layer])
            merged.update(torch.cat(keys, dim=1)[None],
                          torch.cat(values, dim=1)[None], layer)
        return merged

    # ------------------------------------------------------------- capture --

    def _install_hooks(self) -> None:
        self._hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):

            def hook(_module, _inputs, output, layer_idx=layer_idx):
                # [b, seq, q_heads * head_dim] -> keep the last position only.
                self._captured_q[layer_idx] = output[0, -1].view(
                    self.num_q_heads, self.head_dim).float()

            self._hooks.append(layer.self_attn.q_proj.register_forward_hook(hook))

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    # --------------------------------------------------------------- decode --

    @torch.no_grad()
    def encode_request_local(
            self, blocks: Sequence[str]) -> tuple[list[DocumentCache], DynamicCache]:
        """The control: the same documents prefilled as one contextualised run.

        This is what every request-local sparse decoder summarises -- keys that
        saw each other and are committed to the positions this request gave
        them. The tokens are the per-block tokenisations concatenated, not a
        re-tokenisation of the joined text, so the two arms compare the same
        token sequence page for page.
        """
        per_block = [
            self.tokenizer(block, add_special_tokens=False,
                           return_tensors="pt").input_ids[0].to(self.device)
            for block in blocks
        ]
        token_ids = torch.cat(per_block)
        cache = DynamicCache()
        self.model(input_ids=token_ids[None], past_key_values=cache,
                   use_cache=True)
        keys, values = _cache_keys(cache), _cache_values(cache)
        documents, cursor = [], 0
        for ids in per_block:
            span = slice(cursor, cursor + int(ids.shape[0]))
            documents.append(
                DocumentCache(token_ids=ids,
                              keys=[key[0, :, span].clone() for key in keys],
                              values=[value[0, :, span].clone()
                                      for value in values]))
            cursor += int(ids.shape[0])
        return documents, cache

    @torch.no_grad()
    def run(self,
            blocks: Sequence[str],
            prompt: str,
            forced_answer: str,
            max_steps: int = 32,
            request_local: bool = False) -> ProbeResult:
        """Cache the blocks, then log a teacher-forced decode over `forced_answer`.

        Teacher forcing rather than free running: the checkpoint degenerates on
        corpora of this size often enough (~22% at ten documents) that a
        free-running trace would be measuring repetition loops as often as
        retrieval. What is wanted is the attention of a decode that is going
        where a correct answer goes.

        `request_local` swaps the block cache for the contextualised control.
        The decode path below is identical either way; only what is in the
        cache, and whether the query has to be de-rotated to read it, differ.
        """
        if request_local:
            documents, cache = self.encode_request_local(blocks)
            layout = build_layout(documents, self.device)
            # Keys are already at their committed positions: no de-rotation.
            layout.frame_offset = torch.zeros_like(layout.doc_start)
        else:
            documents = [self.encode_document(block) for block in blocks]
            layout = build_layout(documents, self.device)
            cache = self.merge(documents, layout)
        cache_len = layout.num_tokens

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(self.device)
        answer_ids = self.tokenizer(forced_answer, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(self.device)
        forced = answer_ids[0][:max_steps]

        # Prefill everything but the last prompt token; that token is decode
        # step 0, and its attention is the one a router would first act on.
        prefill = prompt_ids[:, :-1]
        position = cache_len
        if prefill.shape[1]:
            positions = torch.arange(cache_len,
                                     cache_len + prefill.shape[1],
                                     device=self.device)[None]
            self.model(input_ids=prefill, past_key_values=cache,
                       position_ids=positions, use_cache=True)
            position = cache_len + prefill.shape[1]

        result = ProbeResult(documents=documents, layout=layout,
                             forced_text=self.tokenizer.decode(forced))
        step_tokens = torch.cat([prompt_ids[0, -1:], forced])

        self._captured_q = [None] * self.num_layers
        self._install_hooks()
        try:
            for step, token in enumerate(step_tokens):
                positions = torch.tensor([[position]], device=self.device)
                outputs = self.model(input_ids=token.view(1, 1),
                                     past_key_values=cache,
                                     position_ids=positions,
                                     use_cache=True,
                                     output_attentions=True)
                # [layers][b, q_heads, 1, kv_len] -> [layers, q_heads, kv_len]
                attention = torch.stack([
                    layer_attn[0, :, 0, :].float()
                    for layer_attn in outputs.attentions
                ])
                result.traces.append(
                    DecodeTrace(
                        step=step,
                        position=position,
                        token_id=int(token),
                        attention=attention[:, :, :cache_len].contiguous(),
                        tail_mass=attention[:, :, cache_len:].sum(-1),
                        query=torch.stack(self._captured_q)))
                position += 1
        finally:
            self._remove_hooks()
        return result

    # ---------------------------------------------------------------- check --

    @torch.no_grad()
    def check_equivalence(self, result: ProbeResult,
                          tolerance: float = 2e-2) -> dict:
        """The de-rotated-query scores must be the model's own attention scores.

        Everything downstream rests on this: descriptors are built in each
        document's local frame, and are only a valid summary of what decode
        sees if scoring the de-rotated query against local-frame keys
        reproduces the attention the model actually computed. Compared as
        post-softmax distributions over the cache region, since that is what
        recall is measured on.
        """
        trace = result.traces[0]
        layout = result.layout
        worst = 0.0
        for layer in range(self.num_layers):
            scores = torch.empty(self.num_q_heads, layout.num_tokens,
                                 device=self.device)
            for doc_idx, document in enumerate(result.documents):
                start = int(layout.doc_start[doc_idx])
                query = self.rotate_query(
                    trace.query[layer].to(self.dtype),
                    trace.position - int(layout.frame_offset[doc_idx]))
                keys = document.keys[layer]  # local frame
                query = query.view(self.num_kv_heads, self.group, self.head_dim)
                logits = torch.einsum("hgd,hjd->hgj", query.float(),
                                      keys.float())
                span = slice(start, start + document.length)
                scores[:, span] = logits.reshape(self.num_q_heads, -1)
            scores /= math.sqrt(self.head_dim)
            # The model's own softmax also spans the tail; renormalise both to
            # the cache region so the comparison is of the same object.
            reference = trace.attention[layer]
            reference = reference / reference.sum(-1, keepdim=True)
            mine = torch.softmax(scores, dim=-1)
            worst = max(worst, float((mine - reference).abs().max()))
        return {"max_abs_diff": worst, "ok": worst <= tolerance}
