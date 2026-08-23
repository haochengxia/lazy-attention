"""Real RAG corpora, in the block layout the Block-FT checkpoints were trained on.

Every LazyRoute experiment needs the same three things: a question, a list of
documents each of which becomes one cache block, and the instruction tail that
stays in the request. Synthetic filler is not a substitute -- a 1B model handed
a few thousand tokens of near-duplicate boilerplate degenerates into "The
answer is: The answer is:", which looks exactly like a broken kernel.

Source: 2WikiMultihopQA dev (`xanhho/2WikiMultihopQA`), already in the HF cache;
each example carries ten Wikipedia paragraphs and marks which of them the
answer needs, which is what makes it a multi-hop routing testbed rather than a
single-needle one.

Template: the Tulu markers `benchmarks/blockbench.py::make_blocks` rewrites the
Llama-3 headers into. This is not cosmetic -- handed the Llama-3 header form,
`hxia7/Llama-3.2-1B-Block-FT` answers that the documents do not contain what is
plainly in them, while the Tulu form answers correctly from the same blocks.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

# The preamble is block 0: a document like any other to the cache, and the one
# the router keeps unconditionally.
PREAMBLE_BLOCK = (
    "<|user|>\n"
    "You are an intelligent AI assistant. Please answer questions based on "
    "the user's instructions. Below are some reference documents that may "
    "help you in answering the user's question.\n\n")

_HF_CACHE = os.path.expanduser(
    os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub")
_TWOWIKI_GLOB = "datasets--xanhho--2WikiMultihopQA"


def instruction_tail(question: str) -> str:
    """The part of the prompt that is *not* a reusable block."""
    return ("\n\nPlease write a high-quality answer for the given question "
            "using only the provided search documents (some of which might be "
            f"irrelevant).\nQuestion: {question}\n<|assistant|>\n")


def format_document(title: str, text: str) -> str:
    return f"- Title: {title}\n{text.strip()}\n"


@dataclass
class Example:
    question: str
    answer: str
    documents: list[str]  # reference documents, one cache block each
    supporting: list[int] = field(default_factory=list)  # indices into documents
    example_id: str = ""

    def blocks(self) -> list[str]:
        """What goes to `document_seqs`: the preamble, then the documents."""
        return [PREAMBLE_BLOCK] + self.documents

    def prompt(self) -> str:
        return instruction_tail(self.question)

    @property
    def supporting_blocks(self) -> list[int]:
        """Supporting documents as *block* indices (block 0 is the preamble)."""
        return [idx + 1 for idx in self.supporting]


def _find_2wiki() -> str:
    for root, _, files in os.walk(os.path.join(_HF_CACHE, _TWOWIKI_GLOB)):
        for name in files:
            if name == "dev.parquet":
                return os.path.join(root, name)
    raise FileNotFoundError(
        "2WikiMultihopQA dev.parquet not found under "
        f"{_HF_CACHE}. Fetch it with `huggingface-cli download --repo-type "
        "dataset xanhho/2WikiMultihopQA`.")


def load_2wiki(limit: int | None = None, path: str | None = None) -> list[Example]:
    """2wiki dev examples, each with its ten paragraphs as documents."""
    import pandas as pd

    frame = pd.read_parquet(path or _find_2wiki())
    if limit is not None:
        frame = frame.iloc[:limit]

    examples: list[Example] = []
    for _, row in frame.iterrows():
        paragraphs = json.loads(row["context"])
        titles = [title for title, _ in paragraphs]
        documents = [
            format_document(title, " ".join(sentences))
            for title, sentences in paragraphs
        ]
        support_titles = {
            title
            for title, _ in json.loads(row["supporting_facts"])
        } if isinstance(row["supporting_facts"], str) else {
            fact[0]
            for fact in row["supporting_facts"]
        }
        supporting = [
            idx for idx, title in enumerate(titles) if title in support_titles
        ]
        examples.append(
            Example(question=row["question"],
                    answer=row["answer"],
                    documents=documents,
                    supporting=supporting,
                    example_id=str(row["_id"])))
    return examples


def distractor_pool(examples: Sequence[Example],
                    exclude: Example | None = None) -> Iterator[str]:
    """Documents from other examples, for padding a corpus out to large M."""
    excluded = set(exclude.documents) if exclude is not None else set()
    for example in examples:
        if exclude is not None and example.example_id == exclude.example_id:
            continue
        for document in example.documents:
            if document not in excluded:
                yield document


def widen(example: Example,
          num_documents: int,
          pool: Sequence[Example],
          seed: int = 0,
          shuffle: bool = False) -> Example:
    """Pad `example` out to `num_documents` blocks with unrelated paragraphs.

    The supporting documents stay in the set and keep being tracked, so recall
    remains measurable; everything else is a distractor drawn from other
    questions.

    `shuffle` decides what the widened corpus is a control for. Left off, the
    example's own documents keep their positions and the distractors are
    appended, so a run at M=200 differs from one at M=10 in the added blocks
    and nothing else -- the comparison isolates corpus size. Turned on,
    positions are randomised, which is the harder and more realistic setting
    but confounds size with where the answer sits.
    """
    if num_documents < len(example.documents):
        raise ValueError(
            f"widen() only grows a corpus: asked for {num_documents} "
            f"documents, the example already has {len(example.documents)}.")

    rng = random.Random(seed)
    distractors = []
    seen = set(example.documents)
    for document in distractor_pool(pool, exclude=example):
        if document in seen:
            continue
        seen.add(document)
        distractors.append(document)
        if len(distractors) >= num_documents - len(example.documents):
            break

    documents = example.documents + distractors
    order = list(range(len(documents)))
    if shuffle:
        rng.shuffle(order)
    position = {old: new for new, old in enumerate(order)}
    return Example(question=example.question,
                   answer=example.answer,
                   documents=[documents[idx] for idx in order],
                   supporting=sorted(position[idx] for idx in example.supporting),
                   example_id=example.example_id)


def lengthen(example: Example,
             num_documents: int,
             paragraphs_per_document: int,
             pool: Sequence[Example],
             seed: int = 0) -> Example:
    """Same cache, spread over fewer and much longer documents.

    `widen` grows a corpus by adding documents; this grows it by growing each
    one. The distinction matters because the two costs in a sparse decode scale
    with different things. The router's host cost tracks the number of *blocks*
    -- 10 documents of 60 paragraphs is the same block count as 600 of one, and
    the same 477 dispatches -- while the model's coherence tracks the number of
    *documents*, which §9b puts at roughly 20 for this 1B checkpoint. Long
    documents are the only shape where a cache big enough for routing to pay
    for itself and a document count the checkpoint can still answer over are
    the same corpus.

    Each supporting paragraph is placed *inside* a document rather than at its
    head, at a position drawn from `seed`. Head placement would put every
    answer in page 0, which the sink stripe and prefix closure both keep for
    free -- selection would look excellent without having selected anything.
    Placed mid-document, finding it is the actual page-level retrieval problem,
    and it is the regime where page-level selection should beat doc-level by
    the most (§9b: 0.965 against 0.409 at a 25% budget).
    """
    if num_documents < len(example.supporting):
        raise ValueError(
            f"lengthen() needs room for every supporting paragraph: asked for "
            f"{num_documents} documents, the example has "
            f"{len(example.supporting)} supporting.")
    if paragraphs_per_document < 1:
        raise ValueError("paragraphs_per_document must be at least 1")

    rng = random.Random(seed)
    filler: list[str] = []
    seen = set(example.documents)
    needed = num_documents * paragraphs_per_document
    for document in distractor_pool(pool, exclude=example):
        if document in seen:
            continue
        seen.add(document)
        filler.append(document)
        if len(filler) >= needed:
            break
    if not filler:
        raise ValueError("no distractor paragraphs available to lengthen with")

    supporting_text = [example.documents[idx] for idx in example.supporting]
    documents, supporting = [], []
    cursor = 0
    for index in range(num_documents):
        # Cycle the filler if the pool is smaller than the corpus asked for;
        # repeated distractors are still distractors, and the alternative is
        # silently building a shorter corpus than was requested.
        parts = [filler[(cursor + k) % len(filler)]
                 for k in range(paragraphs_per_document)]
        cursor += paragraphs_per_document
        if index < len(supporting_text):
            where = rng.randrange(paragraphs_per_document)
            parts[where] = supporting_text[index]
            supporting.append(index)
        documents.append("".join(parts))

    return Example(question=example.question,
                   answer=example.answer,
                   documents=documents,
                   supporting=supporting,
                   example_id=example.example_id)


NEEDLE_TITLE = "Aurora Station Operations Log"
NEEDLE_TEXT = ("Routine maintenance was completed on schedule. The access code "
               "for the Aurora vault is MERIDIAN-7742. All duty personnel are "
               "required to memorise this code and must not write it down.")
NEEDLE_QUESTION = "What is the access code for the Aurora vault?"
NEEDLE_ANSWER = "MERIDIAN-7742"


def needle_example(num_documents: int,
                   pool: Sequence[Example],
                   position: int | None = None,
                   seed: int = 0) -> Example:
    """One distinctive fact hidden in a corpus of unrelated paragraphs.

    Multi-hop 2wiki asks the model to find two documents, combine them, and
    phrase an answer. This asks it to find one document and copy thirteen
    characters. Both are retrieval, but the second needs almost nothing of the
    model beyond attending to the right page -- so it survives a longer context
    than the QA task does, and it is the honest way to show *text* at a corpus
    size where the QA task has already collapsed into repetition.

    The needle is formatted exactly like every other document, so nothing but
    its content distinguishes it: no length tell, no position tell, and it sits
    where `position` says rather than at either end, since the first and last
    documents are the two the attention finds for free.
    """
    if num_documents < 1:
        raise ValueError("a needle corpus needs at least one document")
    where = (num_documents // 2) if position is None else position
    if not 0 <= where < num_documents:
        raise ValueError(f"position {where} outside 0..{num_documents - 1}")

    rng = random.Random(seed)
    distractors = []
    for document in distractor_pool(pool):
        if NEEDLE_ANSWER.lower() in document.lower():
            continue
        distractors.append(document)
        if len(distractors) >= num_documents - 1:
            break
    if len(distractors) < num_documents - 1:
        distractors = [distractors[i % max(len(distractors), 1)]
                       for i in range(num_documents - 1)]
    rng.shuffle(distractors)

    documents = (distractors[:where]
                 + [format_document(NEEDLE_TITLE, NEEDLE_TEXT)]
                 + distractors[where:])
    return Example(question=NEEDLE_QUESTION,
                   answer=NEEDLE_ANSWER,
                   documents=documents,
                   supporting=[where],
                   example_id="needle")
