import re
from typing import List, Dict

# Advanced, presentation-ready chunking
# - Sentence-aware splitting (avoid mid-sentence breaks)
# - Overlapping windows using sentence units to approximate word overlap
# - Simple chunk title from the first sentence
# - Extra metadata useful for debugging and presentations

_SENT_BOUNDARY_REGEX = re.compile(
    r"(?<!\b[A-Z])(?<=[.!?])[\s\u00A0]+(?=[\(\"\'\“\”A-Z0-9])"
)
_WORD_REGEX = re.compile(r"\S+")


def sentence_tokenize(text: str) -> List[str]:
    """Split text into sentences using a pragmatic regex (no external deps)."""
    text = text.strip()
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Split by sentence boundaries
    sents = _SENT_BOUNDARY_REGEX.split(text)
    # Clean up
    return [s.strip() for s in sents if s and not s.isspace()]


def word_count(s: str) -> int:
    return len(_WORD_REGEX.findall(s))


def _estimate_overlap_sentences(chunk_sents: List[str], target_overlap_words: int) -> int:
    """Approximate how many trailing sentences to carry over to meet overlap words."""
    if not chunk_sents or target_overlap_words <= 0:
        return 0
    total = 0
    count = 0
    for s in reversed(chunk_sents):
        total += word_count(s)
        count += 1
        if total >= target_overlap_words:
            break
    # Ensure we don't overlap everything
    return min(count, max(1, len(chunk_sents) - 1))


def _chunk_title(chunk_sents: List[str], max_words: int = 12) -> str:
    if not chunk_sents:
        return ""
    first = chunk_sents[0]
    words = _WORD_REGEX.findall(first)
    if len(words) <= max_words:
        return first
    return " ".join(words[:max_words]) + "…"


def make_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Make high-quality chunks from raw text.

    Args:
        text: Document text.
        chunk_size: Target words per chunk (soft limit).
        overlap: Approximate overlapping words between consecutive chunks.

    Returns:
        List of dicts with keys:
          - chunk_id: sequential id
          - text: chunk content
          - chunk_title: short title for presentation
          - num_words: word count of the chunk
          - start_sent_idx / end_sent_idx: sentence span within the document
    """
    sents = sentence_tokenize(text)
    if not sents:
        return []

    chunks: List[Dict] = []
    i = 0
    chunk_id = 0

    while i < len(sents):
        cur_words = 0
        start_idx = i
        bucket: List[str] = []

        # Accumulate sentences until we reach or slightly exceed target chunk size
        while i < len(sents):
            w = word_count(sents[i])
            # Always include at least one sentence
            if cur_words + w > chunk_size and bucket:
                break
            bucket.append(sents[i])
            cur_words += w
            i += 1

        # Build chunk content and metadata
        chunk_text = " ".join(bucket)
        meta = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "chunk_title": _chunk_title(bucket),
            "num_words": cur_words,
            "start_sent_idx": start_idx,
            "end_sent_idx": i - 1,
        }
        chunks.append(meta)
        chunk_id += 1

        if i >= len(sents):
            break

        # Compute sentence overlap based on target overlap words
        overlap_sents = _estimate_overlap_sentences(bucket, overlap)
        # Step back, but ensure progress (avoid infinite loops)
        step_back = min(overlap_sents, len(bucket) - 1) if len(bucket) > 1 else 0
        i = max(start_idx + 1, i - step_back)

    return chunks