---
layout: post
title: "LLM Fundamentals: Implementing Byte Pair Encoding (BPE), foundations that made it click for me"
date: 2026-08-18 21:55:00 -0700
permalink: /essays/implementing-byte-pair-encoding/
categories: tokenization machine-learning
---

When I first set out to implement Byte Pair Encoding (BPE), I assumed the hard part would be the merging logic. It wasn’t. The real friction came earlier: getting comfortable with what text *is* at the encoding level, and why tokenizers usually don’t operate on “characters” in the way we casually imagine.

This is a short write-up of the foundations that helped me, plus a practical sketch of how a minimal BPE tokenizer fits together.

---

## Text starts as code points, not bytes

Unicode gives us a universal way to talk about characters: **code points**—abstract numbers assigned to symbols.

In Python:

```python
print(ord('A'))        # 65
print(hex(ord('A')))   # 0x41

print(chr(65))         # A
print(chr(128512))     # 😀
```

At this level, “A” is 65, and “😀” is 128512. Clean and consistent.

But this representation is still *abstract*. Computers don’t store “code points” directly in memory in a standardized way—they store **bytes**. That’s where encodings come in.

---

## Encodings define code units (the bytes you actually store)

An encoding like UTF-8 tells us how to turn Unicode code points into **code units** (bytes). Same text, different physical representation depending on the encoding.

```python
text = "A😀"
utf8_bytes = text.encode('utf-8')
print(utf8_bytes)  # b'A\xf0\x9f\x98\x80'
```

The important detail: UTF-8 uses **variable-length bytes**. ASCII characters are one byte; many emojis are four bytes. So “characters” aren’t a stable unit for computation if you care about byte-level reality.

---

## Why not just tokenize Unicode characters?

You *can* tokenize at the character level, but it comes with tradeoffs:

- Unicode is huge (many languages, symbols, combined characters).
- “Character” boundaries can get subtle (grapheme clusters, composed forms, etc.).
- If you want a tokenizer that is robust to any input—especially arbitrary text on the internet—**bytes are a safer universal substrate**.

This is one of the reasons byte-level BPE became popular: it can accept any input string (because any string can be encoded to bytes), and it doesn’t have to assume a particular language or alphabet.

---

## The basic idea of BPE (byte-level)

At a high level, byte-level BPE does something simple:

1. Start with a sequence of bytes (0–255).
2. Count **frequent adjacent byte pairs**.
3. Repeatedly merge the most frequent pair into a new token id.
4. After enough merges, you have:
   - a learned list of merges (the “merge rules”)
   - a vocabulary mapping token ids to byte sequences

The goal isn’t to “understand language.” It’s to build a compact, reusable set of building blocks that represent common byte patterns as single tokens.

<figure class="visual-diagram">
  <img src="/assets/images/bpe-merge-pass.svg" alt="Excalidraw-style diagram of one BPE merge pass: start with byte tokens, count adjacent pairs, choose a pair, and replace matching occurrences." />
  <figcaption>One training merge pass: count adjacent pairs, choose the next merge rule, then rewrite the token sequence.</figcaption>
</figure>

The diagram uses letters to keep the example readable, but for byte-level BPE those letters are just byte ids. After a merge, the new token represents a byte sequence. In the example, `AN` is not a character; it is shorthand for the two-byte sequence that used to be `a` followed by `n`.

---

## A minimal implementation sketch

Here’s the shape of the whole system, end-to-end.

### Training (building merges + vocab)

Pseudo-process:

- Encode training text into bytes.
- For a target vocabulary size:
  - find the most frequent adjacent pair
  - create a new token id
  - merge occurrences of that pair in the corpus representation
- Store merges in an order that represents merge priority.

This produces:

- `merges`: mapping from pair → rank or new id (depending on design)
- `vocab`: mapping from id → byte sequence

One subtle but important point: merged byte sequences **don’t need to be valid UTF-8**. They’re just byte strings.

---

### Decoding: ids → bytes → text

Decoding is conceptually straightforward:

```python
def decode(ids):
    res = [vocab[id] for id in ids]     # vocab[id] is bytes
    text = b"".join(res)
    return text.decode("utf-8", errors="replace")
```

Using `errors="replace"` is a pragmatic choice: if your bytes ever aren’t valid UTF-8 (which can happen depending on vocab construction), you still get a readable result rather than an exception.

---

### Encoding: text → bytes → apply merges

Encoding is where the merge priority matters. A simple approach:

```python
def encode(input):
    ids = list(input.encode("utf-8"))  # start from raw bytes as token ids

    while len(ids) > 1:
        stats = get_stats(ids)  # counts of adjacent pairs
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        new_id = merges[pair]
        ids = merge(pair, new_id, ids)

    return ids
```

What this is doing:

- Look at all adjacent pairs currently in the sequence.
- Choose the pair with the **best (lowest) merge rank** (i.e., earliest learned merge).
- Merge it, and repeat until no merge rule applies.

This “keep applying the highest-priority merge available” pattern is the heart of BPE encoding.

---

## What I like about thinking in bytes

Once I stopped trying to reason about tokenization in terms of “characters,” things got calmer:

- Bytes are unambiguous.
- UTF-8 encoding is deterministic.
- You can always round-trip: text → bytes → tokens → bytes → text (modulo replacement strategy).
- The implementation becomes a careful exercise in sequence transforms, not string mysticism.

It’s not glamorous, but it’s solid—and that’s kind of the point.
