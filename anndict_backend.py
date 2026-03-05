# anndict_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import anndict  # AnnDictionary top-level
from anndict.annotate import ai_cell_type  # marker genes -> cell type :contentReference[oaicite:1]{index=1}


# ----------------------------
# Utilities
# ----------------------------

def parse_genes_flexible(text: str) -> List[str]:
    """
    Split a user-provided marker gene string into a clean list.

    Accepts comma/space/newline/semicolon separated input.
    Keeps order, removes empties, de-duplicates case-insensitively.
    """
    if not isinstance(text, str):
        return []

    # Replace separators with newlines then split
    seps = [",", ";", "\t", "|"]
    for s in seps:
        text = text.replace(s, "\n")

    parts = []
    for line in text.splitlines():
        # allow space-separated genes on a line
        for token in line.strip().split():
            token = token.strip()
            if token:
                parts.append(token)

    # de-dupe case-insensitively, preserve first occurrence
    seen = set()
    out = []
    for g in parts:
        key = g.upper()
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def normalize_label(label: Any) -> str:
    """Lowercase/strip and remove common redundant suffix 'cell(s)'."""
    if not isinstance(label, str):
        return ""
    s = " ".join(label.strip().lower().split())
    s = s.replace(" cells", "").replace(" cell", "")
    return s


def containment_match(true_label: str, pred_label: str) -> bool:
    """Flexible match: either contained in the other (after normalization)."""
    a = normalize_label(true_label)
    b = normalize_label(pred_label)
    if not a or not b:
        return False
    return (a in b) or (b in a)


# ----------------------------
# AnnDictionary / LLM glue
# ----------------------------

@dataclass
class AnnDictLLMConfig:
    """
    Minimal config holder.

    provider: e.g. "openai", "anthropic", "google", etc.
    model: provider model name (varies by provider)
    api_key: the API key string
    """
    provider: str
    model: str
    api_key: str


def configure_anndict_llm(cfg: AnnDictLLMConfig) -> None:
    """
    Configure AnnDictionary's LLM backend.

    AnnDictionary supports multiple providers via anndict.configure_llm_backend(). :contentReference[oaicite:2]{index=2}
    Exact provider/model strings depend on your provider.
    """
    # AnnDictionary advertises a 1-line config switch via configure_llm_backend() :contentReference[oaicite:3]{index=3}
    anndict.configure_llm_backend(
        provider=cfg.provider,
        model=cfg.model,
        api_key=cfg.api_key,
    )


def predict_cell_types_from_markers(
    species: str,
    marker_genes: List[str],
    *,
    tissue: Optional[str] = None,
    top_k: int = 5,
    llm_cfg: Optional[AnnDictLLMConfig] = None,
) -> List[str]:
    """
    Predict probable cell types from a marker gene list.

    Implementation approach:
    - AnnDictionary's ai_cell_type() returns a single label (string) for a gene list :contentReference[oaicite:4]{index=4}
    - To get "probable" / top-k, we query multiple times with slightly different context strings.

    species is injected into tissue/context to help the LLM disambiguate (human vs mouse markers, etc.).

    Returns:
      list of up to top_k unique predicted labels, ranked by frequency across queries
    """
    if llm_cfg is not None:
        configure_anndict_llm(llm_cfg)

    genes = [g for g in marker_genes if isinstance(g, str) and g.strip()]
    if not genes:
        return []

    # Build a context string AnnDictionary accepts: tissue (optional) :contentReference[oaicite:5]{index=5}
    # We pack species into that field to give the model context without changing the API.
    context_parts = []
    if species:
        context_parts.append(f"species={species}")
    if tissue:
        context_parts.append(f"tissue={tissue}")
    context = "; ".join(context_parts) if context_parts else None

    # Multiple prompts to approximate "top-k" probabilities:
    # (AnnDictionary core API returns 1 string; this is a lightweight ensemble on top.)
    prompt_variants = [
        context,
        (context + "; be concise; return a single cell type label") if context else "be concise; return a single cell type label",
        (context + "; prioritize canonical cell ontology names") if context else "prioritize canonical cell ontology names",
        (context + "; if ambiguous, choose the most likely broad cell type") if context else "if ambiguous, choose the most likely broad cell type",
        (context + "; interpret markers as cluster markers from scRNA-seq") if context else "interpret markers as cluster markers from scRNA-seq",
    ]

    preds: List[str] = []
    for tctx in prompt_variants[: max(1, min(len(prompt_variants), top_k * 2))]:
        try:
            pred = ai_cell_type(gene_list=genes, tissue=tctx)  # tissue param exists :contentReference[oaicite:6]{index=6}
            if isinstance(pred, str) and pred.strip():
                preds.append(pred.strip())
        except Exception:
            # keep going; caller can inspect empties
            continue

    # Rank by frequency, then first occurrence
    freq: Dict[str, int] = {}
    first_ix: Dict[str, int] = {}
    for i, p in enumerate(preds):
        key = normalize_label(p)
        if not key:
            continue
        freq[key] = freq.get(key, 0) + 1
        first_ix.setdefault(key, i)

    ranked = sorted(freq.keys(), key=lambda k: (-freq[k], first_ix[k]))
    # Return original-ish labels (use the first seen raw label for each normalized key)
    key_to_raw: Dict[str, str] = {}
    for p in preds:
        k = normalize_label(p)
        if k and k not in key_to_raw:
            key_to_raw[k] = p

    return [key_to_raw[k] for k in ranked[:top_k]]


def batch_predict_and_score(
    df: pd.DataFrame,
    *,
    species_col: str = "species",
    markers_col: str = "marker_genes",
    true_label_col: str = "cell_type",
    tissue_col: Optional[str] = None,
    top_k: int = 5,
    llm_cfg: Optional[AnnDictLLMConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Mass accuracy testing utility.

    Expected df columns:
      - species_col: str
      - markers_col: either (a) list[str] or (b) string that parse_genes_flexible can parse
      - true_label_col: ground truth label

    Outputs:
      - results_df: original df + predicted labels + match flags
      - metrics: dict with top1_exact, top1_containment, topk_containment
    """
    if llm_cfg is not None:
        configure_anndict_llm(llm_cfg)

    rows = []
    for _, row in df.iterrows():
        species = str(row.get(species_col, "") or "")
        tissue = str(row.get(tissue_col, "") or "") if tissue_col else None

        markers_val = row.get(markers_col, "")
        if isinstance(markers_val, list):
            genes = [str(x) for x in markers_val]
        else:
            genes = parse_genes_flexible(str(markers_val))

        true_lbl = str(row.get(true_label_col, "") or "")
        preds = predict_cell_types_from_markers(
            species=species,
            marker_genes=genes,
            tissue=tissue if tissue_col else None,
            top_k=top_k,
            llm_cfg=None,  # already configured above if provided
        )

        top1 = preds[0] if preds else ""
        exact = normalize_label(top1) == normalize_label(true_lbl)
        cont1 = containment_match(true_lbl, top1)
        contk = any(containment_match(true_lbl, p) for p in preds)

        out = dict(row)
        out.update(
            {
                "pred_top1": top1,
                "pred_topk": preds,
                "top1_exact": bool(exact),
                "top1_containment": bool(cont1),
                "topk_containment": bool(contk),
            }
        )
        rows.append(out)

    results = pd.DataFrame(rows)

    metrics = {
        "top1_exact": float(results["top1_exact"].mean()) if len(results) else 0.0,
        "top1_containment": float(results["top1_containment"].mean()) if len(results) else 0.0,
        "topk_containment": float(results["topk_containment"].mean()) if len(results) else 0.0,
    }
    return results, metrics
