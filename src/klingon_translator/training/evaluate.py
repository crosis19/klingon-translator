"""Evaluation utilities: BLEU, chrF, sample translations, and reports."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from klingon_translator.utils.config import (
    BASE_MODEL_ID,
    ENGLISH_CODE,
    KLINGON_CODE,
)

# ── Default sample phrases with expected translations ─────────
DEFAULT_SAMPLE_PHRASES_EN: list[tuple[str, str]] = [
    ("Today is a good day to die.", "Heghlu'meH QaQ jajvam."),
    ("Success!", "Qapla'!"),
    ("What do you want?", "nuqneH?"),
    ("We are Klingons!", "tlhIngan maH!"),
    ("Don't be silly!", "yIDoghQo'!"),
    ("Where is the bathroom?", "nuqDaq 'oH puchpa''e'?"),
    ("I don't understand.", "jIyajbe'."),
    (
        "Revenge is a dish best served cold.",
        "bortaS bIr jablu'DI' reH QaQqu' nay'.",
    ),
    (
        "Only a fool fights in a burning house.",
        "meQtaHbogh qachDaq Suv qoH neH.",
    ),
]

DEFAULT_SAMPLE_PHRASES_TLH: list[tuple[str, str]] = [
    ("Qapla'!", "Success!"),
    ("nuqneH.", "What do you want?"),
    ("tlhIngan maH!", "We are Klingons!"),
    ("HIja'.", "Yes."),
    ("ghobe'.", "No."),
    ("yIDoghQo'!", "Don't be silly!"),
    ("qatlho'.", "Thank you."),
    ("maj.", "Good."),
]


@dataclass
class EvalScores:
    """Bidirectional evaluation scores."""

    bleu_en2tlh: float
    bleu_tlh2en: float
    chrf_en2tlh: float
    chrf_tlh2en: float
    bleu_average: float
    chrf_average: float


@dataclass
class SampleResult:
    """Result of translating a single test phrase."""

    input: str
    expected: str
    predicted: str
    match: bool


def _get_device(device: str | None = None) -> torch.device:
    """Resolve device string to torch.device."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate_batch(
    texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 32,
    max_length: int = 128,
    num_beams: int = 5,
    device: str | None = None,
) -> list[str]:
    """Translate a list of texts in batches."""
    if not texts:
        return []

    dev = _get_device(device)
    model.eval()
    results = []
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(dev)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=max_length,
                num_beams=num_beams,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)

        if (i // batch_size) % 5 == 0:
            n = min(i + batch_size, len(texts))
            print(f"  Translated {n}/{len(texts)}...")

    return results


def evaluate_test_set(
    test_data: list[dict[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    device: str | None = None,
) -> EvalScores:
    """Compute BLEU and chrF for both directions on a test set."""
    import sacrebleu

    en_texts = [p["en"] for p in test_data]
    tlh_refs = [p["tlh"] for p in test_data]
    tlh_texts = [p["tlh"] for p in test_data]
    en_refs = [p["en"] for p in test_data]

    print("Evaluating English -> Klingon...")
    tlh_preds = translate_batch(
        en_texts, model, tokenizer, ENGLISH_CODE, KLINGON_CODE,
        batch_size=batch_size, device=device,
    )
    bleu_en2tlh = sacrebleu.corpus_bleu(tlh_preds, [tlh_refs], force=True)
    chrf_en2tlh = sacrebleu.corpus_chrf(tlh_preds, [tlh_refs])

    print("Evaluating Klingon -> English...")
    en_preds = translate_batch(
        tlh_texts, model, tokenizer, KLINGON_CODE, ENGLISH_CODE,
        batch_size=batch_size, device=device,
    )
    bleu_tlh2en = sacrebleu.corpus_bleu(en_preds, [en_refs], force=True)
    chrf_tlh2en = sacrebleu.corpus_chrf(en_preds, [en_refs])

    scores = EvalScores(
        bleu_en2tlh=round(bleu_en2tlh.score, 2),
        bleu_tlh2en=round(bleu_tlh2en.score, 2),
        chrf_en2tlh=round(chrf_en2tlh.score, 2),
        chrf_tlh2en=round(chrf_tlh2en.score, 2),
        bleu_average=round((bleu_en2tlh.score + bleu_tlh2en.score) / 2, 2),
        chrf_average=round((chrf_en2tlh.score + chrf_tlh2en.score) / 2, 2),
    )

    be = scores.bleu_en2tlh
    bt = scores.bleu_tlh2en
    ba = scores.bleu_average
    print(f"  BLEU  en->tlh: {be}  tlh->en: {bt}  avg: {ba}")
    ce = scores.chrf_en2tlh
    ct = scores.chrf_tlh2en
    ca = scores.chrf_average
    print(f"  chrF  en->tlh: {ce}  tlh->en: {ct}  avg: {ca}")

    return scores


def _translate_single(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    src_lang: str,
    tgt_lang: str,
    device: str | None = None,
) -> str:
    """Translate a single text."""
    dev = _get_device(device)
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(dev)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        out = model.generate(
            **inputs, forced_bos_token_id=tgt_id,
            max_length=128, num_beams=5,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def run_sample_translations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    phrases_en: list[tuple[str, str]] | None = None,
    phrases_tlh: list[tuple[str, str]] | None = None,
    device: str | None = None,
) -> tuple[list[SampleResult], list[SampleResult]]:
    """Translate sample phrases and compare against expected values."""
    if phrases_en is None:
        phrases_en = DEFAULT_SAMPLE_PHRASES_EN
    if phrases_tlh is None:
        phrases_tlh = DEFAULT_SAMPLE_PHRASES_TLH

    en2tlh_results = []
    print("=== English -> Klingon ===")
    for phrase, expected in phrases_en:
        predicted = _translate_single(
            phrase, model, tokenizer, ENGLISH_CODE, KLINGON_CODE, device
        )
        match = predicted.strip() == expected.strip()
        tag = "MATCH" if match else ""
        print(f"  {phrase}")
        print(f"    expected: {expected}")
        print(f"    got:      {predicted}  {tag}")
        print()
        en2tlh_results.append(
            SampleResult(input=phrase, expected=expected,
                         predicted=predicted, match=match)
        )

    tlh2en_results = []
    print("=== Klingon -> English ===")
    for phrase, expected in phrases_tlh:
        predicted = _translate_single(
            phrase, model, tokenizer, KLINGON_CODE, ENGLISH_CODE, device
        )
        match = predicted.strip() == expected.strip()
        tag = "MATCH" if match else ""
        print(f"  {phrase}")
        print(f"    expected: {expected}")
        print(f"    got:      {predicted}  {tag}")
        print()
        tlh2en_results.append(
            SampleResult(input=phrase, expected=expected,
                         predicted=predicted, match=match)
        )

    return en2tlh_results, tlh2en_results


def generate_training_report(
    eval_scores: EvalScores,
    sample_results_en2tlh: list[SampleResult],
    sample_results_tlh2en: list[SampleResult],
    train_result,
    training_config,
    gpu_config,
    tokenizer: PreTrainedTokenizerBase,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    save_path: str | Path | None = None,
) -> dict:
    """Generate and optionally save a JSON training report."""
    tc = training_config
    gc = gpu_config

    en2tlh_matches = sum(1 for s in sample_results_en2tlh if s.match)
    tlh2en_matches = sum(1 for s in sample_results_tlh2en if s.match)

    precision = "BF16" if gc.use_bf16 else "FP16" if gc.use_fp16 else "FP32"

    report = {
        "timestamp": datetime.now().isoformat(),
        "training": {
            "base_model": BASE_MODEL_ID,
            "gpu": gc.gpu_name,
            "gpu_memory_gb": round(gc.gpu_memory_gb, 1),
            "mode": "A100" if gc.is_a100 else "T4",
            "epochs": tc.max_epochs,
            "batch_size": gc.batch_size,
            "gradient_accumulation": gc.gradient_accumulation_steps,
            "effective_batch_size": gc.effective_batch_size,
            "learning_rate": tc.learning_rate,
            "lr_scheduler": tc.lr_scheduler,
            "precision": precision,
            "training_loss": round(train_result.training_loss, 4),
            "total_steps": train_result.global_step,
            "runtime_seconds": round(
                train_result.metrics["train_runtime"], 1
            ),
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "test_pairs": test_pairs,
        },
        "tokenizer": {
            "final_vocab_size": len(tokenizer),
        },
        "metrics": {
            "bleu_en2tlh": eval_scores.bleu_en2tlh,
            "bleu_tlh2en": eval_scores.bleu_tlh2en,
            "bleu_average": eval_scores.bleu_average,
            "chrf_en2tlh": eval_scores.chrf_en2tlh,
            "chrf_tlh2en": eval_scores.chrf_tlh2en,
            "chrf_average": eval_scores.chrf_average,
        },
        "sample_translations": {
            "en_to_tlh": [
                {"input": s.input, "expected": s.expected,
                 "predicted": s.predicted, "match": s.match}
                for s in sample_results_en2tlh
            ],
            "en_to_tlh_matches": (
                f"{en2tlh_matches}/{len(sample_results_en2tlh)}"
            ),
            "tlh_to_en": [
                {"input": s.input, "expected": s.expected,
                 "predicted": s.predicted, "match": s.match}
                for s in sample_results_tlh2en
            ],
            "tlh_to_en_matches": (
                f"{tlh2en_matches}/{len(sample_results_tlh2en)}"
            ),
        },
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {save_path}")

    _print_report_summary(report)

    return report


def _print_report_summary(report: dict) -> None:
    """Print a formatted training report summary."""
    t = report["training"]
    m = report["metrics"]
    s = report["sample_translations"]
    print()
    print("=" * 60)
    print("TRAINING REPORT")
    print("=" * 60)
    ts = report["timestamp"]
    print(f"  Timestamp:   {ts}")
    gpu = t["gpu"]
    mem = t["gpu_memory_gb"]
    print(f"  GPU:         {gpu} ({mem} GB)")
    ep = t["epochs"]
    print(f"  Epochs:      {ep}")
    bs = t["batch_size"]
    ga = t["gradient_accumulation"]
    ebs = t["effective_batch_size"]
    print(f"  Batch:       {bs} x{ga} = {ebs}")
    pr = t["precision"]
    print(f"  Precision:   {pr}")
    tl = t["training_loss"]
    print(f"  Train loss:  {tl}")
    rt = t["runtime_seconds"]
    print(f"  Runtime:     {rt:.0f}s")
    print()
    b1 = m["bleu_en2tlh"]
    b2 = m["bleu_tlh2en"]
    ba = m["bleu_average"]
    print(f"  BLEU  en->tlh: {b1}  tlh->en: {b2}  avg: {ba}")
    c1 = m["chrf_en2tlh"]
    c2 = m["chrf_tlh2en"]
    ca = m["chrf_average"]
    print(f"  chrF  en->tlh: {c1}  tlh->en: {c2}  avg: {ca}")
    print()
    se = s["en_to_tlh_matches"]
    st = s["tlh_to_en_matches"]
    print(f"  Samples  en->tlh: {se}  tlh->en: {st}")
    print("=" * 60)
