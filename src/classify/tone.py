"""Step 5: 톤 점수 (7장). 문장마다 긍정·중립·부정 확률을 구하고, 리포트 톤 = 문장별 (긍정 − 부정)의 평균.
입력·출력 파일 이름은 설정의 text_source를 따른다: summary(v1)는 그대로, pdf(v2)는 이름 끝에 _pdf (19장 코드)."""
import os
import pathlib
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config, source_path

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")   # 고정 리비전 밖의 변환 파일을 받지 않는다

SENTENCES = pathlib.Path("data/interim/sentences.parquet")
PROBS = pathlib.Path("data/interim/sentence_probs.parquet")
TONE = pathlib.Path("data/processed/tone.parquet")


def label_order(id2label):
    """모델 설정의 id2label에서 positive, neutral, negative의 위치를 찾는다 (7장 Step 5)."""
    where = {name.lower(): int(i) for i, name in id2label.items()}
    return [where["positive"], where["neutral"], where["negative"]]


@lru_cache(maxsize=1)
def load_model(name, revision):
    """고정한 리비전의 토크나이저와 모델. 가중치는 그 리비전의 pytorch_model.bin만 쓴다 (16장)."""
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision, use_safetensors=False)
    model.eval()                                               # dropout을 꺼서 매번 같은 결과가 나오게
    return tok, model, label_order(model.config.id2label)


def score_sentences(sentences, cfg=None):
    """문장 목록 → 문장별 (p_pos, p_neu, p_neg) 배열, 입력 순서 그대로 (7장 Step 5).
    토큰 길이순으로 정렬해 model.batch_size개씩 묶고, max_tokens를 넘으면 뒤를 자른다."""
    m = (cfg or load_config())["model"]
    tok, model, order = load_model(m["name"], m["revision"])
    sentences = list(sentences)
    lengths = [len(ids) for ids in tok(sentences, truncation=True, max_length=m["max_tokens"])["input_ids"]]
    idx = np.argsort(lengths, kind="stable")                   # 짧은 문장부터
    out = np.zeros((len(sentences), 3))
    bs = m["batch_size"]
    with torch.no_grad():
        for start in tqdm(range(0, len(idx), bs), desc="톤 점수", disable=len(idx) <= bs):
            batch = idx[start:start + bs]
            enc = tok([sentences[i] for i in batch], padding=True, truncation=True,
                      max_length=m["max_tokens"], return_tensors="pt")
            out[batch] = model(**enc).logits.softmax(-1)[:, order].numpy()   # 원래 자리에 되돌려 넣는다
    return out


def report_tone(probs):
    """한 리포트의 문장별 확률(p_pos, p_neg 열) → 톤 = (p_pos − p_neg)의 평균. −1 완전 부정, +1 완전 긍정."""
    return float((probs["p_pos"] - probs["p_neg"]).mean())


def build_tone(cfg):
    """sentences.parquet → sentence_probs.parquet, tone.parquet (7장 Step 5, 18장 열 이름). v2는 각각 _pdf."""
    sentences, probs_path, tone_path = (source_path(p, cfg) for p in (SENTENCES, PROBS, TONE))
    s = pd.read_parquet(sentences)
    p = score_sentences(s["text"], cfg)
    rev = cfg["model"]["revision"]
    probs = s[["nid", "sent_idx"]].assign(p_pos=p[:, 0], p_neu=p[:, 1], p_neg=p[:, 2], model_revision=rev)
    probs_path.parent.mkdir(parents=True, exist_ok=True)
    probs.to_parquet(probs_path, index=False)

    g = probs.groupby("nid")
    tone = pd.DataFrame({"tone": g[["p_pos", "p_neg"]].apply(report_tone), "n_sentences": g.size()}).reset_index()
    tone["model_revision"] = rev
    tone_path.parent.mkdir(parents=True, exist_ok=True)
    tone.to_parquet(tone_path, index=False)
    return probs, tone


if __name__ == "__main__":
    t0 = time.time()
    cfg = load_config()
    print("글 출처:", cfg.get("text_source", "summary"), "→", source_path(TONE, cfg))
    probs, tone = build_tone(cfg)
    print(f"문장 {len(probs):,}개, 리포트 {len(tone):,}건, {(time.time() - t0) / 60:.1f}분")
    print("리포트 톤:", tone["tone"].describe()[["min", "25%", "50%", "75%", "max"]].round(3).to_dict())