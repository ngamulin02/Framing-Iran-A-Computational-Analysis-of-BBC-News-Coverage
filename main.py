import re
import collections
import itertools
import math
from pathlib import Path

import pandas as pd
import spacy


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


ALIASES = {
    "donald trump": "Trump",
    "donald j trump": "Trump",
    "u.s.": "US",
    "united states": "US",
    "united states of america": "US",
    "america": "US",
    "u.k.": "UK",
    "united kingdom": "UK",
    "britain": "UK",
    "great britain": "UK",
    "iranian government": "Iran",
    "islamic republic": "Iran",
    "islamic republic of iran": "Iran",
    "tehran government": "Iran",
    "strait of hormuz": "Strait of Hormuz",
    "the strait": "Strait of Hormuz",
    "strait": "Strait of Hormuz",
    "irgc": "IRGC",
    "revolutionary guard": "IRGC",
    "revolutionary guards": "IRGC",
    "islamic revolutionary guard corps": "IRGC",
    "centcom": "Centcom",
    "us central command": "Centcom",
    "benjamin netanyahu": "Netanyahu",
    "ayatollah khamenei": "Khamenei",
    "ali khamenei": "Khamenei",
    "supreme leader": "Khamenei",
    "rubio": "Marco Rubio",
    "marco rubio": "Marco Rubio",
    "craig": "Craig Foreman",
    "craig foreman": "Craig Foreman",
    "lindsay": "Lindsay Foreman",
    "lindsay foreman": "Lindsay Foreman",
    "stana": "George Stana",
    "george stana": "George Stana",
}

LABEL_OVERRIDES = {
    "Trump": "PERSON",
    "Zeraati": "PERSON",
    "Axios": "ORG",
}

MIN_MENTIONS_FOR_CSV = 2

ENTITY_STOPLIST = {"bbc"}
MIN_ENTITY_LEN = 2


df = pd.read_csv("bbc_sentences.csv")
print(f"Loaded {len(df)} sentences across {df['article_id'].nunique()} articles.")


try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("en_core_web_lg not found; falling back to en_core_web_sm.")
    print("Install with: python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_sm")

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP"}


def normalise(text):
    t = text.strip()
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[\u2019']s$", "", t)
    canonical = ALIASES.get(t.lower())
    if canonical:
        return canonical
    t = t.strip(" .,;:'\"")
    canonical = ALIASES.get(t.lower())
    return canonical if canonical else t


def is_valid_entity(text):
    if len(text) < MIN_ENTITY_LEN:
        return False
    if text.lower() in ENTITY_STOPLIST:
        return False
    if re.fullmatch(r"[\d\W]+", text):
        return False
    return True


entity_counts = collections.defaultdict(
    lambda: {
        "label": None,
        "mentions": 0,
        "articles": set(),
        "display_forms": collections.Counter(),
        "sentence_ids": set(),
    }
)

cooc = collections.defaultdict(int)
agency = collections.defaultdict(int)

sentences = df["sentence"].tolist()
metadata = df[["article_id", "sentence_id"]].to_records(index=False)

print("Running NER...")
for (article_id, sentence_id), doc in zip(metadata, nlp.pipe(sentences, batch_size=64)):
    sent_key = (int(article_id), int(sentence_id))
    sentence_ents = []

    for ent in doc.ents:
        if ent.label_ not in KEEP_LABELS:
            continue
        raw = ent.text.strip()
        if not raw:
            continue
        canonical = normalise(raw)
        if not is_valid_entity(canonical):
            continue
        info = entity_counts[canonical]
        info["label"] = ent.label_
        info["mentions"] += 1
        info["articles"].add(int(article_id))
        info["sentence_ids"].add(sent_key)
        info["display_forms"][raw] += 1
        sentence_ents.append(canonical)

        root = ent.root
        while root.dep_ == "conj" and root.head is not root:
            root = root.head
        dep = root.dep_
        head = root.head
        role = None
        verb_token = None
        if dep == "nsubj":
            role, verb_token = "subject", head
        elif dep == "nsubjpass":
            role, verb_token = "passive_subject", head
        elif dep == "dobj":
            role, verb_token = "object", head
        elif dep == "pobj" and head.dep_ == "agent":
            role, verb_token = "subject", head.head
        if role and verb_token is not None and verb_token.pos_ in {"VERB", "AUX"}:
            agency[(canonical, verb_token.lemma_.lower(), role)] += 1

    unique_in_sent = list(dict.fromkeys(sentence_ents))
    for a, b in itertools.combinations(unique_in_sent, 2):
        cooc[frozenset({a, b})] += 1


entities_rows = []
for canonical, info in entity_counts.items():
    entities_rows.append(
        {
            "entity": canonical,
            "label": LABEL_OVERRIDES.get(canonical, info["label"]),
            "mentions": info["mentions"],
            "articles": len(info["articles"]),
        }
    )

entities_df = (
    pd.DataFrame(entities_rows)
    .sort_values("mentions", ascending=False)
    .reset_index(drop=True)
)
entities_df_out = entities_df[entities_df["mentions"] >= MIN_MENTIONS_FOR_CSV]
entities_df_out.to_csv(OUTPUT_DIR / "entities.csv", index=False)
print(
    f"Wrote entities.csv ({len(entities_df_out)} entities with >= {MIN_MENTIONS_FOR_CSV} mentions; "
    f"{len(entities_df) - len(entities_df_out)} single-mention entities filtered)."
)


total_sentences = len(df)
cooc_rows = []
for pair, shared in cooc.items():
    if shared < 3:
        continue
    a, b = sorted(pair)
    count_a = len(entity_counts[a]["sentence_ids"])
    count_b = len(entity_counts[b]["sentence_ids"])
    p_ab = shared / total_sentences
    p_a = count_a / total_sentences
    p_b = count_b / total_sentences
    pmi = math.log2(p_ab / (p_a * p_b)) if p_a > 0 and p_b > 0 else 0.0
    cooc_rows.append(
        {
            "entity_a": a,
            "entity_b": b,
            "shared_sentences": shared,
            "count_a": count_a,
            "count_b": count_b,
            "pmi": round(pmi, 3),
        }
    )

cooc_df = (
    pd.DataFrame(cooc_rows)
    .sort_values("shared_sentences", ascending=False)
    .reset_index(drop=True)
)
cooc_df.to_csv(OUTPUT_DIR / "cooccurrence.csv", index=False)
print(f"Wrote cooccurrence.csv ({len(cooc_df)} pairs with ≥3 co-occurrences).")


agency_rows = [
    {"entity": e, "verb": v, "role": r, "count": c}
    for (e, v, r), c in agency.items()
]
agency_df = (
    pd.DataFrame(agency_rows)
    .sort_values(["entity", "count"], ascending=[True, False])
    .reset_index(drop=True)
)
agency_df.to_csv(OUTPUT_DIR / "agency.csv", index=False)
print(f"Wrote agency.csv ({len(agency_df)} (entity, verb, role) tuples).")


if len(agency_df):
    summary = (
        agency_df.pivot_table(
            index="entity", columns="role", values="count",
            aggfunc="sum", fill_value=0,
        )
        .reindex(columns=["subject", "passive_subject", "object"], fill_value=0)
        .reset_index()
    )
    summary["total"] = summary["subject"] + summary["passive_subject"] + summary["object"]
    total_safe = summary["total"].astype(float).replace(0, float("nan"))
    patient_total = (summary["object"] + summary["passive_subject"]).astype(float).replace(0, float("nan"))
    summary["agency_ratio"] = (summary["subject"] / total_safe).round(3)
    summary["patient_ratio"] = ((summary["object"] + summary["passive_subject"]) / total_safe).round(3)
    summary["passive_share"] = (summary["passive_subject"] / patient_total).round(3)
    summary = summary.sort_values("total", ascending=False).reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "agency_summary.csv", index=False)
    print(f"Wrote agency_summary.csv ({len(summary)} entities).")

    role_totals = agency_df.groupby("role")["count"].sum().to_dict()
    print("\nRole totals across corpus:")
    for role in ("subject", "passive_subject", "object"):
        print(f"  {role:20s} {role_totals.get(role, 0)}")

    meaningful = summary[summary["total"] >= 5]
    if len(meaningful):
        print("\nMost active entities (top 5 by agency_ratio, total >= 5):")
        for _, row in meaningful.sort_values("agency_ratio", ascending=False).head(5).iterrows():
            print(f"  {row['entity']:25s} agency={row['agency_ratio']:.2f}  total={row['total']}")
        print("\nMost acted-upon entities (top 5 by patient_ratio, total >= 5):")
        for _, row in meaningful.sort_values("patient_ratio", ascending=False).head(5).iterrows():
            print(f"  {row['entity']:25s} patient={row['patient_ratio']:.2f}  total={row['total']}")

print("\nDone. Output files: entities.csv, cooccurrence.csv, agency.csv, agency_summary.csv")