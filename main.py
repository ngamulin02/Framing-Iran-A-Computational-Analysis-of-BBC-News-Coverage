import re
import collections
import itertools
import math
from pathlib import Path

import pandas as pd
import spacy



# Configuration


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_MENTIONS_FOR_CSV = 2
MIN_COOC_SHARED = 3
MIN_AGENCY_TOTAL = 5
MIN_DYAD_COUNT = 2
LOW_COUNT_FLAG_THRESHOLD = 10

ENTITY_STOPLIST = {"bbc"}
MIN_ENTITY_LEN = 2

# When True, fold NORP forms (Iranian, Israeli) and metonymic capitals
# (Tehran, Washington) into their state. Set to False to recover the
# surface-form distinction.
COLLAPSE_METONYMS = True



# Alias / label tables


ALIASES_CORE = {
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

# Metonym / NORP mappings. Applied only when COLLAPSE_METONYMS is True.
# The judgment call: in a war-coverage corpus, "Tehran" almost always
# refers to the Iranian government metonymically, not to traffic in the
# city. Same for "Washington" and "Beirut". If you care about surface
# distinctions, flip the flag.
ALIASES_METONYMS = {
    # NORP nationality forms
    "iranian": "Iran",
    "iranians": "Iran",
    "israeli": "Israel",
    "israelis": "Israel",
    "american": "US",
    "americans": "US",
    "british": "UK",
    "lebanese": "Lebanon",
    "pakistani": "Pakistan",
    "pakistanis": "Pakistan",
    # Capital cities standing for governments
    "tehran": "Iran",
    "washington": "US",
    "beirut": "Lebanon",
    "jerusalem": "Israel",
    "islamabad": "Pakistan",
    "london": "UK",
}

ALIASES = dict(ALIASES_CORE)
if COLLAPSE_METONYMS:
    ALIASES.update(ALIASES_METONYMS)

LABEL_OVERRIDES = {
    "Trump": "PERSON",
    "Zeraati": "PERSON",
    "Axios": "ORG",
}



# Predicate-class lexicons


ACTION_VERBS = {
    # Military / kinetic
    "attack", "strike", "kill", "bomb", "hit", "shoot", "fire", "launch",
    "raid", "invade", "assault", "destroy", "wound", "injure", "damage",
    "intercept", "shoot down", "defeat", "occupy", "capture", "seize",
    "blockade", "mine", "ram", "hijack", "board", "assassinate", "execute",
    # Coercive / state
    "sanction", "ban", "impose", "enforce", "expel", "arrest", "detain",
    "kidnap", "hold", "release", "free", "target",
    # Operational / logistical
    "deploy", "send", "move", "position", "station", "patrol",
    "fund", "supply", "arm", "train", "support", "back", "aid",
    "build", "develop", "shut", "close", "block", "retaliate", "intervene",
    "withdraw",
}

COMMUNICATIVE_VERBS = {
    "say", "tell", "claim", "state", "report", "announce", "declare",
    "describe", "note", "add", "explain", "comment", "write", "post",
    "tweet", "speak", "respond", "confirm", "admit", "acknowledge",
    "warn", "threaten", "demand", "urge", "call", "ask", "reject",
    "agree", "deny", "accuse", "condemn", "criticise", "criticize",
    "praise", "insist", "stress", "emphasise", "emphasize", "argue",
    "suggest", "vow", "pledge", "characterise", "characterize", "label",
}

MENTAL_VERBS = {
    "believe", "want", "see", "view", "consider", "fear", "hope",
    "expect", "think", "know", "understand", "feel", "wish", "plan",
    "intend", "seek", "aim", "regard", "deem",
}

ACTION_NOUNS = {
    "strike", "strikes", "attack", "attacks", "raid", "raids",
    "bombing", "bombings", "operation", "operations", "campaign",
    "offensive", "assault", "incursion", "incursions", "invasion",
    "response", "retaliation", "sanction", "sanctions", "threat",
    "deployment", "intervention", "blockade", "withdrawal",
    "support", "aid", "funding", "backing",
}

# Prepositions that introduce the target of an action verb. Gated on
# the verb being action-class so that "comment on Iran" (communicative)
# and "talks on Iran's program" don't fire as targeting.
TARGET_PREPS = {"against", "at", "into", "toward", "towards", "on", "over"}


def classify_verb_lemma(lemma):
    lemma = lemma.lower()
    if lemma in ACTION_VERBS:
        return "action"
    if lemma in COMMUNICATIVE_VERBS:
        return "communicative"
    if lemma in MENTAL_VERBS:
        return "mental"
    return "other"


def is_action_noun(lemma):
    return lemma.lower() in ACTION_NOUNS



# Load corpus + NLP model


df = pd.read_csv("bbc_sentences.csv")
print(f"Loaded {len(df)} sentences across {df['article_id'].nunique()} articles.")
print(f"Metonym collapse: {'ON' if COLLAPSE_METONYMS else 'OFF'}")

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("en_core_web_lg not found; falling back to en_core_web_sm.")
    print("Install with: python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_sm")

KEEP_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP"}



# Entity normalisation


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



# Role analysis
#
# Returns (role, predicate_token, predicate_lemma, predicate_class) or None.
#
# Verbal patterns:
#   subject         nsubj of a verb
#   passive_subject nsubjpass of a verb
#   object          dobj of a verb
#   subject (agent) pobj of "by" where "by" is the agent of a passive verb
#   prep_target     pobj of an action-targeting preposition whose head is
#                   an action verb, OR compound of such a pobj (one level).
#                   "Iran fired missiles at the US"  -> US is prep_target
#                   "Iran fired missiles at US bases" -> US is prep_target
#                   (via compound on "bases")
#
# Nominal patterns (deverbal action nouns only):
#   nominal_agent   poss of an action noun         "Iran's strike"
#   nominal_agent   pobj of "by" attached to an    "the attack by Iran"
#                   action noun
#   nominal_agent   amod of an action noun for     "Iranian strike"
#                   NORP entities


def _prep_target_check(prep_token):
    """Given a preposition token, return the governing verb if it's an
    action verb attached to a target-introducing preposition, else None."""
    if prep_token.lemma_.lower() not in TARGET_PREPS:
        return None
    verb = prep_token.head
    if verb.pos_ not in {"VERB", "AUX"}:
        return None
    if classify_verb_lemma(verb.lemma_) != "action":
        return None
    return verb


def analyse_entity_role(ent):
    root = ent.root
    while root.dep_ == "conj" and root.head is not root:
        root = root.head

    dep = root.dep_
    head = root.head

    # Verbal patterns 
    if head.pos_ in {"VERB", "AUX"}:
        lemma = head.lemma_.lower()
        if dep == "nsubj":
            return ("subject", head, lemma, classify_verb_lemma(lemma))
        if dep == "nsubjpass":
            return ("passive_subject", head, lemma, classify_verb_lemma(lemma))
        if dep == "dobj":
            return ("object", head, lemma, classify_verb_lemma(lemma))

    # Passive agent: "X was attacked by ENT" -> ENT is subject of "attack"
    if dep == "pobj" and head.dep_ == "agent":
        verb = head.head
        if verb.pos_ in {"VERB", "AUX"}:
            lemma = verb.lemma_.lower()
            return ("subject", verb, lemma, classify_verb_lemma(lemma))

    # Prepositional target: "Iran fired at the US" -> US is prep_target
    if dep == "pobj":
        verb = _prep_target_check(head)
        if verb is not None:
            lemma = verb.lemma_.lower()
            return ("prep_target", verb, lemma, classify_verb_lemma(lemma))

    # Prepositional target via compound: "Iran fired at US bases"
    # -> US is compound of "bases", "bases" is pobj of "at"
    if dep == "compound":
        parent = head
        if parent.dep_ == "pobj":
            verb = _prep_target_check(parent.head)
            if verb is not None:
                lemma = verb.lemma_.lower()
                return ("prep_target", verb, lemma, classify_verb_lemma(lemma))

    #  Nominal patterns 
    if dep == "poss" and head.pos_ == "NOUN" and is_action_noun(head.lemma_):
        return ("nominal_agent", head, head.lemma_.lower(), "action_nominal")

    if dep == "pobj" and head.lemma_.lower() == "by":
        gp = head.head
        if gp.pos_ == "NOUN" and is_action_noun(gp.lemma_):
            return ("nominal_agent", gp, gp.lemma_.lower(), "action_nominal")

    if (
        dep == "amod"
        and ent.label_ == "NORP"
        and head.pos_ == "NOUN"
        and is_action_noun(head.lemma_)
    ):
        return ("nominal_agent", head, head.lemma_.lower(), "action_nominal")

    return None



# Main analysis loop


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

# agency: (canonical, lemma, class, role) -> {count, example_sent, articles}
agency = collections.defaultdict(
    lambda: {"count": 0, "example_sent": None, "articles": set()}
)

# directed: (subject, lemma, class, object) -> {dobj, prep, passive,
#                                               example_sent, articles}
directed = collections.defaultdict(
    lambda: {
        "dobj": 0,
        "prep": 0,
        "passive": 0,
        "example_sent": None,
        "articles": set(),
    }
)

# For the fragility check we need the UNION of article ids per
# (entity, role) and per (entity, subject_predicate_class), not the sum
# of per-predicate article counts. Tracked separately because aggregating
# from agency_df with aggfunc="sum" would overcount any article that
# contains the same entity-role pair under multiple different predicates.
subj_class_articles = collections.defaultdict(set)
# (entity, predicate_class) -> set of article_ids, subject role only
role_articles = collections.defaultdict(set)
# (entity, role) -> set of article_ids, union across all predicates

sentences = df["sentence"].tolist()
metadata = df[["article_id", "sentence_id"]].to_records(index=False)

print("Running NER + role analysis...")
for (article_id, sentence_id), doc in zip(metadata, nlp.pipe(sentences, batch_size=64)):
    aid = int(article_id)
    sid = int(sentence_id)
    sent_key = (aid, sid)
    sent_label = f"{aid}:{sid}"
    sentence_ents = []
    # verb_token.i -> {"lemma":..., "class":..., "subjects":[], "objects":[],
    #                  "passives":[], "prep_targets":[]}
    verb_arg_map = {}

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
        info["articles"].add(aid)
        info["sentence_ids"].add(sent_key)
        info["display_forms"][raw] += 1
        sentence_ents.append(canonical)

        result = analyse_entity_role(ent)
        if result is None:
            continue
        role, pred_token, pred_lemma, pred_class = result

        slot_a = agency[(canonical, pred_lemma, pred_class, role)]
        slot_a["count"] += 1
        slot_a["articles"].add(aid)
        if slot_a["example_sent"] is None:
            slot_a["example_sent"] = sent_label

        # Track article-level presence per (entity, role) and per
        # (entity, predicate_class) for subjects. These give correct
        # set-union counts at output time (see note above).
        role_articles[(canonical, role)].add(aid)
        if role == "subject":
            subj_class_articles[(canonical, pred_class)].add(aid)

        # Track verbal-predicate arguments for directed-dyad extraction.
        # prep_target is treated like a direct object in dyad logic but
        # tracked separately so the breakdown is visible in the output.
        if role in {"subject", "object", "passive_subject", "prep_target"}:
            slot = verb_arg_map.setdefault(
                pred_token.i,
                {
                    "lemma": pred_lemma,
                    "class": pred_class,
                    "subjects": [],
                    "objects": [],
                    "passives": [],
                    "prep_targets": [],
                },
            )
            if role == "subject":
                slot["subjects"].append(canonical)
            elif role == "object":
                slot["objects"].append(canonical)
            elif role == "passive_subject":
                slot["passives"].append(canonical)
            else:  # prep_target
                slot["prep_targets"].append(canonical)

    # Extract directed dyads from this sentence
    for slot in verb_arg_map.values():
        subjects = slot["subjects"]
        key_prefix = (slot["lemma"], slot["class"])
        for s in subjects:
            for o in slot["objects"]:
                if s != o:
                    d = directed[(s, *key_prefix, o)]
                    d["dobj"] += 1
                    d["articles"].add(aid)
                    if d["example_sent"] is None:
                        d["example_sent"] = sent_label
            for o in slot["prep_targets"]:
                if s != o:
                    d = directed[(s, *key_prefix, o)]
                    d["prep"] += 1
                    d["articles"].add(aid)
                    if d["example_sent"] is None:
                        d["example_sent"] = sent_label
            for p in slot["passives"]:
                if s != p:
                    d = directed[(s, *key_prefix, p)]
                    d["passive"] += 1
                    d["articles"].add(aid)
                    if d["example_sent"] is None:
                        d["example_sent"] = sent_label

    # Co-occurrence
    unique_in_sent = list(dict.fromkeys(sentence_ents))
    for a, b in itertools.combinations(unique_in_sent, 2):
        cooc[frozenset({a, b})] += 1



# Write entities.csv


entities_rows = [
    {
        "entity": canonical,
        "label": LABEL_OVERRIDES.get(canonical, info["label"]),
        "mentions": info["mentions"],
        "articles": len(info["articles"]),
    }
    for canonical, info in entity_counts.items()
]

entities_df = (
    pd.DataFrame(entities_rows)
    .sort_values("mentions", ascending=False)
    .reset_index(drop=True)
)
entities_df_out = entities_df[entities_df["mentions"] >= MIN_MENTIONS_FOR_CSV]
entities_df_out.to_csv(OUTPUT_DIR / "entities.csv", index=False)
print(
    f"Wrote entities.csv ({len(entities_df_out)} entities; "
    f"{len(entities_df) - len(entities_df_out)} single-mention entities filtered)."
)



# Write cooccurrence.csv with PMI and PPMI


total_sentences = len(df)
cooc_rows = []
for pair, shared in cooc.items():
    if shared < MIN_COOC_SHARED:
        continue
    a, b = sorted(pair)
    count_a = len(entity_counts[a]["sentence_ids"])
    count_b = len(entity_counts[b]["sentence_ids"])
    p_ab = shared / total_sentences
    p_a = count_a / total_sentences
    p_b = count_b / total_sentences
    if p_a > 0 and p_b > 0:
        pmi = math.log2(p_ab / (p_a * p_b))
    else:
        pmi = 0.0
    ppmi = max(0.0, pmi)
    cooc_rows.append(
        {
            "entity_a": a,
            "entity_b": b,
            "shared_sentences": shared,
            "count_a": count_a,
            "count_b": count_b,
            "pmi": round(pmi, 3),
            "ppmi": round(ppmi, 3),
            "low_count": shared < LOW_COUNT_FLAG_THRESHOLD,
        }
    )

cooc_df = (
    pd.DataFrame(cooc_rows)
    .sort_values("shared_sentences", ascending=False)
    .reset_index(drop=True)
)
cooc_df.to_csv(OUTPUT_DIR / "cooccurrence.csv", index=False)
print(
    f"Wrote cooccurrence.csv ({len(cooc_df)} pairs with >= {MIN_COOC_SHARED}; "
    f"{(cooc_df['low_count']).sum()} flagged low_count)."
)



# Write agency.csv


agency_rows = [
    {
        "entity": e,
        "predicate": v,
        "predicate_class": c,
        "role": r,
        "count": d["count"],
        "articles": len(d["articles"]),
        "example_sent": d["example_sent"],
    }
    for (e, v, c, r), d in agency.items()
]
agency_df = (
    pd.DataFrame(agency_rows)
    .sort_values(["entity", "count"], ascending=[True, False])
    .reset_index(drop=True)
)
agency_df.to_csv(OUTPUT_DIR / "agency.csv", index=False)
print(f"Wrote agency.csv ({len(agency_df)} (entity, predicate, class, role) tuples).")



# Write agency_summary.csv with class breakdown and article counts
#
# Columns:
#   subj_action, subj_communicative, subj_mental, subj_other
#                            Subject counts split by verb class.
#   obj_total, pass_total, prep_target_total
#                            Patient roles. prep_target is the count of
#                            mentions where the entity is the target of
#                            an action verb via a preposition (e.g. fired
#                            at US, sanctioned on Iran).
#   nom_agent_total          Nominal agent count.
#   total                    Sum of all role-bearing mentions.
#   action_agency_ratio      (subj_action + nom_agent_total) / total.
#   communicative_share      subj_communicative / subj_total.
#   patient_ratio            (obj + pass + prep_target) / total.
#                            Now includes prep_target in numerator since
#                            prep targets are also things-acted-upon.
#   passive_share            pass / (obj + pass). Strict patient denominator
#                            (excludes prep_target) so this measures agent-
#                            suppression specifically.
#   *_articles               Number of distinct articles in which the
#                            corresponding role appears. A fragility check:
#                            if a count is spread across few articles, the
#                            finding is concentrated rather than systematic.


def build_pivot(df_, mask, index_col, columns_col, values_col, columns_order, prefix):
    if mask.any():
        piv = df_[mask].pivot_table(
            index=index_col,
            columns=columns_col,
            values=values_col,
            aggfunc="sum",
            fill_value=0,
        )
    else:
        piv = pd.DataFrame(index=pd.Index([], name=index_col))
    for c in columns_order:
        if c not in piv.columns:
            piv[c] = 0
    piv = piv[columns_order]
    piv.columns = [f"{prefix}{c}" for c in piv.columns]
    return piv


if len(agency_df):
    # --- Sentence-level counts ---
    subj_filter = agency_df["role"] == "subject"
    subj = build_pivot(
        agency_df, subj_filter, "entity", "predicate_class", "count",
        ["action", "communicative", "mental", "other"], "subj_",
    )
    subj["subj_total"] = subj.sum(axis=1)

    other_filter = agency_df["role"] != "subject"
    other_roles = build_pivot(
        agency_df, other_filter, "entity", "role", "count",
        ["passive_subject", "object", "prep_target", "nominal_agent"], "",
    )
    other_roles = other_roles.rename(columns={
        "passive_subject": "pass_total",
        "object": "obj_total",
        "prep_target": "prep_target_total",
        "nominal_agent": "nom_agent_total",
    })

    summary = subj.join(other_roles, how="outer").fillna(0).astype(int)
    summary["total"] = (
        summary["subj_total"] + summary["pass_total"]
        + summary["obj_total"] + summary["prep_target_total"]
        + summary["nom_agent_total"]
    )

    total_safe = summary["total"].astype(float).replace(0, float("nan"))
    subj_safe = summary["subj_total"].astype(float).replace(0, float("nan"))
    strict_patient = (
        (summary["obj_total"] + summary["pass_total"])
        .astype(float)
        .replace(0, float("nan"))
    )

    summary["action_agency_ratio"] = (
        (summary["subj_action"] + summary["nom_agent_total"]) / total_safe
    ).round(3)
    summary["communicative_share"] = (
        summary["subj_communicative"] / subj_safe
    ).round(3)
    summary["patient_ratio"] = (
        (summary["obj_total"] + summary["pass_total"] + summary["prep_target_total"])
        / total_safe
    ).round(3)
    summary["passive_share"] = (summary["pass_total"] / strict_patient).round(3)

    # --- Article-level fragility counts (distinct articles per role) ---
    # Use the set-union tracking dicts populated during the main loop.
    # The earlier attempt to derive these via pivot_table(aggfunc="sum")
    # on the agency dataframe was wrong: it added per-predicate article
    # counts so any article containing the same entity-role under more
    # than one predicate was counted multiple times.
    entities_in_summary = list(summary.index)

    def _per_role_arts(role):
        return pd.Series(
            {e: len(role_articles.get((e, role), set())) for e in entities_in_summary},
            name=role,
        )

    def _per_subj_class_arts(cls):
        return pd.Series(
            {e: len(subj_class_articles.get((e, cls), set())) for e in entities_in_summary},
            name=cls,
        )

    summary["subj_action_articles"] = _per_subj_class_arts("action")
    summary["subj_communicative_articles"] = _per_subj_class_arts("communicative")
    summary["subj_mental_articles"] = _per_subj_class_arts("mental")
    summary["subj_other_articles"] = _per_subj_class_arts("other")
    summary["pass_articles"] = _per_role_arts("passive_subject")
    summary["obj_articles"] = _per_role_arts("object")
    summary["prep_target_articles"] = _per_role_arts("prep_target")
    summary["nom_agent_articles"] = _per_role_arts("nominal_agent")

    art_cols = [c for c in summary.columns if c.endswith("_articles")]
    summary[art_cols] = summary[art_cols].fillna(0).astype(int)

    summary = summary.sort_values("total", ascending=False).reset_index()
    summary.to_csv(OUTPUT_DIR / "agency_summary.csv", index=False)
    print(f"Wrote agency_summary.csv ({len(summary)} entities).")
else:
    summary = pd.DataFrame()



# Write directed_pairs.csv
#
# Each row: (subject, verb, verb_class, object) with breakdown columns
# count_dobj   - subject V object (direct object: "Israel attacked Iran")
# count_prep   - subject V at/against/etc object ("Iran fired at US")
# count_passive - object was V'd by subject ("Iran was attacked by US")
# count_total  - sum of all three
# articles     - distinct articles containing the dyad
# example_sent - one source sentence ("<article_id>:<sentence_id>")


directed_rows = []
for (s, v, c, o), d in directed.items():
    total = d["dobj"] + d["prep"] + d["passive"]
    if total < MIN_DYAD_COUNT:
        continue
    directed_rows.append({
        "subject": s,
        "verb": v,
        "verb_class": c,
        "object": o,
        "count_dobj": d["dobj"],
        "count_prep": d["prep"],
        "count_passive": d["passive"],
        "count_total": total,
        "articles": len(d["articles"]),
        "example_sent": d["example_sent"],
    })

if directed_rows:
    directed_df = (
        pd.DataFrame(directed_rows)
        .sort_values("count_total", ascending=False)
        .reset_index(drop=True)
    )
else:
    directed_df = pd.DataFrame(columns=[
        "subject", "verb", "verb_class", "object",
        "count_dobj", "count_prep", "count_passive",
        "count_total", "articles", "example_sent",
    ])
directed_df.to_csv(OUTPUT_DIR / "directed_pairs.csv", index=False)
print(
    f"Wrote directed_pairs.csv "
    f"({len(directed_df)} directed dyads with count_total >= {MIN_DYAD_COUNT})."
)



# Console summary + sensitivity checks


print("\n=== Threshold sensitivity (entity mentions) ===")
for thresh in (2, 5, 10, 20):
    n = (entities_df["mentions"] >= thresh).sum()
    print(f"  >= {thresh:>2} mentions: {n} entities")

print("\n=== Role totals across corpus ===")
role_totals = agency_df.groupby("role")["count"].sum().to_dict()
for role in ("subject", "passive_subject", "object", "prep_target", "nominal_agent"):
    print(f"  {role:18s} {role_totals.get(role, 0)}")

print("\n=== Subject distribution by verb class ===")
subj_dist = (
    agency_df[agency_df["role"] == "subject"]
    .groupby("predicate_class")["count"].sum().to_dict()
)
subj_total_all = sum(subj_dist.values()) or 1
for cls in ("action", "communicative", "mental", "other"):
    n = subj_dist.get(cls, 0)
    pct = 100 * n / subj_total_all
    print(f"  {cls:18s} {n:>5}  ({pct:.1f}%)")

if len(summary):
    meaningful = summary[summary["total"] >= MIN_AGENCY_TOTAL]
    if len(meaningful):
        print(f"\n=== Top action-agency entities (total >= {MIN_AGENCY_TOTAL}) ===")
        top = meaningful.sort_values("action_agency_ratio", ascending=False).head(10)
        print(f"  {'entity':25s} {'action_ag':>10} {'comm_share':>11} {'total':>6}")
        for _, row in top.iterrows():
            comm = row["communicative_share"]
            comm_str = f"{comm:.2f}" if pd.notna(comm) else "  - "
            print(
                f"  {row['entity']:25s} "
                f"{row['action_agency_ratio']:>10.2f} "
                f"{comm_str:>11} "
                f"{row['total']:>6}"
            )

        print(f"\n=== Top patient-ratio entities (total >= {MIN_AGENCY_TOTAL}) ===")
        top_p = meaningful.sort_values("patient_ratio", ascending=False).head(10)
        print(f"  {'entity':25s} {'patient':>8} {'pass_share':>11} {'total':>6}")
        for _, row in top_p.iterrows():
            ps = row["passive_share"]
            ps_str = f"{ps:.2f}" if pd.notna(ps) else "  - "
            print(
                f"  {row['entity']:25s} "
                f"{row['patient_ratio']:>8.2f} "
                f"{ps_str:>11} "
                f"{row['total']:>6}"
            )

        # Article-level fragility check on the headline finding
        print("\n=== Fragility check (sentence count vs article spread) ===")
        print(f"  {'entity':12s} {'role':25s} {'sents':>6} {'arts':>6}")
        checks = [
            ("Iran", "subj_action", "subj_action_articles"),
            ("Iran", "obj_total", "obj_articles"),
            ("Iran", "prep_target_total", "prep_target_articles"),
            ("Trump", "subj_communicative", "subj_communicative_articles"),
            ("Israel", "subj_action", "subj_action_articles"),
            ("US", "subj_action", "subj_action_articles"),
        ]
        for ent, sent_col, art_col in checks:
            row = summary[summary["entity"] == ent]
            if row.empty:
                continue
            s = int(row[sent_col].iloc[0])
            a = int(row[art_col].iloc[0])
            print(f"  {ent:12s} {sent_col:25s} {s:>6} {a:>6}")


def probe_dyad(subj, obj, verb_class="action"):
    """Sum count_total across all verbs in the given class for (subj, obj)."""
    if not len(directed_df):
        return (0, 0, 0, 0)
    mask = (directed_df["subject"] == subj) & (directed_df["object"] == obj)
    if verb_class is not None:
        mask &= directed_df["verb_class"] == verb_class
    sub = directed_df.loc[mask]
    return (
        int(sub["count_total"].sum()),
        int(sub["count_dobj"].sum()),
        int(sub["count_prep"].sum()),
        int(sub["count_passive"].sum()),
    )


print("\n=== Directional asymmetry probes (action verbs only) ===")
print(f"  {'subject -> object':30s} {'total':>6} {'dobj':>5} {'prep':>5} {'pass':>5}")
probe_pairs = [
    ("US", "Iran"), ("Iran", "US"),
    ("Israel", "Iran"), ("Iran", "Israel"),
    ("Israel", "Hezbollah"), ("Hezbollah", "Israel"),
]
for a, b in probe_pairs:
    tot, dobj, prep, passive = probe_dyad(a, b)
    label = f"{a} -> {b}"
    print(f"  {label:30s} {tot:>6} {dobj:>5} {prep:>5} {passive:>5}")


print("\nDone. Output files in", OUTPUT_DIR)
print("  entities.csv  cooccurrence.csv  agency.csv")
print("  agency_summary.csv  directed_pairs.csv")