# Framing Iran: A Computational Analysis of BBC News Coverage

## 1. Introduction

This project examines how the BBC frames Iran in its online news coverage of the 2026 Iran war through a computational analysis of 93 articles published in April and May 2026. The pipeline applies Named Entity Recognition (NER), entity co-occurrence statistics, and a syntactic agency model to surface patterns across a corpus of 3,588 sentences. The central questions are: which actors dominate coverage, how are they relationally paired in discourse, and who is framed as an agent of action versus a target of it.


## 2. Methodology

### scraper.py

`scraper.py` builds the raw corpus. It reads a dictionary of article titles and BBC URLs defined in `articles.py` and iterates through each entry. For every article it sends an HTTP GET request, parses the HTML with BeautifulSoup, and extracts text from all `<p>` tags inside the `<article>` element (falling back to the full page if no article tag is found). Two filters clean the output: paragraphs shorter than 40 characters are discarded as likely navigation fragments or captions, and a blacklist removes standard BBC footer text. The remaining paragraphs are concatenated and sentence-tokenised using NLTK's Punkt model. Each sentence is written as a row to `bbc_sentences.csv` with columns for article ID, headline, source URL, sentence position, and sentence text.

### main.py

`main.py` reads `bbc_sentences.csv` and runs three complementary analyses, writing results to the `outputs/` directory.

**Named Entity Recognition.** The spaCy `en_core_web_lg` pipeline is applied to every sentence in batch. Five entity types are retained: PERSON, ORG, GPE, LOC, and NORP. Surface forms are normalised through a canonical alias table. The alias table has two layers. The first folds explicit name variants ("United States", "U.S.", "America" → US; "Revolutionary Guards" → IRGC; "Benjamin Netanyahu" → Netanyahu). The second, gated by the `COLLAPSE_METONYMS` flag and on by default, folds NORP nationality forms and metonymic capitals into their state: "Iranian" and "Tehran" both resolve to Iran, "Israeli" and "Jerusalem" to Israel, "Lebanese" and "Beirut" to Lebanon, and so on. For a framing analysis these refer to the same actor; the flag exists so the surface-form view is recoverable when needed. Entities appearing fewer than twice across the corpus are excluded from the final `entities.csv`.

**Entity co-occurrence.** For each sentence the deduplicated set of recognised entities is enumerated and all unordered pairs are recorded. Pair counts are aggregated across the full corpus. Only pairs co-occurring in three or more sentences are retained. For each qualifying pair, Pointwise Mutual Information (PMI) and Positive PMI (PPMI) are computed; pairs with fewer than ten shared sentences are flagged as `low_count` because PMI is unstable for sparse co-occurrences. Results are written to `cooccurrence.csv`.

**Agency analysis.** For each entity mention the dependency parse is inspected to determine the grammatical role of the entity's root token. Six roles are distinguished. Four verbal roles cover the standard argument positions: active subject (`nsubj`), passive subject (`nsubjpass`), direct object (`dobj`), and passive agent (`pobj` of a verbal `agent` head, as in "X was attacked by ENT"). A fifth verbal role, `prep_target`, captures entities that are the prepositional object of a target-introducing preposition (`against`, `at`, `into`, `toward`, `on`, `over`) whose head is an action verb; the compound case is also covered one level deep so that "at US bases" credits US, not just "bases". `prep_target` is gated on the verb being action-class so that "comment on Iran" and "talks on Iran's program" do not fire. A sixth role, `nominal_agent`, captures agency expressed through deverbal action nouns: "Iran's strike" (possessive of action noun), "the attack by Iran" (pobj of `by` attached to an action noun), and (for NORP entities) "Iranian strike" (amod of action noun) all register as nominal agents. Action nouns are drawn from a hand-curated list of military and coercive deverbals (strike, attack, raid, bombing, sanction, deployment, intervention, blockade, etc.).

For each role-bearing mention the governing predicate is also classified into one of four classes against hand-curated lexicons: **action** (attack, strike, kill, bomb, launch, sanction, impose, intercept, deploy, etc.), **communicative** (say, tell, announce, declare, warn, threaten, accuse, deny, etc.), **mental** (believe, want, fear, expect, etc.), and **other** for everything else (light verbs, copulas, intransitives). Counts are aggregated per entity and written to `agency.csv` along with the article count and an example sentence id (`"<article_id>:<sentence_id>"`) for each tuple.

A summary table (`agency_summary.csv`) breaks subject counts down by predicate class and derives four ratios. `action_agency_ratio` is the proportion of role-bearing mentions where the entity is the subject of an action verb or a nominal agent of an action noun; it isolates "doing things in the world" from "being quoted in an article". `communicative_share` is the proportion of subject mentions that fall on communicative verbs; a high value indicates the entity functions primarily as a quoted source. `patient_ratio` is the proportion of mentions as direct object, passive subject, or prepositional target. `passive_share` is the proportion of strict patient mentions (direct object plus passive subject) that are passive rather than direct-object, since the passive construction is the canonical site of agent suppression.


## 3. Results

### 3.1 Entity landscape

The corpus yielded 388 unique entities with two or more mentions after NORP and metonym collapse. The fifteen most-mentioned entities are shown below.

| Entity | Mentions | Articles |
|---|---|---|
| Iran | 1,475 | 90 |
| US | 896 | 85 |
| Trump | 377 | 69 |
| Israel | 365 | 71 |
| Strait of Hormuz | 206 | 64 |
| Lebanon | 180 | 28 |
| Pakistan | 144 | 48 |
| UK | 130 | 39 |
| Hezbollah | 98 | 22 |
| Gulf | 66 | 37 |
| Middle East | 59 | 34 |
| Centcom | 57 | 19 |
| Vance | 56 | 10 |
| IRGC | 49 | 18 |
| Netanyahu | 43 | 18 |

*Table 1. Top 15 entities by mention count.*

Iran is the dominant entity at 1,475 mentions across 90 of 93 articles. This figure is the result of folding "Iranian" and "Tehran" into the same entity. The same operation produces clearer counts for Israel (365), Lebanon (180), Pakistan (144), and the UK (130). Trump is the most-mentioned individual at 377 mentions in 69 articles. The Strait of Hormuz appears in 64 of 93 articles, still more than any single state actor outside Iran/US/Israel/Lebanon, signalling the corpus's strong maritime-security framing.

### 3.2 Co-occurrence patterns

269 entity pairs met the threshold of three or more shared sentences. The ten most frequent pairings are listed below.

| Entity A | Entity B | Shared sentences | PMI |
|---|---|---|---|
| Iran | US | 504 | 0.907 |
| Iran | Trump | 190 | 0.582 |
| Iran | Israel | 167 | 0.599 |
| Israel | US | 153 | 1.101 |
| Trump | US | 145 | 0.820 |
| Iran | Strait of Hormuz | 116 | 0.755 |
| Israel | Lebanon | 92 | 2.754 |
| Iran | Pakistan | 82 | 0.890 |
| Strait of Hormuz | US | 68 | 0.613 |
| Hezbollah | Israel | 64 | 2.869 |

*Table 2. Top 10 co-occurring entity pairs.*

The Iran-US dyad dominates at 504 sentences. Iran-Trump (190) and Iran-Israel (167) are the next most prominent pairings. High-PMI pairs outside the central axis include Hezbollah-Lebanon (3.643), Hezbollah-Israel (2.869), and Israel-Lebanon (2.754); their elevated PMI partly reflects that Hezbollah and Lebanon are mentioned together more often than not (Hezbollah appears in 97 sentences, 51 of which also mention Lebanon).

### 3.3 Agency analysis

The agency summary for the most active entities is shown below. The table is restricted to entities with at least 30 total role-bearing appearances, ordered by `action_agency_ratio`.

| Entity | Action agency | Comm. share | Patient ratio | Passive share | Total |
|---|---|---|---|---|---|
| Israel | 0.485 | 0.191 | 0.111 | 0.000 | 171 |
| US | 0.257 | 0.172 | 0.151 | 0.132 | 284 |
| Hezbollah | 0.143 | 0.148 | 0.357 | 0.400 | 42 |
| Iran | 0.130 | 0.308 | 0.183 | 0.107 | 447 |
| UK | 0.125 | 0.167 | 0.406 | 0.308 | 32 |
| Vance | 0.079 | 0.233 | 0.211 | 0.375 | 38 |
| Centcom | 0.029 | 0.886 | 0.000 | — | 35 |
| Trump | 0.019 | 0.665 | 0.049 | 0.333 | 264 |
| Netanyahu | 0.000 | 0.552 | 0.033 | 0.000 | 30 |
| Lebanon | 0.000 | 0.200 | 0.583 | 0.286 | 36 |
| Strait of Hormuz | 0.000 | 0.000 | 0.794 | 0.130 | 68 |

*Table 3. Action agency, communicative share, patient ratio, and passive share for major entities.*

Israel has the highest action-agency ratio in the corpus at 0.485, substantially ahead of the US (0.257) and roughly four times that of Iran (0.130). Of Israel's 171 role-bearing mentions, 41 are active subjects of action verbs (most commonly *launch* (18), *attack* (13), and *kill*/*target*/*hit* (2 each)), and 42 are nominal agents (the deverbal patterns *attack* (16), *strike* (13), *offensive* (5), *campaign* (4), and *operation* (2)). The combination of these two sources, both expressing Israeli action, drives the ratio. Israel's passive share is zero: it is never positioned as a passive grammatical subject in the corpus. The collapsing of NORP "Israeli" into Israel materially affects this measure.

The US ranks second at 0.257. It is the most-active state actor when measured by verbal action subjects alone (65, compared to Israel's 41), but its larger denominator (284 total mentions versus Israel's 171) brings the ratio down. Its most frequent action-subject verbs are *attack* (14), *launch* (13), *enforce* (8), and *intercept* (6). 

Iran's profile is the most distinctive of the major actors. Total role-bearing mentions: 447, by far the largest in the corpus. But of these, only 40 are subjects of action verbs and 18 are nominal agents, yielding an action-agency ratio of 0.130, lower than every other major state actor. Iran's communicative share (0.308) is by contrast almost twice as high as Israel's or the US's, reflecting 107 verbal subject appearances with communicative verbs (most often *say* (26), *agree* (19), *respond* (11), *deny* (10), *insist* (6)). The most frequent verbs taking Iran as direct object are *attack* (12), *accuse* (9), *infuriate* (3), and *urge* (3). Iran's *prep_target* mentions are dominated by *launch* (6) and one instance of *shoot*. 

Trump and Netanyahu remain near-pure speakers. Trump's 264 role-bearing mentions include 248 verbal subjects, of which 165 are with communicative verbs (88 with *say* alone) and only 2 with action verbs. Netanyahu shows the same pattern at lower volume: 16 of 29 subject mentions are communicative, 0 are action. 

Lebanon (patient ratio 0.583) and the Strait of Hormuz (0.794) are predominantly patient entities. The Strait's role profile remains the cleanest: 47 direct-object mentions (most often *reopen* (11), *transit* (5), *block* (4), *close* (4)), 7 passive subjects, and no action subjects at all. 


## 4. Discussion

Three findings hold up cleanly across methodology and warrant being stated as the central results.

The first is the dominance of the Iran-US dyad in news framing. Iran and the US co-occur in 504 sentences, more than triple the next most frequent pair. This is not by itself a framing claim. Both states are participants in the events being reported, so high co-occurrence is partially expected, but the lopsidedness (US mentions appearing more often with Iran than alone) confirms that Iran's legibility in the corpus runs primarily through American policy.

The second is the asymmetric action-agency profile. Israel is the most active grammatical agent of action in the corpus (action-agency 0.485), ahead of the US (0.257), with Iran's action-agency considerably lower (0.130). Trump and Netanyahu, the two most-mentioned individuals on the Western side, have action-agency ratios near zero (0.019 and 0.000); their prominence is the prominence of a quoted source rather than a depicted actor. The verb-class split and the nominalisation detection together produce a picture that conflating subjects with subjects-of-action would obscure: it is not that "the West" dominates active grammatical agency in this corpus, but specifically that Israel does, while the US's prominence is divided between depicted action and reported speech, and individual Western political figures contribute almost no depicted action at all.

A secondary cluster around Lebanon and Hezbollah confirms a smaller narrative thread on the northern theatre alongside the primary US-Iran frame. The Strait of Hormuz remains the cleanest patient entity in the data, geography functioning as a contested space that states act upon. Its 64-article spread, comparable to Israel's, makes clear that the corpus's maritime-security framing is not concentrated in a few pieces but distributed across most coverage.
