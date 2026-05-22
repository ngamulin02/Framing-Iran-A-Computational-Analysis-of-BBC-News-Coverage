# Framing Iran: A Computational Analysis of BBC News Coverage

## 1. Introduction

This project examines how the BBC frames Iran in its online news coverage of the 2026 Iran war through a computational analysis of 93 articles published in April and May 2026. Rather than relying on close reading alone, the analysis applies Named Entity Recognition (NER), entity co-occurrence statistics, and syntactic agency extraction to surface patterns across a corpus of 3,588 sentences. The central questions are: which actors dominate coverage, how are they relationally paired in discourse, and who is framed as an agent versus a patient of action?


## 2. Methodology

### scraper.py

`scraper.py` builds the raw corpus. It reads a dictionary of article titles and BBC URLs defined in `articles.py` and iterates through each entry. For every article it sends an HTTP GET request, parses the HTML with BeautifulSoup, and extracts text from all `<p>` tags inside the `<article>` element (falling back to the full page if no article tag is found). Two filters clean the output: paragraphs shorter than 40 characters are discarded as likely navigation fragments or captions, and a blacklist removes standard BBC footer text (copyright notices, external-linking disclaimers). The remaining paragraphs are concatenated and sentence-tokenised using NLTK's Punkt model. Each sentence is written as a row to `bbc_sentences.csv` with columns for article ID, headline, source URL, sentence position, and sentence text.

### main.py

`main.py` reads `bbc_sentences.csv` and runs three complementary analyses, writing results to the `outputs/` directory.

**Named Entity Recognition.** The spaCy `en_core_web_lg` pipeline (falling back to `en_core_web_sm` if unavailable) is applied to every sentence in batch. Five entity types are retained: PERSON, ORG, GPE, LOC, and NORP. Surface forms are normalised through a canonical alias table (e.g. "United States", "U.S.", and "America" all resolve to "US"; "Revolutionary Guards" resolves to "IRGC") and minor punctuation is stripped. Entities appearing fewer than twice across the corpus are excluded from the final `entities.csv`.

**Entity co-occurrence.** For each sentence the deduplicated set of recognised entities is enumerated and all unordered pairs are recorded. Pair counts are aggregated across the full corpus. Only pairs co-occurring in three or more sentences are retained. For each qualifying pair, Pointwise Mutual Information (PMI) is computed as a measure of associative strength beyond what chance co-occurrence would predict. Results are written to `cooccurrence.csv`.

**Agency analysis.** For each entity mention the dependency parse is inspected to determine the grammatical role of the entity's root token: active subject (`nsubj`), passive subject (`nsubjpass`), or direct object (`dobj`). Prepositional-agent constructions (`pobj` of an `agent` head) are treated as active subjects. Counts are aggregated per entity and written to `agency.csv`. A summary table (`agency_summary.csv`) adds derived ratios: `agency_ratio` (proportion of appearances as active subject), `patient_ratio` (proportion as object or passive subject), and `passive_share` (proportion of patient appearances that are passive rather than direct object).


## 3. Results

### 3.1 Entity landscape

The corpus yielded 403 unique entities with two or more mentions, generating a heavily skewed distribution. The fifteen most-mentioned entities are shown below.

| Entity | Type | Mentions | Articles |
|---|---|---|---|
| Iran | GPE | 919 | 89 |
| US | GPE | 793 | 85 |
| Trump | PERSON | 377 | 69 |
| Iranian | NORP | 345 | 80 |
| Israel | GPE | 246 | 61 |
| Strait of Hormuz | LOC | 206 | 64 |
| Tehran | GPE | 163 | 65 |
| Lebanon | GPE | 136 | 28 |
| Israeli | NORP | 107 | 41 |
| Hezbollah | ORG | 98 | 22 |
| UK | GPE | 94 | 35 |
| Pakistan | GPE | 73 | 41 |
| Gulf | LOC | 66 | 37 |
| Middle East | LOC | 59 | 34 |
| Centcom | ORG | 57 | 19 |

*Table 1. Top 15 entities by mention count.*

Iran is the most mentioned entity (919 mentions across 89 of 93 articles), closely followed by the US (793, 85 articles). Trump is the most prominent individual (377 mentions, 69 articles). The Strait of Hormuz, a geographic feature rather than a state actor, appears in 64 of 93 articles (more than Israel or Lebanon) signalling a strong maritime-security orientation in the corpus.

### 3.2 Co-occurrence patterns

344 entity pairs met the threshold of three or more shared sentences. The ten most frequent pairings are listed below. PMI (Pointwise Mutual Information) quantifies how much more often a pair co-occurs than would be expected if both entities appeared independently; positive values indicate a stronger-than-chance association, negative values the opposite.

| Entity A | Entity B | Shared sentences | PMI |
|---|---|---|---|
| Iran | US | 343 | 1.006 |
| Iran | Trump | 151 | 0.782 |
| Trump | US | 137 | 0.862 |
| Iranian | US | 130 | 0.959 |
| Israel | US | 100 | 1.131 |
| Iran | Israel | 95 | 0.836 |
| Iran | Strait of Hormuz | 88 | 0.887 |
| Iran | Iranian | 65 | -0.262 |
| Strait of Hormuz | US | 65 | 0.671 |
| Tehran | US | 64 | 0.985 |

*Table 2. Top 10 co-occurring entity pairs.*

The Iran-US dyad dominates at 343 co-occurrences, more than double the next most frequent pair. Of the top ten pairs, eight involve Iran (or a referring form) on one side and the US or Trump on the other. High-PMI pairs outside this axis include Hezbollah-Lebanon (PMI 3.402), Hezbollah-Israeli (3.166) and Israel-Lebanese (3.135). 

### 3.3 Agency analysis

The agency summary for the most active entities is shown below. The table is restricted to entities with at least 30 total appearances and a substantively interpretable ratio. Several categories of entity are excluded. First, many lower-frequency entities in the full dataset carry an agency ratio of exactly 1.0, meaning they never appear as a grammatical object or passive subject, but their totals are small enough (typically under 20) that the ratio reflects sparse data rather than a consistent framing pattern. These are omitted to avoid over-interpreting noise. Second, Tehran and Iranians are excluded as near-synonymous surface forms of Iran that would duplicate rather than add to the picture. 

| Entity | Agency ratio | Patient ratio | Total appearances |
|---|---|---|---|
| Netanyahu | 0.967 | 0.033 | 30 |
| Trump | 0.954 | 0.046 | 260 |
| Israel | 0.921 | 0.079 | 114 |
| US | 0.869 | 0.131 | 244 |
| Iran | 0.809 | 0.191 | 320 |
| Hezbollah | 0.643 | 0.357 | 42 |
| UK | 0.600 | 0.400 | 30 |
| Lebanon | 0.387 | 0.613 | 31 |
| Strait of Hormuz | 0.206 | 0.794 | 68 |

*Table 3. Agency and patient ratios for major entities.*

Trump (0.954), Netanyahu (0.967), and Israel (0.921) have the highest agency ratios among major actors. They are overwhelmingly framed as initiators of action. Iran has a moderately high agency ratio (0.809) but also accumulates the largest absolute patient count (54 direct object appearances), meaning it is simultaneously the most written-about subject and the most written-about target. Lebanon (0.387) and the Strait of Hormuz (0.206) are predominantly patient entities; things that are acted upon rather than actors in their own right.

## 4. Discussion

The most consistent pattern in the data is the dominance of the Iran-US dyad. Iran appears alongside the US in 343 sentences, and no other relationship comes close in raw frequency. This positions Iran as legible in BBC coverage primarily through the lens of American policy rather than as an autonomous subject of news. Trump's exceptional presence (377 mentions, the most of any individual) reinforces this: the US side of the dyad is further personalised around a single figure, making the dominant frame not just bilateral but highly individualised.

The appearance of the Strait of Hormuz in 64 articles (comparable in breadth to Israel) combined with a very low agency ratio (0.206) confirms that it functions in the discourse as a contested space that states act upon rather than an actor in its own right, which is unsurprising for a piece of geography. Its strong co-occurrence with the US (65 sentences) suggests it is framed as an object of American strategic interest.

The agency ratios reveal a clear asymmetry. Israel, Netanyahu, and Trump are overwhelmingly active grammatical subjects, while Lebanon, Hezbollah, and the Strait of Hormuz carry high patient ratios. Iran sits in the middle: active in most sentences where it appears, yet accumulating more object appearances in absolute terms than any other entity. This dual role, simultaneously the initiating subject of crises and the target of sanctions, strikes, and blockades, reflects the structurally contested position Iran occupies in the coverage.

A smaller but distinct cluster around Lebanon, Hezbollah, Israel, and Beirut (high mutual PMI values) points to a secondary narrative thread concerning the Lebanon war running alongside the primary US-Iran frame. 
