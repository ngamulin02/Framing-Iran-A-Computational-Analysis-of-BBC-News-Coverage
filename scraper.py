import csv
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

from articles import ARTICLES

HEADERS = {"User-Agent": "Mozilla/5.0"}

BLACKLIST = [
    "Copyright",
    "All rights reserved",
    "The BBC is not responsible for the content of external sites",
    "Read about our approach to external linking",
]

OUTPUT_FILE = "bbc_sentences.csv"


def scrape_article(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article") or soup

    clean_paragraphs = []
    for p in article.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) < 40:
            continue
        if any(b.lower() in text.lower() for b in BLACKLIST):
            continue
        clean_paragraphs.append(text)

    return " ".join(clean_paragraphs)


def main():
    rows = []
    for article_id, (title, url) in enumerate(ARTICLES.items(), start=1):
        print(f"Scraping: {title}")
        article_text = scrape_article(url)
        sentences = sent_tokenize(article_text)
        for i, sentence in enumerate(sentences, start=1):
            rows.append({"article_id": article_id, "title": title, "link": url, "sentence_id": i, "sentence": sentence})

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["article_id", "title", "link", "sentence_id", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} sentences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
