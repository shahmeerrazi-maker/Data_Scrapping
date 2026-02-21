# medium_scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
import os
from tqdm import tqdm
import logging
import sys

# -------------------------
# Configuration
# -------------------------
INPUT_CSV = "input_urls.csv"          # put your file here (1 column named 'url')
OUTPUT_CSV = "scrapping_results.csv"
CHECKPOINT_EVERY = 500                # persist every N processed articles
MAX_WORKERS = 10                      # parallel threads (start modest; increase if stable)
REQUEST_TIMEOUT = 15
RETRY_ON_FAIL = 3
MIN_SLEEP = 0.5
MAX_SLEEP = 1.5

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler(sys.stdout)]
)

USER_AGENTS = [
    # a small rotation of UA strings
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
]

# -------------------------
# Helper functions
# -------------------------
def random_sleep():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

def safe_get(url, session, retries=RETRY_ON_FAIL):
    for attempt in range(retries):
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            r = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.text
            else:
                logging.warning("Non-200 %s for %s (attempt %s)", r.status_code, url, attempt+1)
        except Exception as e:
            logging.warning("Error fetching %s: %s (attempt %s)", url, str(e), attempt+1)
        random_sleep()
    return None

def extract_text_from_soup(soup):
    # Best-effort: read soup article tags, then fallback
    texts = []
    article = soup.find('article')
    if article:
        for p in article.find_all('p'):
            texts.append(p.get_text(strip=True))
    else:
        # fallback to main content selectors
        for sel in ['section', 'div']:
            for block in soup.select(f"{sel}"):
                # Heuristic: pick blocks with many <p>
                ps = block.find_all('p')
                if len(ps) > 2:
                    texts = [p.get_text(strip=True) for p in ps]
                    if texts:
                        break
            if texts:
                break
    return "\n\n".join(texts).strip()

def count_external_links(soup, base_domain):
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        parsed = urlparse(href)
        if parsed.scheme in ('http', 'https'):
            if parsed.netloc and base_domain not in parsed.netloc:
                links.add(href)
    return len(links)

def extract_image_urls(soup, base_url):
    imgs = []
    for img in soup.find_all('img', src=True):
        src = img.get('src')
        if src and src.startswith('//'):
            src = 'https:' + src
        if src and src.startswith('/'):
            src = urljoin(base_url, src)
        if src:
            imgs.append(src)
    # de-duplicate
    return list(dict.fromkeys(imgs))

def extract_meta_content(soup, name):
    tag = soup.find("meta", {"name": name}) or soup.find("meta", {"property": name})
    if tag:
        return tag.get("content", "").strip()
    return ""

def extract_author(soup):
    # Medium has author link area; try common patterns
    author = ""
    author_url = ""
    a = soup.find("a", {"class": re.compile(r".*author.*", re.I)}) or soup.find("a", {"rel": "author"})
    if a:
        author = a.get_text(strip=True)
        author_url = a.get('href') or ""
    # fallback to meta
    if not author:
        author = extract_meta_content(soup, "author") or extract_meta_content(soup, "og:article:author")
    return author.strip(), author_url

def extract_claps(soup):
    # claps are sometimes in <button> or as meta; try to find numbers in text like 'K' or full numbers
    # NOTE: this can miss some pages due to JS rendering
    text = soup.get_text(" ")
    m = re.search(r'(\d{1,3}(?:[,\d]{0,}\d)?)(?:\s*clap|\s*claps)', text, re.I)
    if m:
        return m.group(1).replace(",", "")
    # sometimes 'K' shorthand e.g. 3.4K
    m2 = re.search(r'([0-9]+(?:\.[0-9])?)\s*K', text, re.I)
    if m2:
        try:
            val = float(m2.group(1)) * 1000
            return str(int(val))
        except:
            pass
    return ""

def extract_reading_time(soup):
    # try meta
    rt = extract_meta_content(soup, "twitter:data1") or extract_meta_content(soup, "readingTime")
    if rt:
        return rt
    # fallback: search text like '5 min read'
    text = soup.get_text(" ")
    m = re.search(r'(\d+)\s+min', text, re.I)
    return m.group(1) + " min" if m else ""

def keywords_from_text(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop = set(["those","which","there","their","about","would","could","should","these","other","where","when","what","this","that","with","from","have","will","your"])
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w,c in sorted_words[:top_n]]

def parse_article(url, session):
    result = {
        "url": url,
        "title": "",
        "subtitle": "",
        "text": "",
        "num_images": 0,
        "image_urls": "",
        "num_external_links": 0,
        "author_name": "",
        "author_url": "",
        "claps": "",
        "reading_time": "",
        "keywords": ""
    }
    html = safe_get(url, session)
    if not html:
        return result

    soup = BeautifulSoup(html, "lxml")

    # Title
    title = extract_meta_content(soup, "og:title") or (soup.title.string if soup.title else "")
    result["title"] = title.strip()

    # Subtitle (try description meta or subtitle tags)
    subtitle = extract_meta_content(soup, "description") or extract_meta_content(soup, "og:description")
    result["subtitle"] = subtitle.strip()

    # Text (best-effort)
    text = extract_text_from_soup(soup)
    result["text"] = text

    # Images
    imgs = extract_image_urls(soup, url)
    result["num_images"] = len(imgs)
    result["image_urls"] = "|".join(imgs)

    # External links
    base_domain = urlparse(url).netloc
    result["num_external_links"] = count_external_links(soup, base_domain)

    # Author
    a_name, a_url = extract_author(soup)
    result["author_name"] = a_name
    result["author_url"] = a_url

    # Claps & reading time
    result["claps"] = extract_claps(soup)
    result["reading_time"] = extract_reading_time(soup)

    # Keywords
    result["keywords"] = ",".join(keywords_from_text(text, top_n=10))

    return result

# -------------------------
# Main flow
# -------------------------
def load_input(input_csv):
    df = pd.read_csv(input_csv)
    if "url" not in df.columns:
        # assume first column contains URLs
        df.columns = ["url"]
    return df

def save_checkpoint(results, processed_idx):
    # append to output CSV (create if not exists)
    df = pd.DataFrame(results)
    if os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False)
    logging.info("Saved checkpoint: %s rows appended. Processed index: %s", len(df), processed_idx)

def get_start_index():
    # resume: count rows already written in OUTPUT_CSV
    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            return existing.shape[0]
        except Exception as e:
            logging.warning("Could not read existing output csv: %s", str(e))
            return 0
    return 0

def main():
    df_urls = load_input(INPUT_CSV)
    total = df_urls.shape[0]
    start_idx = get_start_index()
    logging.info("Total urls: %s, will resume from index: %s", total, start_idx)

    session = requests.Session()
    results_buffer = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        pbar = tqdm(total=total, initial=start_idx, desc="Scraping")
        for idx in range(start_idx, total):
            url = str(df_urls.loc[idx, "url"]).strip()
            # submit parse task
            futures[executor.submit(parse_article, url, session)] = idx

            # throttle submission so we don't submit all at once
            if len(futures) >= MAX_WORKERS * 4:
                for fut in as_completed(list(futures.keys())):
                    i = futures.pop(fut)
                    try:
                        res = fut.result()
                        results_buffer.append(res)
                        processed += 1
                        pbar.update(1)
                    except Exception as e:
                        logging.exception("Error in worker: %s", e)
                    # checkpoint periodically
                    if len(results_buffer) >= CHECKPOINT_EVERY:
                        save_checkpoint(results_buffer, start_idx + processed)
                        results_buffer = []
                random_sleep()

        # finish remaining futures
        for fut in as_completed(list(futures.keys())):
            try:
                res = fut.result()
                results_buffer.append(res)
                processed += 1
                pbar.update(1)
            except Exception as e:
                logging.exception("Worker final error: %s", e)

        # final save
        if results_buffer:
            save_checkpoint(results_buffer, start_idx + processed)
            results_buffer = []

    logging.info("Done scraping. Total processed this run: %s", processed)

if __name__ == "__main__":
    main()
