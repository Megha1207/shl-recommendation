import json
import re
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.sync_api import sync_playwright
from tqdm import tqdm
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com/products/product-catalog/"
DOMAIN = "https://www.shl.com"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TEST_TYPE_FULL = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations"
}

PAGINATION_STEP = 12


# ---------------------------------------------------------
# 1. Scrape catalog table
# ---------------------------------------------------------
def scrape_table_type(page, type_param: int, seen_urls: set) -> list:
    products = []
    start = 0
    consecutive_empty = 0

    while True:
        url = f"{BASE_URL}?start={start}&type={type_param}"
        print(f"  Fetching: {url}")

        page.goto(url, timeout=60000)
        page.wait_for_timeout(2500)

        soup = BeautifulSoup(page.content(), "html.parser")
        new_rows = 0

        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if not cells:
                continue

            link_tag = cells[0].find("a", href=True)
            if not link_tag:
                continue

            href = link_tag["href"]
            if "product-catalog/view" not in href:
                continue

            name = link_tag.get_text(strip=True)
            if not name:
                continue

            raw_url = DOMAIN + href if href.startswith("/") else href
            canonical_url = raw_url.replace(
                "shl.com/solutions/products", "shl.com/products"
            ).rstrip("/")

            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            remote_support  = "Yes" if len(cells) > 1 and cells[1].find("img") else "No"
            adaptive_support = "Yes" if len(cells) > 2 and cells[2].find("img") else "No"

            letters = [
                span.get_text(strip=True)
                for span in (cells[3].find_all("span") if len(cells) > 3 else [])
                if span.get_text(strip=True) in TEST_TYPE_FULL
            ]

            products.append({
                "name": name,
                "url": canonical_url,
                "test_types": [TEST_TYPE_FULL[l] for l in letters],
                "remote_support": remote_support,
                "adaptive_support": adaptive_support,
            })
            new_rows += 1

        print(f"    → new rows: {new_rows} | total so far: {len(products)}")

        if new_rows == 0:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
        else:
            consecutive_empty = 0

        start += PAGINATION_STEP

    return products


def get_catalog_products() -> list:
    print("\nCollecting products from catalog...\n")
    all_products = []
    seen_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=HEADERS["User-Agent"])
        page = context.new_page()

        for type_param in [1, 2]:
            print(f"\n--- type={type_param} ---")
            products = scrape_table_type(page, type_param, seen_urls)
            all_products.extend(products)
            print(f"  Subtotal: {len(all_products)}")

        browser.close()

    print(f"\nTotal products collected: {len(all_products)}")
    return all_products


# ---------------------------------------------------------
# 2. Enrich with Playwright (JS-rendered pages)
# ---------------------------------------------------------
def enrich_product_details(session, product: dict, retries: int = 3) -> dict:
    url = product["url"]

    for attempt in range(retries):
        try:
            response = session.get(url, timeout=20)

            if response.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue

            if response.status_code != 200:
                product.setdefault("description", "")
                product.setdefault("duration", None)
                return product

            soup = BeautifulSoup(response.text, "html.parser")

            # ---------------------------
            # Description
            # Find the h4 "Description" then grab the next sibling paragraph
            # ---------------------------
            description = ""
            for h4 in soup.find_all("h4"):
                if h4.get_text(strip=True).lower() == "description":
                    # Next sibling element contains the description text
                    sibling = h4.find_next_sibling()
                    if sibling:
                        description = sibling.get_text(separator=" ", strip=True)
                    # If no sibling tag, try next string node
                    if not description:
                        next_el = h4.find_next(["p", "div", "span"])
                        if next_el:
                            description = next_el.get_text(separator=" ", strip=True)
                    break

            # ---------------------------
            # Duration
            # Find h4 "Assessment length" then parse "= N" from next sibling
            # Format: "Approximate Completion Time in minutes = 13"
            # ---------------------------
            duration = None
            for h4 in soup.find_all("h4"):
                if "assessment length" in h4.get_text(strip=True).lower():
                    sibling = h4.find_next_sibling()
                    if sibling:
                        text = sibling.get_text(separator=" ", strip=True)
                    else:
                        next_el = h4.find_next(["p", "div", "span"])
                        text = next_el.get_text(separator=" ", strip=True) if next_el else ""

                    # Match "= 13" or "13 minutes"
                    match = re.search(r"=\s*(\d+)", text)
                    if not match:
                        match = re.search(r"(\d+)\s*min", text, re.I)
                    if match:
                        duration = int(match.group(1))
                    break

            product["description"] = description
            product["duration"] = duration
            return product

        except Exception as e:
            if attempt == retries - 1:
                product.setdefault("description", "")
                product.setdefault("duration", None)
                return product
            time.sleep(2 ** attempt)

    return product


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------
def main():
    products = get_catalog_products()

    if len(products) < 377:
        print(f"\nWARNING: Only {len(products)} products — expected 377+")
    else:
        print(f"\nTarget met: {len(products)} products")

    print("\nFetching product detail pages...\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    assessments = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(enrich_product_details, session, product): product
            for product in products
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                assessments.append(result)

    os.makedirs("data", exist_ok=True)
    with open("data/assessments.json", "w") as f:
        json.dump(assessments, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Final assessments:    {len(assessments)}")
    print(f"With description:     {sum(1 for a in assessments if a.get('description'))}")
    print(f"With duration:        {sum(1 for a in assessments if a.get('duration'))}")
    print(f"With test types:      {sum(1 for a in assessments if a.get('test_types'))}")
    print(f"Remote Yes:           {sum(1 for a in assessments if a.get('remote_support') == 'Yes')}")
    print(f"Adaptive Yes:         {sum(1 for a in assessments if a.get('adaptive_support') == 'Yes')}")
    print(f"{'='*40}")
    print("Saved to data/assessments.json")


if __name__ == "__main__":
    main()