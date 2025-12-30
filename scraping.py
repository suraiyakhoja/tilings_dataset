import os
import csv
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://tilingsearch.mit.edu"
AREA_INDEX = BASE + "/area1.htm"

OUT_DIR = "tilings_dataset"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "tilings_metadata.csv")

def get_list_pages():
    """
    From area1.htm, find all 'list' links like area/SPAL.htm, area/MORL.htm, etc.
    """
    resp = requests.get(AREA_INDEX)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    list_pages = []
    for a in soup.find_all("a"):
        text = a.get_text(strip=True).lower()
        if text == "list":  # the 'see list' link for each region
            href = a.get("href", "")
            if href:
                list_url = urljoin(AREA_INDEX, href)
                list_pages.append(list_url)

    return list_pages

def get_pattern_codes_from_list(list_url):
    """
    From a list page like area/SPAL.htm, extract all pattern codes like 'data1/E2'.
    """
    resp = requests.get(list_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pattern_codes = []
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        # On the table pages, the pattern links look like 'data1/E2'
        if text.startswith("data") and "/" in text:
            pattern_codes.append(text)

    return pattern_codes

def extract_wallpaper_group(symmetry_line):
    """
    From a line like 'The symmetry group of the tiling is 4*2 (p4g).'
    return 'p4g'.
    """
    m = re.search(r"\(([^()]+)\)", symmetry_line)
    if m:
        return m.group(1).strip()
    return ""

def fold_from_group(group):
    """
    Very rough mapping: wallpaper group string -> 3-, 4-, 6-fold, or 'other'.
    """
    g = group.lower()
    if "6" in g:
        return "6-fold"
    if "4" in g:
        return "4-fold"
    if "3" in g:
        return "3-fold"
    return "other"

def get_symmetry_for_pattern(pattern_code):
    """
    Given 'data1/E2', fetch https://tilingsearch.mit.edu/HTML/data1/E2.html
    and extract the symmetry line.
    """
    html_url = f"{BASE}/HTML/{pattern_code}.html"
    resp = requests.get(html_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    symmetry_line = ""
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if "symmetry group of the tiling" in text.lower():
            symmetry_line = text
            break

    group = extract_wallpaper_group(symmetry_line) if symmetry_line else ""
    fold = fold_from_group(group) if group else ""
    return html_url, symmetry_line, group, fold

def main():
    list_pages = get_list_pages()
    print(f"Found {len(list_pages)} region list pages.")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "region_list_page",
            "pattern_code",
            "pattern_html_url",
            "symmetry_line",
            "wallpaper_group",
            "fold_family",
        ])

        for list_url in list_pages:
            print(f"\nProcessing list page: {list_url}")
            pattern_codes = get_pattern_codes_from_list(list_url)
            print(f"  Found {len(pattern_codes)} patterns.")

            for code in pattern_codes:
                try:
                    html_url, sym_line, group, fold = get_symmetry_for_pattern(code)
                except Exception as e:
                    print(f"    Error on {code}: {e}")
                    html_url, sym_line, group, fold = "", "", "", ""

                writer.writerow([
                    list_url,
                    code,
                    html_url,
                    sym_line,
                    group,
                    fold,
                ])

    print(f"\nDone! Metadata saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
