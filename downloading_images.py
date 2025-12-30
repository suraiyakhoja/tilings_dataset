import os
import csv
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://tilingsearch.mit.edu"
AREA_INDEX = BASE + "/area1.htm"

ROOT_OUT = "tilings_dataset"
os.makedirs(ROOT_OUT, exist_ok=True)

# Folders for each fold family
FOLD_FOLDERS = ["3-fold", "4-fold", "6-fold", "other"]
for fold in FOLD_FOLDERS:
    os.makedirs(os.path.join(ROOT_OUT, fold), exist_ok=True)

CSV_PATH = os.path.join(ROOT_OUT, "tilings_metadata.csv")

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
        if text == "list":
            href = a.get("href", "")
            if href:
                list_url = urljoin(AREA_INDEX, href)
                list_pages.append(list_url)

    return list_pages


def get_pattern_codes_from_list(list_url):
    """
    From a list page like area/SPAL.htm, extract all pattern codes like 'data10/P007'.
    """
    resp = requests.get(list_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pattern_codes = []
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        if text.startswith("data") and "/" in text:
            pattern_codes.append(text)

    return pattern_codes


def extract_wallpaper_group(symmetry_line):
    """
    From 'The symmetry group of the tiling is 3*3 (p31m).' return 'p31m'.
    """
    m = re.search(r"\(([^()]+)\)", symmetry_line)
    if m:
        return m.group(1).strip()
    return ""


def fold_from_group(group):
    """
    Map wallpaper group string -> 3-fold / 4-fold / 6-fold / other.
    """
    g = group.lower()
    if "6" in g:
        return "6-fold"
    if "4" in g:
        return "4-fold"
    if "3" in g:
        return "3-fold"
    if "2" in g or "mm" in g or "mg" in g or "gg" in g:
        return "2-fold"
    return "other"


def get_symmetry_for_pattern(pattern_code):
    """
    Given 'data10/P007', fetch https://tilingsearch.mit.edu/HTML/data10/P007.html
    and extract the symmetry line + wallpaper group + fold.
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

    if symmetry_line:
        group = extract_wallpaper_group(symmetry_line)
        fold = fold_from_group(group)
    else:
        group = ""
        fold = "other"

    return html_url, symmetry_line, group, fold


def download_pdf(pattern_code, fold):
    """
    Download PDF for a pattern into the appropriate fold folder.
    pattern_code: 'data10/P007'
    URL: https://tilingsearch.mit.edu/contents/data10/P007.pdf
    """
    pdf_url = f"{BASE}/contents/{pattern_code}.pdf"
    safe_name = pattern_code.replace("/", "_") + ".pdf"
    out_path = os.path.join(ROOT_OUT, fold, safe_name)

    try:
        r = requests.get(pdf_url, timeout=15)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"  Downloaded PDF to {out_path}")
    except Exception as e:
        print(f"  Failed to download {pdf_url}: {e}")


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
            "pdf_path",
        ])

        for list_url in list_pages:
            print(f"\nProcessing list page: {list_url}")
            pattern_codes = get_pattern_codes_from_list(list_url)
            print(f"  Found {len(pattern_codes)} patterns.")

            for code in pattern_codes:
                try:
                    html_url, sym_line, group, fold = get_symmetry_for_pattern(code)
                except Exception as e:
                    print(f"  Error extracting symmetry for {code}: {e}")
                    html_url, sym_line, group, fold = "", "", "", "other"

                # Download PDF into corresponding fold folder
                download_pdf(code, fold)

                writer.writerow([
                    list_url,
                    code,
                    html_url,
                    sym_line,
                    group,
                    fold,
                    os.path.join(ROOT_OUT, fold, code.replace("/", "_") + ".pdf"),
                ])

    print(f"\nDone! Metadata + PDFs organized under: {ROOT_OUT}")


if __name__ == "__main__":
    main()
