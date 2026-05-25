"""
downloader.py — Download PDFs from patseer URLs.

Usage:
    from src.downloader import download_pdfs
    download_pdfs(subset_df, cfg)
"""

import time
import requests
from pathlib import Path
from tqdm import tqdm


def download_pdfs(df, cfg: dict) -> dict:
    """
    Download PDFs for each patent in df.
    Skips patents already downloaded.
    Returns a dict: {record_number: pdf_path}
    """
    pdf_dir = Path(cfg["paths"]["pdfs"])
    pdf_dir.mkdir(parents=True, exist_ok=True)

    timeout = cfg["downloader"]["timeout_seconds"]
    retries = cfg["downloader"]["retry_attempts"]
    delay   = cfg["downloader"]["delay_between_requests"]

    results = {}
    failed  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
        record_id = str(row["Record Number"])
        url = row["pdf_url"]
        out_path = pdf_dir / f"{record_id}.pdf"

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size > 1000:
            results[record_id] = out_path
            continue

        success = False
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                results[record_id] = out_path
                success = True
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    print(f"\n  FAILED: {record_id} — {e}")
                    failed.append(record_id)

        if success:
            time.sleep(delay)

    print(f"\nDownload complete: {len(results)} OK, {len(failed)} failed.")
    if failed:
        print(f"  Failed patents: {failed}")
    return results
