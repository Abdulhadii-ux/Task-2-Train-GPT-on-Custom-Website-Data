import requests
from bs4 import BeautifulSoup
import time

# All pages on the site
URLS = [
    "https://stemp21.com/",
    "https://stemp21.com/about/",
    "https://stemp21.com/stemp-lab/",
    "https://stemp21.com/affiliated-schools-in-pakistan/",
    "https://stemp21.com/punjab/",
    "https://stemp21.com/kpk/",
    "https://stemp21.com/sindh/",
    "https://stemp21.com/balochistan/",
    "https://stemp21.com/islamabad/",
    "https://stemp21.com/azad-kashmir/",
    "https://stemp21.com/gilgit-baltistan/",
    "https://stemp21.com/certified-trainer/",
    "https://stemp21.com/international-collaborations/",
    "https://stemp21.com/national-collaborations/",
    "https://stemp21.com/faqs/",
]

all_text = []

for url in URLS:
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove nav, footer, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Clean up blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        page_text = "\n".join(lines)
        all_text.append(page_text)
        print(f"✓ Scraped: {url}")
        time.sleep(1)  # be polite, don't hammer the server
    except Exception as e:
        print(f"✗ Failed: {url} — {e}")

# Save everything to one text file
with open("stemp21_data.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print("\nDone! Saved to stemp21_data.txt")