import json
import random
import time
import requests

from bs4 import BeautifulSoup as bs


"""
Web Scraper for stuff ig and homes and nlp corpus n stuff
"""

DEBUG = True

# Rotate through multiple agents to reduce fingerprinting
USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/118.0.0.0 Safari/537.36"
    ),
]

# Session to avoid some 403 errors
session = requests.Session()

session.headers.update({
    "User-Agent": random.choice(USER_AGENTS),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})

# Retry settings for resilience against transient errors
MAX_RETRIES = 3
RETRY_CODES = {403, 429, 500, 502, 503}


def get_html(url: str, referer: str = ""):
    """
    Retrieves raw HTML from a URL, retrying on common error codes.
    """

    last_response = None

    for attempt in range(1, MAX_RETRIES + 1):

        # Rotate user agent and set referer each attempt
        session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            **({"Referer": referer} if referer else {}),
        })

        try:
            response = session.get(url, timeout=15)
            last_response = response

            if response.status_code not in RETRY_CODES:
                response.raise_for_status()
                return response.text

            if DEBUG:
                print(f"  [{attempt}/{MAX_RETRIES}] HTTP {response.status_code} — retrying {url}")

        except requests.exceptions.RequestException as e:

            if attempt == MAX_RETRIES:
                raise

            if DEBUG:
                print(f"  [{attempt}/{MAX_RETRIES}] Request error — retrying: {e}")

        # Exponential backoff: 2s, 4s, 8s
        time.sleep(2 ** attempt)

    # All retries exhausted on a retryable status code
    if last_response is not None:
        last_response.raise_for_status()

    raise requests.exceptions.RetryError(
        f"Failed to fetch {url} after {MAX_RETRIES} attempts"
    )

#Saves formatted HTML to soup.txt for inspection.
def save_html(soup: bs):

    with open("soup.txt", "w", encoding="utf-8") as file:
        file.write(soup.prettify())

# Retrieves the description/body text from a Craigslist listing.
def retrieve_description(url: str, referer: str = ""):

    try:
        html = get_html(url, referer=referer)

    except Exception as e:

        if DEBUG:
            print(f"ERROR: could not get description from {url}")
            print(f"   {e}")

        return ""

    soup = bs(html, "lxml")

    description = soup.select_one("#postingbody")

    if not description:
        return ""

    text = description.get_text(" ", strip=True)

    return text.replace("QR Code Link to This Post", "").strip()


def retrieve_text(city: str):

    BASE = ".craigslist.org/search"
    TYPE = "apa"

    url = f"https://{city}{BASE}/{TYPE}"

    try:
        html = get_html(url)

    except Exception as e:

        if DEBUG:
            print(f"ERROR: {url}")
            print(f"   {url} problem with address")
            print(f"   {e}")

        return

    # Turns raw HTML into searchable tree
    soup = bs(html, "lxml")

    if DEBUG:
        save_html(soup)

    listings = []

    for row in soup.select(".cl-static-search-result"):

        title = row.select_one(".title")
        location = row.select_one(".location")
        price = row.select_one(".price")
        link = row.select_one("a")

        link_url = (
            link["href"]
            if link and link.has_attr("href")
            else ""
        )

        description = (
            # Pass search URL as referer to look like natural navigation
            retrieve_description(link_url, referer=url)
            if link_url
            else ""
        )

        # Random jitter looks more human to bot detection
        time.sleep(random.uniform(1.0, 2.5))

        listings.append({
            "title": (
                title.text.strip()
                if title
                else ""
            ),

            "location": (
                location.text.strip()
                if location
                else ""
            ),

            "price": (
                price.text.strip()
                if price
                else ""
            ),

            "link": link_url,

            "description": description,
        })

        print(title)

    # Saves scraped data into JSON
    with open("scraped.json", "w", encoding="utf-8") as file:
        json.dump(listings, file, indent=2)

    print(url)


if __name__ == "__main__":
    retrieve_text("seattle")
