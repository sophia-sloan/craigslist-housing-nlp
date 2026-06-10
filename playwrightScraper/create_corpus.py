from scrape_playwright_new import scrape_city
import asyncio
import json
import time

BASE = ".craigslist.org/search"
TYPE = "apa"

# Fill in city subdomains as keys, human-readable names as values
CITIES = [
    "bellingham",
    "kennewick",
    "pasco",
    "richland",
    "moses lake",
    "olympic peninsula",
    "pullman",
    "moscow",
    "seattle",
    "tacoma",
    "skagit",
    "island",
    "SJI",
    "spokane",
    "coeur d'alene",
    "wenatchee",
    "yakima"
]

MAX_PAGES_PER_CITY = 1


async def main():
    start = time.time()
    all_listings = []

    for city in CITIES:
        url = f"https://{city}{BASE}/{TYPE}"
        print(f"\nScraping {city} ({url}) ...")
        try:
            listings = await scrape_city(url, max_pages=MAX_PAGES_PER_CITY)
            for listing in listings:
                listing["city"] = city
            all_listings.extend(listings)
            print(f"  => {len(listings)} listings collected from {city}")
        except Exception as e:
            print(f"  => Skipping {city}: {e}")

    with open("corpus.json", "w") as f:
        json.dump(all_listings, f, indent=2)

    print(f"\nDone in {time.time() - start:.2f}s. Saved {len(all_listings)} total listings to corpus.json")


if __name__ == "__main__":
    asyncio.run(main())
