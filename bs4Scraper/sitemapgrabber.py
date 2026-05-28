import json
import re
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup as bs

"""
- Grabs all the city names and splits by hyphens and slashes
    - Ex: seattle-tacoma becomes "seattle\ntacoma"
- Outputs them by state in us_cities.json
"""


def clean_city_names(city_text):
    city_text = re.sub(r",\s*[A-Z]{2}.*$", "", city_text.strip())
    city_names = re.split(r"\s*[/\-]\s*", city_text)
    return [city for city in city_names if city]


# Web scraper for us_section of craigslist sitemap https://craigslist.org/sitemap.html
def state_city_names_from_sitemap(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request) as response:
        html = response.read().decode("utf-8")

    # Turns raw HTML into searchable tree
    soup = bs(html, "lxml")

    us_section = soup.find(string=lambda text: text and text.strip() == "US")
    us_block = us_section.find_next("blockquote")

    states = {}
    current_state = None

    for item in us_block.children:
        if isinstance(item, str):
            state = item.strip()
            if state:
                current_state = state
                states[current_state] = []
        elif item.name == "ul" and current_state:
            for link in item.select("a"):
                states[current_state].extend(clean_city_names(link.text))

    return states


url = "https://craigslist.org/sitemap.html"
states = state_city_names_from_sitemap(url)

with open("us_cities.json", "w", encoding="utf-8") as file:
    json.dump(states, file, indent=2)

for state, cities in states.items():
    print(f"{state}: {len(cities)} cities")

print(f"\nSaved {len(states)} states to us_cities.json")
