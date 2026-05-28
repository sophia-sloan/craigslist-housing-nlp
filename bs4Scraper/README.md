# bs4webscraper.py
- scrapes the web for title, description, pricing, and location
- outputs scraped.json, and soup.txt
- soup.txt is a HTML file of the scraped soup page (currently the listings page)
- scraped.json is the formatted corpus (working on getting description working)
- ISSUES: HTML 403 error, scrapes only one city, very slow

# sitemapgrabber.py
- Scrapes the craigslist sitemap
- outputs us_cities.json, which will be used for scraping each city that craigslist recognizes for bs4webscraper.py