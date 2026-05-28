from playwright.async_api import async_playwright
import asyncio
import json
import time

# requirements:
#   playwright






BASE_URL = "https://bellingham.craigslist.org/search/apa"
MAX_PAGES = 1
DELAY_SEC = 0.5
CONCURRENT_TABS = 10  # number of simultaneous tabs that are opened to retrieve descriptions

SELECTORS = {
    "listing_container": "div.gallery-card",
    "title":             "span.label",
    "price":             "span.priceinfo",
    "neighborhood":      "span.result-location",
    "url":               "a.main",
}


async def get_text(el):
    try:
        return (await el.inner_text()).strip()
    except:
        return None


async def get_attr(el, attr):
    try:
        return await el.get_attribute(attr)
    except:
        return None


async def get_description(browser, listing):
    page = await browser.new_page()
    try:
        await page.route("**/*", lambda route: (
            route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"]
            else route.continue_()
        ))
        await page.goto(listing["url"], wait_until="domcontentloaded")
        # await page.wait_for_timeout(1000)
        el = await page.query_selector("#postingbody")
        text = await get_text(el) if el else None
        listing["description"] = text.replace("QR Code Link to This Post", "").strip() if text else None
    except:
        listing["description"] = None
    finally:
        if page:
            try:
                await page.close()
            except:
                pass
    return listing


async def main():
    start = time.time()
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for page_num in range(MAX_PAGES):
            url = f"{BASE_URL}?start={page_num * 120}"
            await page.goto(url)
            await page.wait_for_timeout(1500)

            containers = await page.query_selector_all(SELECTORS["listing_container"])
            print(f"Found {len(containers)} containers")


            raw = []
            for item in containers:
                title_el = await item.query_selector(SELECTORS["title"])
                if title_el:
                    print(await title_el.inner_text())
                raw.append({
                    "title":        await get_text(await item.query_selector(SELECTORS["title"])),
                    "price":        await get_text(await item.query_selector(SELECTORS["price"])),
                    "neighborhood": await get_text(await item.query_selector(SELECTORS["neighborhood"])),
                    "url":          await get_attr(await item.query_selector(SELECTORS["url"]), "href"),
                })


            description_counter = 0
            for i in range(0, len(raw), CONCURRENT_TABS):
                batch = [l for l in raw[i:i + CONCURRENT_TABS] if l["url"]]
                await asyncio.gather(*[get_description(browser, listing) for listing in batch])
                description_counter += len(batch)
                print(f"Fetched {description_counter}/{len(raw)} descriptions", end='\r')
                await asyncio.sleep(DELAY_SEC)

            results.extend(raw)

        await browser.close()

    with open("listings.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done in {time.time() - start:.2f}s. Saved {len(results)} listings to listings.json")


asyncio.run(main())