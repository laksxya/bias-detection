import json
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from newspaper import Article, Config
import tldextract
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- Settings ---
QUERY = os.environ.get("REPORT_TOPIC")
NUM_LINKS = 25
MIN_CONTENT_CHARS = 250
REQUEST_DELAY_SECONDS = 2.0
OUTPUT_FILE = Path(__file__).parent / "links.json"
HEADLESS = True  # Set to False to watch the browser for debugging

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
}


def unwrap_google_url(href: str) -> str | None:
    """Decode Google redirect URLs (/url? or https://www.google.com/url?) to the real target."""
    try:
        if "/url?" not in href:
            return href
        parsed = urllib.parse.urlparse(href if href.startswith(
            "http") else "https://www.google.com" + href)
        qs = urllib.parse.parse_qs(parsed.query)
        target = qs.get("q", [None])[0] or qs.get("url", [None])[0]
        return target
    except Exception:
        return None


def accept_google_consent_if_present(driver: webdriver.Chrome, wait: WebDriverWait):
    """Try to accept Google's consent/terms on first load (handles iframe too)."""
    try:
        # Try top-level buttons
        btn = None
        candidates = [
            (By.ID, "L2AGLb"),
            (By.XPATH,
             "//button//span[contains(text(),'I agree')]/ancestor::button"),
            (By.XPATH,
             "//button//span[contains(text(),'Accept')]/ancestor::button"),
        ]
        for by, sel in candidates:
            try:
                btn = wait.until(EC.element_to_be_clickable((by, sel)))
                if btn:
                    btn.click()
                    time.sleep(1.0)
                    return
            except Exception:
                pass

        # Try inside consent iframe
        iframes = driver.find_elements(
            By.CSS_SELECTOR, "iframe[src*='consent'] , iframe[name='callout']")
        for ifr in iframes:
            try:
                driver.switch_to.frame(ifr)
                for by, sel in candidates:
                    try:
                        btn = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((by, sel)))
                        if btn:
                            btn.click()
                            time.sleep(1.0)
                            driver.switch_to.default_content()
                            return
                    except Exception:
                        pass
                driver.switch_to.default_content()
            except Exception:
                driver.switch_to.default_content()
    except Exception:
        pass


def get_serp_news_links(query: str, num_links: int) -> list[str]:
    """Open google.com news tab (tbm=nws) and extract first N publisher links."""
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=en-IN")
    options.add_argument(f"--user-agent={REQUEST_HEADERS['User-Agent']}")
    # Light anti-automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 10)
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={q}&tbm=nws&hl=en&gl=IN&pws=0"
        driver.get(url)
        time.sleep(2)

        # Reduce webdriver fingerprint
        try:
            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception:
            pass

        accept_google_consent_if_present(driver, wait)

        # Detect captcha/unusual traffic
        page_source = driver.page_source.lower()
        if "unusual traffic" in page_source or "/sorry/" in driver.current_url:
            print("Blocked by Google (captcha/unusual traffic).")
            return []

        # Wait for results to appear (any of the common containers)
        try:
            wait.until(
                EC.any_of(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div.dbsr a")),
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "a.WlydOe")),
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div#search a"))
                )
            )
        except Exception:
            pass

        links: list[str] = []
        seen = set()

        def add_link(href: str):
            if not href:
                return
            # Unwrap redirectors
            target = unwrap_google_url(href)
            if not target:
                return
            netloc = urllib.parse.urlparse(target).netloc.lower()
            if not netloc or "google.com" in netloc or "news.google.com" in netloc:
                return
            if target not in seen:
                seen.add(target)
                links.append(target)

        # Collect until we have enough; try scroll and next page if needed
        attempts = 0
        while len(links) < num_links and attempts < 6:
            attempts += 1

            # 1) Direct publisher anchors in news cards
            for a in driver.find_elements(By.CSS_SELECTOR, "div.dbsr a"):
                add_link(a.get_attribute("href"))

            # 2) Sometimes news anchors have class WlydOe
            for a in driver.find_elements(By.CSS_SELECTOR, "a.WlydOe"):
                add_link(a.get_attribute("href"))

            # 3) Fallback: unwrap generic Google redirects
            for a in driver.find_elements(By.CSS_SELECTOR, "div#search a[href*='/url?']"):
                add_link(a.get_attribute("href"))

            if len(links) >= num_links:
                break

            # Scroll a bit
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(1.2)

            # Try clicking "Next" if present (news uses pagination)
            try:
                next_sel = [
                    (By.ID, "pnnext"),
                    (By.CSS_SELECTOR, "a#pnnext"),
                    (By.XPATH, "//a/span[text()='Next']/ancestor::a"),
                ]
                clicked = False
                for by, sel in next_sel:
                    try:
                        nxt = driver.find_element(by, sel)
                        nxt.click()
                        time.sleep(1.5)
                        clicked = True
                        break
                    except Exception:
                        continue
                if not clicked:
                    # If no next, break after last sweep
                    pass
            except Exception:
                pass

        return links[:num_links]
    finally:
        driver.quit()


def scrape_article(url: str):
    """Scrape article content and metadata using Newspaper3k with a simple fallback."""
    config = Config()
    config.browser_user_agent = REQUEST_HEADERS["User-Agent"]
    config.request_timeout = 12

    # Attempt 1: newspaper3k
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        article.nlp()
        if len(article.text) >= MIN_CONTENT_CHARS:
            extracted = tldextract.extract(url)
            return {
                "url": url,
                "media_source": f"{extracted.domain}.{extracted.suffix}",
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date.strftime("%Y-%m-%d") if article.publish_date else None,
                "top_image": article.top_image,
                "keywords": article.keywords,
                "summary": article.summary,
                "content": article.text
            }
        else:
            print(
                f"--> newspaper3k short content ({len(article.text)} chars): {url}")
    except Exception:
        pass

    # Attempt 2: requests + BeautifulSoup fallback
    try:
        r = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        title_el = soup.find("h1")
        title_text = title_el.get_text(strip=True) if title_el else ""
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        if len(content) >= MIN_CONTENT_CHARS:
            extracted = tldextract.extract(url)
            return {
                "url": url,
                "media_source": f"{extracted.domain}.{extracted.suffix}",
                "title": title_text or None,
                "authors": [],
                "publish_date": None,
                "top_image": None,
                "keywords": [],
                "summary": "",
                "content": content
            }
        else:
            print(f"--> Fallback short content ({len(content)} chars): {url}")
            return None
    except Exception:
        return None


def main():
    print(f"Searching Google News for: '{QUERY}'")
    links = get_serp_news_links(QUERY, NUM_LINKS)
    if not links:
        print("No links extracted from SERP.")
        return

    print(f"Collected {len(links)} links. Scraping...")
    articles = []
    for url in links:
        print(f"Scraping: {url}")
        data = scrape_article(url)
        if data:
            articles.append(data)
        time.sleep(REQUEST_DELAY_SECONDS)

    if articles:
        OUTPUT_FILE.write_text(json.dumps(
            articles, indent=4, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ Saved {len(articles)} articles to {OUTPUT_FILE}")
    else:
        print("\n❌ All scrapes failed or were too short.")


if __name__ == "__main__":
    main()
