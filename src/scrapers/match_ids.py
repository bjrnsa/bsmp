import time
from pathlib import Path
from typing import List, Set

import yaml
from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from src.scrapers.core.browser import BrowserManager
from src.scrapers.core.database import DatabaseManager


class MatchIDScraper:
    """Scrapes match IDs from FlashScore and stores them in a database."""

    def __init__(
        self,
        config_path: Path = Path("config/flashscore_urls.yaml"),
        db_path: str = "data/processed/database.db",
        wait_time: int = 3,
    ):
        self.config_path = config_path
        self.db_path = db_path
        self.wait_time = wait_time
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Verify required files and directories exist."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> List[str]:
        """Load league URLs from YAML configuration."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        urls = []
        for entry in config:
            base_url = entry["league"]
            for season in entry["seasons"]:
                season_suffix = f"-{season - 1}-{season}/results/"
                urls.append(f"{base_url}{season_suffix}")
        return urls

    def _scroll_to_bottom(self, driver) -> None:
        """Scroll to page bottom to load more content."""
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def _load_all_matches(self, driver) -> None:
        """Continuously load matches until no more are available."""
        try:
            while True:
                self._scroll_to_bottom(driver)
                button = WebDriverWait(driver, self.wait_time).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "event__more--static"))
                )
                button.click()
                time.sleep(self.wait_time)
        except (TimeoutException, ElementClickInterceptedException):
            pass

    def _extract_ids(self, browser: BrowserManager, url: str) -> Set[str]:
        """Extract unique match IDs from a league season page."""
        with browser.get_driver(url) as driver:
            self._load_all_matches(driver)
            soup = BeautifulSoup(driver.page_source, "html.parser")

        return {
            element.get("id").split("_")[-1]
            for element in soup.find_all(
                "div", class_=lambda x: x and "event__match--withRowLink" in x
            )
        }

    def _get_existing_ids(self) -> Set[str]:
        """Retrieve already stored match IDs from database."""
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("SELECT match_id FROM handball_match_id")
            return {row[0] for row in cursor.fetchall()}

    def scrape(self, headless: bool = True) -> int:
        """Main scraping workflow."""
        urls = self._load_config()
        existing_ids = self._get_existing_ids()
        browser = BrowserManager(headless=headless)
        new_ids = set()

        for url in tqdm(urls, desc="Processing leagues"):
            try:
                league_ids = self._extract_ids(browser, url)
                new_ids.update(league_ids - existing_ids)
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")

        with DatabaseManager(self.db_path) as cursor:
            cursor.executemany(
                "INSERT INTO handball_match_id (match_id, source) VALUES (?, ?)",
                [(id_, "flashscore") for id_ in new_ids],
            )

        return len(new_ids)


if __name__ == "__main__":
    scraper = MatchIDScraper()
    count = scraper.scrape(headless=False)
    print(f"Scraping complete. New entries: {count}")
