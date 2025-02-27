"""This module contains the MatchIDScraper class for scraping match IDs from FlashScore."""

import time
from pathlib import Path
from typing import List, Set

import yaml
from bs4 import BeautifulSoup, Tag
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from bsmp.scrapers.core.browser import BrowserManager
from bsmp.scrapers.core.database import DatabaseManager


class MatchIDScraper:
    """Scrapes match IDs from FlashScore and stores them in a database."""

    NETWORK_DELAY = 5
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        config_path: Path = Path("config/flashscore_urls.yaml"),
        db_path: str = "database/database.db",
    ):
        """Initializes the MatchIDScraper with configuration and database paths.

        Parameters
        ----------
        config_path : Path, optional
            Path to the configuration file containing URLs. Defaults to "config/flashscore_urls.yaml".
        db_path : str, optional
            Path to the SQLite database file. Defaults to "database/database.db".
        """
        self.config_path = config_path
        self.db_path = db_path
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Verify required files and directories exist.

        Raises:
        ------
        FileNotFoundError
            If the configuration file is missing.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> List[str]:
        """Load league URLs from YAML configuration.

        Returns:
        -------
        List[str]
            List of league URLs to scrape.
        """
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
        """Scroll to page bottom to load more content.

        Parameters
        ----------
        driver : selenium.webdriver.Chrome
            The Chrome WebDriver instance.
        """
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def _load_all_matches(self, driver) -> None:
        """Continuously load matches until no more are available.

        Parameters
        ----------
        driver : selenium.webdriver.Chrome
            The Chrome WebDriver instance.
        """
        try:
            while True:
                self._scroll_to_bottom(driver)
                button = WebDriverWait(driver, self.NETWORK_DELAY).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "event__more--static"))
                )
                time.sleep(self.NETWORK_DELAY)
                button.click()

        except (TimeoutException, ElementClickInterceptedException):
            pass

    def _extract_ids(self, browser: BrowserManager, url: str) -> Set[str]:
        """Extract unique match IDs from a league season page.

        Parameters
        ----------
        browser : BrowserManager
            The BrowserManager instance.
        url : str
            The URL of the league season page.

        Returns:
        -------
        Set[str]
            Set of unique match IDs.
        """
        with browser.get_driver(url) as driver:
            self._load_all_matches(driver)
            soup = BeautifulSoup(driver.page_source, "html.parser")

        matches = soup.find_all("div", class_="event__match--withRowLink")
        return {
            str(element.attrs["id"]).split("_")[-1]
            for element in matches
            if isinstance(element, Tag)
        }

    def _get_existing_ids(self) -> Set[str]:
        """Retrieve already stored match IDs from database.

        Returns:
        -------
        Set[str]
            Set of existing match IDs.
        """
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("SELECT match_id FROM handball_match_id")
            return {row[0] for row in cursor.fetchall()}

    def scrape(self, headless: bool = True) -> int:
        """Main scraping workflow.

        Parameters
        ----------
        headless : bool, optional
            Whether to run the browser in headless mode. Defaults to True.

        Returns:
        -------
        int
            Number of new match IDs scraped and stored.
        """
        urls = self._load_config()
        existing_ids = self._get_existing_ids()
        browser = BrowserManager(headless=headless)
        new_ids = set()

        for url in tqdm(urls, desc="Processing leagues"):
            try:
                league_ids = self._extract_ids(browser, url)
                new_ids.update(league_ids - existing_ids)
            except Exception:
                pass

        with DatabaseManager(self.db_path) as cursor:
            cursor.executemany(
                "INSERT INTO handball_match_id (match_id, source) VALUES (?, ?)",
                [(id_, "flashscore") for id_ in new_ids],
            )

        return len(new_ids)


if __name__ == "__main__":
    match_id_scraper = MatchIDScraper()
    match_id_scraper.scrape()
