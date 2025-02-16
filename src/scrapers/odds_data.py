import time
from pathlib import Path
from typing import Iterator, List, Tuple

from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from src.scrapers.core.browser import BrowserManager
from src.scrapers.core.database import DatabaseManager


class OddsDataScraper:
    """Scrapes and stores betting odds data from FlashScore matches."""

    BASE_URL = "https://www.flashscore.com/match"
    NETWORK_DELAY = 3
    DEFAULT_BATCH_SIZE = 100

    def __init__(self, db_path: str = "database/database.db", max_retries: int = 3):
        self.db_path = db_path
        self.max_retries = max_retries
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Ensure required filesystem resources exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _fetch_pending_matches(self) -> Iterator[str]:
        """Generator for matches needing odds data collection."""
        query = """
        SELECT m.match_id 
        FROM handball_match_id m
        LEFT JOIN handball_odds_data o ON m.match_id = o.flashscore_id
        WHERE o.flashscore_id IS NULL
        """

        # Fetch matches without odds data
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute(query)
            for row in cursor.fetchall():
                yield row[0]

    def _parse_odds_row(self, odds_row: Tag) -> List[Tuple[str, float, float, float]]:
        """Extract bookmaker odds from odds row element."""
        results = []
        for bookmaker in odds_row.find_all(class_="oddsRowContent"):
            try:
                name = bookmaker.find("img")["alt"]
                odds_values = [
                    self._parse_odd(el.text)
                    for el in bookmaker.find_all(class_="oddsValueInner")
                ]
                results.append((name, *odds_values))
            except (AttributeError, ValueError):
                continue
        return results

    def _parse_odd(self, value: str) -> float:
        """Safely convert odd value to float."""
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return float("nan")

    def _process_match_odds(
        self, match_id: str, browser: BrowserManager
    ) -> List[Tuple]:
        """Scrape and parse odds data for a single match."""
        url = f"{self.BASE_URL}/{match_id}/#/match-summary"
        records = []

        try:
            with browser.get_driver(url) as driver:
                time.sleep(self.NETWORK_DELAY)
                soup = BeautifulSoup(driver.page_source, "html.parser")

                if odds_row := soup.find(class_="oddsRow"):
                    for bookmaker in self._parse_odds_row(odds_row):
                        records.append((match_id, *bookmaker))

        except Exception as e:
            print(f"Failed to process {match_id}: {str(e)}")

        return records

    def scrape(
        self, batch_size: int = DEFAULT_BATCH_SIZE, headless: bool = True
    ) -> None:
        """Orchestrate scraping workflow with progress tracking."""
        browser = BrowserManager(headless=headless)
        data_buffer = []

        with tqdm(desc="Processing odds", unit="match") as progress:
            for match_id in self._fetch_pending_matches():
                if records := self._process_match_odds(match_id, browser):
                    data_buffer.extend(records)
                    progress.update(1)

                if len(data_buffer) >= batch_size:
                    self._store_batch(data_buffer)
                    data_buffer.clear()

            if data_buffer:
                self._store_batch(data_buffer)

    def _store_batch(self, records: List[Tuple]) -> None:
        """Batch insert records with transaction management."""
        query = """
        INSERT INTO handball_odds_data (
            flashscore_id, bookmaker, home_win_odds, draw_odds, away_win_odds
        ) VALUES (?, ?, ?, ?, ?)
        """
        with DatabaseManager(self.db_path) as cursor:
            cursor.executemany(query, records)


# Usage example
if __name__ == "__main__":
    odds_scraper = OddsDataScraper()
    odds_scraper.scrape(batch_size=10)
