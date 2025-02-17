import time
from pathlib import Path
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

from bsmp.scrapers.core.browser import BrowserManager
from bsmp.scrapers.core.database import DatabaseManager


class MatchDataScraper:
    """Scrapes detailed match data and stores it in a structured format."""

    BASE_URL = "https://www.flashscore.com/match/"
    NETWORK_DELAY = 2
    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        config_path: Path = Path("config/flashscore_urls.yaml"),
        db_path: str = "database/database.db",
    ):
        """
        Initializes the MatchDataScraper with configuration and database paths.

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
        """
        Ensure required files and directories exist.

        Raises
        ------
        FileNotFoundError
            If the configuration file is missing.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file missing: {self.config_path}")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _calculate_season(self, month: int, year: int) -> str:
        """
        Determine season string based on match date.

        Parameters
        ----------
        month : int
            The month of the match.
        year : int
            The year of the match.

        Returns
        -------
        str
            The season string in the format "YYYY/YYYY".
        """
        return f"{year}/{year + 1}" if month >= 8 else f"{year - 1}/{year}"

    def _parse_datetime(self, dt_str: str) -> Tuple[str, int, int]:
        """
        Extract datetime components from raw string.

        Parameters
        ----------
        dt_str : str
            The datetime string to parse.

        Returns
        -------
        Tuple[str, int, int]
            The original datetime string, month, and year.
        """
        date_parts = dt_str.split(".")
        _, month = map(int, date_parts[:2])
        year_time = date_parts[2].split()
        year = int(year_time[0])
        return dt_str, month, year

    def _parse_period_scores(self, elements: list) -> List[int]:
        """
        Safely convert BeautifulSoup elements to integer scores.

        Parameters
        ----------
        elements : list
            List of BeautifulSoup elements containing score data.

        Returns
        -------
        List[int]
            List of integer scores.
        """
        return [
            int(el.text) if el.text.strip() else 0 for el in elements if el and el.text
        ]

    def _extract_scores(
        self, soup: BeautifulSoup
    ) -> Tuple[int, int, int, int, int, int, str]:
        """
        Extract and validate match scores with comprehensive error handling.

        Parameters
        ----------
        soup : BeautifulSoup
            The BeautifulSoup object containing the page content.

        Returns
        -------
        Tuple[int, int, int, int, int, int, str]
            The extracted scores and match result.
        """
        try:
            # Extract all period score elements
            periods = soup.find_all(class_="smh__part")
            home_scores = self._parse_period_scores(periods[:3])  # Even indices
            away_scores = self._parse_period_scores(periods[3:])  # Odd indices

            home_scores.extend([0] * (3 - len(home_scores))) if len(
                home_scores
            ) < 3 else home_scores
            away_scores.extend([0] * (3 - len(away_scores))) if len(
                away_scores
            ) < 3 else away_scores

            # Calculate totals
            home_total, h1, h2 = home_scores
            away_total, a1, a2 = away_scores

            # Determine match outcome
            result = (
                "H"
                if home_total > away_total
                else "A"
                if home_total < away_total
                else "D"
            )

            return (home_total, h1, h2, away_total, a1, a2, result)

        except (IndexError, ValueError, AttributeError) as e:
            print(f"Score parsing error: {str(e)}")
            return (0, 0, 0, 0, 0, 0, "U")  # 'U' for unknown

    def _extract_match_details(self, soup: BeautifulSoup) -> Optional[Tuple]:
        """
        Parse complete match details from page content.

        Parameters
        ----------
        soup : BeautifulSoup
            The BeautifulSoup object containing the page content.

        Returns
        -------
        Optional[Tuple]
            The extracted match details, or None if extraction fails.
        """
        try:
            header = soup.find("span", class_="tournamentHeader__country")
            dt_str, month, year = self._parse_datetime(
                header.find_next("div", class_="duelParticipant__startTime").text
            )

            return (
                header.text.split(":")[0].strip(),  # Country
                header.find("a").text.split(" -")[0].strip(),  # League
                self._calculate_season(month, year),
                header.text.split(" - ")[-1].strip(),  # Match info
                dt_str,
                *self._extract_teams(soup),
                *self._extract_scores(soup),
            )
        except Exception as e:
            print(f"Error parsing match details: {str(e)}")
            return None

    def _extract_teams(self, soup: BeautifulSoup) -> Tuple[str, str]:
        """
        Extract home and away team names.

        Parameters
        ----------
        soup : BeautifulSoup
            The BeautifulSoup object containing the page content.

        Returns
        -------
        Tuple[str, str]
            The home and away team names.
        """
        return (
            soup.find("div", class_="duelParticipant__home")
            .find("a", class_="participant__participantName participant__overflow")
            .text.strip(),
            soup.find("div", class_="duelParticipant__away")
            .find("a", class_="participant__participantName participant__overflow")
            .text.strip(),
        )

    def scrape(
        self, batch_size: int = DEFAULT_BATCH_SIZE, headless: bool = True
    ) -> None:
        """
        Main scraping workflow with progress tracking.

        Parameters
        ----------
        batch_size : int, optional
            Number of records to insert in each batch. Defaults to DEFAULT_BATCH_SIZE.
        headless : bool, optional
            Whether to run the browser in headless mode. Defaults to True.
        """
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("""
                SELECT m.match_id 
                FROM handball_match_id m
                LEFT JOIN handball_match_data d ON m.match_id = d.flashscore_id
                WHERE d.flashscore_id IS NULL
            """)
            match_ids = [row[0] for row in cursor.fetchall()]

        if not match_ids:
            print("No matches requiring data collection")
            return

        browser = BrowserManager(headless=headless)
        data_buffer = []

        with tqdm(total=len(match_ids), desc="Scraping matches") as pbar:
            for match_id in match_ids:
                url = f"{self.BASE_URL}{match_id}/#/match-summary"
                with browser.get_driver(url) as driver:
                    time.sleep(self.NETWORK_DELAY)
                    details = self._extract_match_details(
                        BeautifulSoup(driver.page_source, "html.parser")
                    )

                if details:
                    data_buffer.append((match_id, *details))
                    if len(data_buffer) >= batch_size:
                        self._store_batch(data_buffer)
                        data_buffer.clear()

                pbar.update(1)

            if data_buffer:
                self._store_batch(data_buffer)

    def _store_batch(self, data: List[Tuple]) -> None:
        """
        Batch insert match records into database.

        Parameters
        ----------
        data : List[Tuple]
            List of match data tuples to insert.
        """
        query = """
        INSERT INTO handball_match_data (
            flashscore_id, country, league, season, match_info, datetime,
            home_team, away_team, home_goals_full, home_goals_h1, home_goals_h2,
            away_goals_full, away_goals_h1, away_goals_h2, result
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with DatabaseManager(self.db_path) as cursor:
            cursor.executemany(query, data)

        print(f"Inserted {len(data)} records")


if __name__ == "__main__":
    pass
