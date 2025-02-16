import sqlite3
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class MatchDataLoader:
    """Flexible data loader for match data with caching and dynamic filtering."""

    def __init__(
        self,
        sport: str = "handball",
        date_format: str = "%d.%m.%Y %H:%M",
        db_path: Union[str, Path] = "database/database.db",
        connection: Optional[sqlite3.Connection] = None,
    ):
        """
        Initialize data loader with configuration options

        :param sport: Sport type (e.g., 'handball', 'football')
        :param db_path: Path to SQLite database file
        :param connection: Optional existing database connection
        """
        self.sport = sport
        self.date_format = date_format
        self.db_path = Path(db_path)
        self.conn = connection or self._create_connection()
        self._validate_tables()

    def _create_connection(self) -> sqlite3.Connection:
        """Establish database connection with error handling"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def _validate_tables(self) -> None:
        """Verify required tables exist in database"""
        required_tables = {f"{self.sport}_match_data", f"{self.sport}_current_clubs"}
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        if not required_tables.issubset(existing_tables):
            missing = required_tables - existing_tables
            raise ValueError(f"Missing required tables: {', '.join(missing)}")

    def _get_current_teams(
        self, season: str, league: str, filters: Optional[dict] = None
    ) -> pd.Series:
        """
        Retrieve current teams with optional filters

        :param season: Season identifier
        :param league: League name
        :param filters: Additional filter criteria
        :return: Series of team names
        """
        base_query = f"""
            SELECT team_name 
            FROM {self.sport}_current_clubs
            WHERE season = ? AND league = ?
        """
        params = [season, league]

        if filters:
            filter_clauses = [f"{k} = ?" for k in filters]
            base_query += " AND " + " AND ".join(filter_clauses)
            params.extend(filters.values())

        teams = pd.read_sql(base_query, self.conn, params=params)["team_name"]
        if teams.empty:
            raise ValueError(f"No teams found for {season} {league}")
        return teams

    def load_matches(
        self,
        season: str,
        league: str,
        date_range: Optional[tuple] = None,
        team_filters: Optional[dict] = None,
        result_mapping: dict = {"H": 1, "A": -1, "D": 0},
    ) -> pd.DataFrame:
        """
        Load match data with flexible filtering options

        :param season: Season identifier
        :param league: League name
        :param date_range: Tuple of (start_date, end_date) as strings
        :param team_filters: Additional filters for teams table
        :param result_mapping: Custom result value mapping
        :return: Processed DataFrame of match data
        """
        try:
            teams = self._get_current_teams(season, league, team_filters)

            # Add team index mapping
            self.team_list = sorted(teams.unique())
            self.team_to_idx = {
                team: idx + 1 for idx, team in enumerate(self.team_list)
            }

            query = f"""
                SELECT 
                    flashscore_id, season, match_info, datetime, league,
                    home_team, away_team, 
                    home_goals_full AS hg, 
                    home_goals_h1 AS h1, home_goals_h2 AS h2,
                    away_goals_full AS ag,
                    away_goals_h1 AS a1, away_goals_h2 AS a2,
                    result
                FROM {self.sport}_match_data
                WHERE home_team IN {tuple(teams)}
                AND away_team IN {tuple(teams)}
            """

            df = pd.read_sql(
                query, self.conn, parse_dates={"datetime": {"format": self.date_format}}
            )

            if date_range:
                df = df.query("@date_range[0] <= datetime <= @date_range[1]")

            df = self._process_data(df, result_mapping)
            return df

        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")
            return pd.DataFrame()

    def get_team_mapping(self) -> pd.DataFrame:
        """Get team ID mapping for model reference"""
        return pd.DataFrame(
            {"team": self.team_list, "index": range(len(self.team_list))}
        )

    def _process_data(self, df: pd.DataFrame, result_mapping: dict) -> pd.DataFrame:
        """Post-process dataframe with common transformations"""
        if df.empty:
            return df

        return (
            df.assign(
                home_index=lambda x: x["home_team"].map(self.team_to_idx),
                away_index=lambda x: x["away_team"].map(self.team_to_idx),
                result=lambda x: x["result"].map(result_mapping),
                season_year=lambda x: x["season"].str.split("/").str[1].astype(int),
            )
            .sort_values("datetime")
            .set_index("flashscore_id")
            .pipe(self._add_additional_features)
        )

    def _add_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features to dataframe"""
        return df.assign(
            goal_difference=lambda x: x["hg"] - x["ag"],
            goal_difference_h1=lambda x: x["h1"] - x["a1"],
            goal_difference_h2=lambda x: x["h2"] - x["a2"],
            total_goals=lambda x: x["hg"] + x["ag"],
        )

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


## Usage Example
if __name__ == "__main__":
    loader = MatchDataLoader(sport="handball")
    df = loader.load_matches(
        season="2024/2025",
        league="Herre Handbold Ligaen",
        date_range=("2024-08-01", "2025-02-01"),
        team_filters={"country": "Denmark"},
    )
    # coords, home_idx, away_idx = loader.match_coordinates()

    print(f"Loaded {len(df)} matches")
    print(df.head())
