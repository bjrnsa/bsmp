# %%
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class MatchDataLoader:
    """Data loader for match data with caching and dynamic filtering."""

    def __init__(
        self,
        sport: str = "handball",
        date_format: str = "%d.%m.%Y %H:%M",
        db_path: Union[str, Path] = "database/database.db",
        connection: Optional[sqlite3.Connection] = None,
    ):
        """
        Initialize data loader with configuration options.

        Parameters
        ----------
        sport : str, optional
            Sport type (e.g., 'handball', 'football'). Defaults to 'handball'.
        date_format : str, optional
            Date format for parsing dates. Defaults to '%d.%m.%Y %H:%M'.
        db_path : Union[str, Path], optional
            Path to SQLite database file. Defaults to 'database/database.db'.
        connection : Optional[sqlite3.Connection], optional
            Optional existing database connection. Defaults to None.
        """
        self.sport = sport
        self.date_format = date_format
        self.db_path = Path(db_path)
        self.conn = connection or self._create_connection()
        self._validate_tables()

    def _create_connection(self) -> sqlite3.Connection:
        """
        Establish database connection with error handling.

        Returns
        -------
        sqlite3.Connection
            The SQLite database connection.

        Raises
        ------
        FileNotFoundError
            If the database file does not exist.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def _validate_tables(self) -> None:
        """
        Verify required tables exist in the database.

        Raises
        ------
        ValueError
            If required tables are missing from the database.
        """
        required_tables = {f"{self.sport}_match_data", f"{self.sport}_current_clubs"}
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        if not required_tables.issubset(existing_tables):
            missing = required_tables - existing_tables
            raise ValueError(f"Missing required tables: {', '.join(missing)}")

    def _get_current_teams(
        self,
        league: str,
        seasons: Optional[List[str]] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        """
        Retrieve current teams with optional filters.

        Parameters
        ----------
        league : str
            League name.
        seasons : Optional[List[str]], optional
            List of season identifiers. Defaults to None.
        filters : Optional[Dict[str, str]], optional
            Additional filter criteria. Defaults to None.

        Returns
        -------
        pd.Series
            Series of team names.

        Raises
        ------
        ValueError
            If no teams are found for the specified criteria.
        """
        base_query = f"""
            SELECT team_name 
            FROM {self.sport}_current_clubs
            WHERE 1=1
        """

        params = []

        if league:
            base_query += " AND league = ?"
            params.append(league)

        if seasons:
            season_clause = " OR ".join(["season = ?" for _ in seasons])
            base_query += f" AND ({season_clause})"
            params.extend(seasons)

        if filters:
            filter_clauses = [f"{k} = ?" for k in filters]
            base_query += " AND " + " AND ".join(filter_clauses)
            params.extend(filters.values())

        teams = pd.read_sql(base_query, self.conn, params=params)["team_name"]
        if teams.empty:
            raise ValueError(f"No teams found for {seasons} {league}")
        return teams

    def load_matches(
        self,
        league: Optional[str] = None,
        seasons: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        team_filters: Optional[Dict[str, str]] = None,
        result_mapping: Dict[str, int] = {"H": 1, "A": -1, "D": 0},
    ) -> pd.DataFrame:
        """
        Load match data with flexible filtering options.

        Parameters
        ----------
        league : Optional[str], optional
            League name. Defaults to None.
        seasons : Optional[List[str]], optional
            List of season identifiers. Defaults to None.
        date_range : Optional[Tuple[str, str]], optional
            Tuple of (start_date, end_date) as strings. Defaults to None.
        team_filters : Optional[Dict[str, str]], optional
            Additional filters for teams table. Defaults to None.
        result_mapping : Dict[str, int], optional
            Custom result value mapping. Defaults to {"H": 1, "A": -1, "D": 0}.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame of match data.

        Raises
        ------
        sqlite3.Error
            If there is a database error.
        """
        try:
            teams = self._get_current_teams(league, seasons, team_filters)

            # Add team index mapping
            self.team_to_idx = {
                team: idx + 1 for idx, team in enumerate(sorted(teams.unique()))
            }

            query = f"""
                SELECT 
                    flashscore_id, season, match_info, datetime, league,
                    home_team, away_team, 
                    home_goals_full AS home_goals, 
                    away_goals_full AS away_goals,
                    result
                FROM {self.sport}_match_data
                WHERE home_team IN {tuple(teams)}
                AND away_team IN {tuple(teams)}
            """
            params = []
            if seasons:
                season_clause = " OR ".join(["season = ?" for _ in seasons])
                query += f" AND ({season_clause})"
                params.extend(seasons)

            df = pd.read_sql(
                query,
                self.conn,
                parse_dates={"datetime": {"format": self.date_format}},
                params=params,
            )

            if date_range:
                df = df.query("@date_range[0] <= datetime <= @date_range[1]")

            df = self._process_data(df, result_mapping)
            return df

        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")
            return pd.DataFrame()

    def _process_data(
        self, df: pd.DataFrame, result_mapping: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Post-process dataframe with common transformations.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to process.
        result_mapping : Dict[str, int]
            Mapping of result values.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame.
        """
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

    def _add_additional_features(
        self,
        df: pd.DataFrame,
        home_goals: str = "home_goals",
        away_goals: str = "away_goals",
    ) -> pd.DataFrame:
        """
        Add calculated features to dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to add features to.
        home_goals : str, optional
            Column name for home goals. Defaults to "home_goals".
        away_goals : str, optional
            Column name for away goals. Defaults to "away_goals".

        Returns
        -------
        pd.DataFrame
            The DataFrame with additional features.
        """
        return df.assign(
            goal_difference=lambda x: x[home_goals] - x[away_goals],
            total_goals=lambda x: x[home_goals] + x[away_goals],
        )

    def close(self) -> None:
        """
        Close database connection.
        """
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    pass
