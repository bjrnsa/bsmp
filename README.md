# Bayesian Sports Models Project

This project implements Bayesian sports models using Stan, focusing on handball match data. The package includes tools for scraping match data, storing it in a SQLite database, and performing Bayesian analysis.

## Installation

I have set up the enviroment using [uv](https://astral.sh/blog/uv). Once installeed, you can clone the git, terminal into it and run:

```sh
uv sync
```

Apologies for any problems. This is my first attempt making anything package related.

## Scraper

The scraper is designed to extract handball match data from FlashScore and store it in a SQLite database. The scraper consists of several components:

- **BrowserManager**: Manages the Selenium WebDriver for browser automation.
- **DatabaseManager**: Manages SQLite database connections and operations.
- **MatchIDScraper**: Scrapes match IDs from FlashScore and stores them in the database.
- **MatchDataScraper**: Scrapes detailed match data and stores it in a structured format.
- **OddsDataScraper**: Scrapes and stores betting odds data from FlashScore matches.

### Usage

1. **Scrape Match IDs**

```python
from bsmp.scrapers.match_ids import MatchIDScraper

scraper = MatchIDScraper()
scraper.scrape()
```

2. **Scrape Match Data**

```python
from bsmp.scrapers.match_data import MatchDataScraper

scraper = MatchDataScraper()
scraper.scrape()
```

3. **Scrape Odds Data**

```python
from bsmp.scrapers.odds_data import OddsDataScraper

scraper = OddsDataScraper()
scraper.scrape()
```

## Bayesian Models

The package includes two Bayesian models for predicting match outcomes:

- **Negative Binomial Hierarchical Model**
- **Poisson Hierarchical Model**

### Usage

1. **Load Match Data**

```python
from bsmp.data_models.data_loader import MatchDataLoader

loader = MatchDataLoader(sport="handball")
df = loader.load_matches(
    league="Herre Handbold Ligaen",
    seasons=["2024/2025"],
)
```

2. **Fit Models**

```python
from bsmp.sports_model.bayesian.nbinom_hierarchical import NbinomHierarchical
from bsmp.sports_model.bayesian.poisson_hierarchical import PoissonHierarchical

nbinom_model = NbinomHierarchical(data=df, stem="nbinom_hierarchical")
nbinom_model.fit()

poisson_model = PoissonHierarchical(data=df, stem="poisson_hierarchical")
poisson_model.fit()
```

3. **Generate Predictions**

```python
team_1 = "Aalborg"
team_2 = "Skanderborg AGF"

nbinom_prediction = nbinom_model.predict(home_team=team_1, away_team=team_2, max_goals=90, n_samples=5000)
poisson_prediction = poisson_model.predict(home_team=team_1, away_team=team_2, max_goals=90, n_samples=5000)

print(nbinom_prediction)
print(poisson_prediction)
```

## Contents

- **bsmp/data_models**: Contains data loading and processing utilities.
- **bsmp/scrapers**: Contains scrapers for extracting match data, match IDs, and odds data from FlashScore.
- **bsmp/sports_model/bayesian**: Contains Bayesian models for predicting match outcomes.
- **bsmp/sports_model/utils.py**: Contains utility functions for the package.

## Work In Progress (WIP)

- Adding support for additional sports and leagues.
- Improving model performance and adding more advanced Bayesian models.
- Enhancing the scraper to handle more complex scenarios and additional data sources.
- Adding more comprehensive unit tests and documentation.

## Credits

- **Packages**: [Penalty Blog](https://github.com/martineastwood/penaltyblog/tree/master)
- **Author**: [Andrew Mack](https://www.amazon.com/stores/author/B07SGMRCVD)
- **References**:
  - [Stan Documentation](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
  - [cmdstanpy Documentation](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html)
