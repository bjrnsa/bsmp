# %%
from bsmp.scrapers.match_data import MatchDataScraper
from bsmp.scrapers.match_ids import MatchIDScraper
from bsmp.scrapers.odds_data import OddsDataScraper

match_id_scraper = MatchIDScraper()
match_id_scraper.scrape()

match_data_scraper = MatchDataScraper()
match_data_scraper.scrape()

odds_data_scraper = OddsDataScraper()
odds_data_scraper.scrape(batch_size=10)

# %%
