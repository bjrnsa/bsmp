from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


class BrowserManager:
    def __init__(self, headless=False):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")

    @contextmanager
    def get_driver(self, url: str = None):
        """Yields a driver instance with optional initial navigation"""
        driver = webdriver.Chrome(options=self.options)
        try:
            if url:
                driver.get(url)
                self._handle_cookies(driver)
            yield driver
        finally:
            driver.quit()

    def _handle_cookies(self, driver):
        """Handle cookies after page load"""
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            ).click()
        except Exception as e:
            print(f"No cookie banner found: {str(e)}")


# Usage example
if __name__ == "__main__":
    browser = BrowserManager(headless=False)
    with browser.get_driver(url="https://www.flashscore.com") as driver:
        print(driver.title)
        # Additional page interactions here
