"""This module contains the BrowserManager class for managing the Selenium WebDriver."""

from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


class BrowserManager:
    """Manages the Selenium WebDriver for browser automation.

    Attributes:
    ----------
    options : selenium.webdriver.chrome.options.Options
        Chrome options for the WebDriver.
    """

    def __init__(self, headless=False):
        """Initializes the BrowserManager with optional headless mode.

        Parameters
        ----------
        headless : bool, optional
            If True, runs the browser in headless mode. Defaults to False.
        """
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")

    @contextmanager
    def get_driver(self, url: str | None = None):
        """Context manager that yields a WebDriver instance with optional initial navigation.

        Parameters
        ----------
        url : str, optional
            The URL to navigate to after initializing the driver. Defaults to None.

        Yields:
        ------
        selenium.webdriver.Chrome
            The Chrome WebDriver instance.
        """
        driver = webdriver.Chrome(options=self.options)
        try:
            if url:
                driver.get(url)
                self._handle_cookies(driver)
            yield driver
        finally:
            driver.quit()

    def _handle_cookies(self, driver):
        """Handles cookie consent banner after page load.

        Parameters
        ----------
        driver : selenium.webdriver.Chrome
            The Chrome WebDriver instance.
        """
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            ).click()
        except Exception as e:
            print(f"No cookie banner found: {str(e)}")


if __name__ == "__main__":
    pass
