import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright
import logging
import json
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikisourceScraper:
    def __init__(self, chapter_url: str, chapter_num: int, output_dir: str = "chapters"):
        self.url = chapter_url
        self.chapter_num = chapter_num
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.text_file = self.output_dir / f"chapter_{self.chapter_num}.txt"
        self.screenshot_file = self.output_dir / f"chapter_{self.chapter_num}.png"
    async def scrape_chapter(self):
        """Scrape chapter content and take screenshot"""
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                logger.info(f"Loading page: {self.url}")
                await page.goto(self.url, wait_until="networkidle")
                
                # Wait for content to load
                await page.wait_for_selector("div#mw-content-text", timeout=10000)
                
                # Extract chapter content
                content = await self._extract_content(page)
                
                # Save text content
                text_file = self.output_dir / f"chapter_{self.chapter_num}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Content saved to {text_file}")
                
                # Take full page screenshot
                screenshot_file = self.output_dir / f"chapter_{self.chapter_num}.png"
                await page.screenshot(path=screenshot_file, full_page=True)
                logger.info(f"Screenshot saved to {screenshot_file}")
                
                return {
                    "text_file": str(text_file),
                    "screenshot_file": str(screenshot_file),
                    "content_length": len(content),
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error during scraping: {e}")
                return {"success": False, "error": str(e)}
            
            finally:
                await browser.close()
    
    async def _extract_content(self, page):
        """Extract main chapter content from the page"""
        try:
            # Try to get the main content div
            content_selector = "div#mw-content-text div.mw-parser-output"
            
            # Wait for content to be present
            await page.wait_for_selector(content_selector, timeout=5000)
            
            # Extract text content, focusing on paragraphs and preserving structure
            content_parts = []
            
            # Get the title
            title_element = await page.query_selector("h1#firstHeading")
            if title_element:
                title = await title_element.inner_text()
                content_parts.append(f"# {title}\n\n")
            
            # Get main content paragraphs
            paragraphs = await page.query_selector_all(f"{content_selector} p")
            
            for paragraph in paragraphs:
                text = await paragraph.inner_text()
                text = text.strip()
                if text and len(text) > 20:  # Filter out very short paragraphs
                    content_parts.append(text + "\n\n")
            
            # If no paragraphs found, try getting all text from content div
            if not content_parts or len(content_parts) <= 1:
                logger.warning("No paragraphs found, extracting all text from content div")
                content_div = await page.query_selector(content_selector)
                if content_div:
                    all_text = await content_div.inner_text()
                    # Clean up the text
                    lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                    content_parts = ['\n'.join(lines)]
            
            content = ''.join(content_parts).strip()
            
            if not content:
                raise Exception("No content extracted from page")
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            # Fallback: get all visible text
            try:
                body_text = await page.query_selector("body")
                fallback_content = await body_text.inner_text()
                logger.info("Using fallback content extraction")
                return fallback_content
            except:
                raise Exception("Failed to extract any content from page")

async def main():
    """Main execution function"""
    scraper = WikisourceScraper()
    
    logger.info("Starting Chapter 1 scraping process...")
    result = await scraper.scrape_chapter()
    
    if result["success"]:
        logger.info("✅ Scraping completed successfully!")
        logger.info(f"📄 Text file: {result['text_file']}")
        logger.info(f"📸 Screenshot: {result['screenshot_file']}")
        logger.info(f"📊 Content length: {result['content_length']} characters")
    else:
        logger.error(f"❌ Scraping failed: {result['error']}")
    
    return result

def install_dependencies():
    """Install required dependencies"""
    import subprocess
    import sys
    
    packages = ["playwright"]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Install playwright browsers
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        logger.info("Playwright browsers installed")
    except:
        logger.warning("Could not install Playwright browsers automatically")

# Add to imports
import argparse



# Add argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter_url", required=True)
    parser.add_argument("--chapter_num", type=int, required=True)
    args = parser.parse_args()
    
    scraper = WikisourceScraper(args.chapter_url, args.chapter_num)
    result = asyncio.run(scraper.scrape_chapter())
    
    # Print result for orchestrator
    print(json.dumps({
        "text_file": str(scraper.text_file),
        "screenshot_file": str(scraper.screenshot_file),
        "content_length": result.get("content_length", 0),
        "success": result.get("success", False)
    }))