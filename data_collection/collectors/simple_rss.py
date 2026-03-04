import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List
import asyncio
import html
import re

from ..base.base_news import news_data
from ..base.news_collector import base_news_collector
from ..parsers.rss_parser import RSSParser


class SimpleRSSCollector(base_news_collector):

    RSS_FEEDS = {
        'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        'ft': 'https://www.ft.com/?format=rss',
        'wsj': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
        'yahoo': 'https://finance.yahoo.com/news/rssindex',
        'economist': 'https://www.economist.com/finance-and-economics/rss.xml',
        'marketwatch': 'https://feeds.content.dowjones.io/public/rss/mw_market_bulletin',
        'investing': 'https://www.investing.com/rss/news.rss',
        'zerohedge': 'https://feeds.feedburner.com/zerohedge/feed',
        'techcrunch': 'https://techcrunch.com/feed/',
        'recode': 'https://www.vox.com/recode/rss/index.xml',
        'businessinsider': 'https://www.businessinsider.com/rss',
        'fortune': 'https://fortune.com/feed',
        'barrons': 'https://www.barrons.com/feed/rss',
        'federalreserve': 'https://www.federalreserve.gov/feeds/press_all.xml',
        'ecb': 'https://www.ecb.europa.eu/rss/html/pr.en.html',
    }

    BACKUP_FEEDS = {
        'reuters': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best',
        'cnbc': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
    }

    def __init__(self, config: dict):
        parser = RSSParser({
            'source_name': config.get('source_name', 'simple_rss'),
            'max_items': config.get('max_per_source', 50)
        })
        super().__init__("simple_rss", config, parser)

        self.feeds = {}
        feed_names = config.get('feeds', list(self.RSS_FEEDS.keys()))

        for name in feed_names:
            if name in self.RSS_FEEDS:
                self.feeds[name] = self.RSS_FEEDS[name]
            elif name in self.BACKUP_FEEDS:
                self.feeds[name] = self.BACKUP_FEEDS[name]

    async def fetch_raw(self, start_date: datetime, end_date: datetime):
        pass

    async def collect(self, start_date: datetime, end_date: datetime) -> List[news_data]:
        self.logger.info(f"Collecting from {len(self.feeds)} sources")

        all_news = []
        timeout = aiohttp.ClientTimeout(total=15)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for source_name, url in self.feeds.items():
                tasks.append(self._fetch_feed(session, source_name, url))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)

        filtered = [
            news for news in all_news
            if start_date <= news.published_at <= end_date
        ]

        self.logger.info(f"Collected {len(filtered)} news items")
        return filtered

    async def _fetch_feed(self, session: aiohttp.ClientSession,
                         source_name: str, url: str) -> List[news_data]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            }

            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []

                content = await response.text()
                if not content or len(content) < 100:
                    return []

                feed = feedparser.parse(content)

                news_items = []
                max_items = self.config.get('max_per_source', 30)

                for entry in feed.entries[:max_items]:
                    try:
                        published = self._parse_date(entry)

                        if published < datetime.now() - timedelta(days=7):
                            continue

                        title = self._clean_text(entry.get('title', ''))
                        description = self._clean_text(entry.get('description', ''))
                        content_text = self._clean_text(entry.get('content', [{'value': ''}])[0].get('value', ''))

                        full_text = f"{title} {description} {content_text}".strip()

                        symbols = self.parser.extract_symbols(full_text)
                        categories = self.parser.categorize(full_text)

                        news = news_data(
                            title=title,
                            content=full_text[:2000],
                            published_at=published,
                            source=f"rss_{source_name}",
                            url=entry.get('link', ''),
                            categories=categories,
                            symbols=symbols,
                            raw_data={
                                'title': title,
                                'summary': description,
                                'authors': entry.get('authors', []),
                                'tags': [tag.get('term', '') for tag in entry.get('tags', [])]
                            }
                        )
                        news_items.append(news)

                    except Exception:
                        continue

                return news_items

        except Exception:
            return []

    def _parse_date(self, entry) -> datetime:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])

        return datetime.now()

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

