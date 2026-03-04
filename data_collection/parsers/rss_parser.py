import feedparser
from datetime import datetime
from typing import List, Dict, Any
import re
import html

from ..base.base_news import base_news_parser, news_data


class RSSParser(base_news_parser):

    def __init__(self, config: Dict[str, Any]):
        super().__init__("rss_parser", config)

    def parse(self, raw_data: str) -> List[news_data]:
        try:
            feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

            feed = feedparser.parse(raw_data)

            if hasattr(feed, 'bozo') and feed.bozo and feed.bozo_exception:
                self.logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

            result = []
            max_items = self.config.get('max_items', 100)

            for entry in feed.entries[:max_items]:
                try:
                    published = self._parse_date(entry)

                    content = self._extract_content(entry)

                    content = self._clean_html(content)
                    title = self._clean_html(entry.title)

                    full_text = f"{title} {content}"

                    symbols = self.extract_symbols(full_text)

                    categories = self.categorize(full_text)

                    news = news_data(
                        title=title,
                        content=content[:5000],
                        published_at=published,
                        source=self.config.get('source_name', 'rss'),
                        url=entry.get('link', ''),
                        categories=categories,
                        symbols=symbols,
                        author=entry.get('author'),
                        raw_data=entry
                    )

                    result.append(news)

                except Exception as e:
                    self.logger.error(f"Error parsing entry: {e}")
                    continue

            self.logger.info(f"Parsed {len(result)} news items from RSS")
            return result

        except Exception as e:
            self.logger.error(f"Error parsing RSS feed: {e}")
            return []

    def _clean_html(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _parse_date(self, entry) -> datetime:
        date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']

        for field in date_fields:
            if hasattr(entry, field) and getattr(entry, field):
                try:
                    return datetime(*getattr(entry, field)[:6])
                except:
                    continue

        date_strings = []
        if hasattr(entry, 'published'):
            date_strings.append(entry.published)
        if hasattr(entry, 'updated'):
            date_strings.append(entry.updated)

        for date_str in date_strings:
            try:
                from dateutil import parser
                return parser.parse(date_str)
            except:
                continue

        return datetime.now()

    def _extract_content(self, entry) -> str:
        content_fields = [
            ('content', lambda x: x[0].value if x else None),
            ('summary', lambda x: x),
            ('description', lambda x: x),
            ('subtitle', lambda x: x),
        ]

        for field_name, extractor in content_fields:
            if hasattr(entry, field_name):
                value = getattr(entry, field_name)
                if value:
                    extracted = extractor(value)
                    if extracted:
                        return extracted

        return ""

