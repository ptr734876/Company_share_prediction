from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from .base_news import base_news_parser, news_data

class base_news_collector(ABC):
    
    def __init__(self, name: str, config: Dict[str, Any], parser: base_news_parser):
        self.name = name
        self.config = config
        self.parser = parser
        self.logger = logging.getLogger(f'{__name__}.{name}')
        self._setup_logging()
    
    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    async def fetch_raw(self, start_date: datetime, end_date: datetime) -> Any:
        pass
    
    async def collect(self, start_date: datetime, end_date: datetime) -> List[news_data]:
        self.logger.info(f'Collecting news from {start_date} to {end_date}')
        
        try:
            raw_data = await self.fetch_raw(start_date, end_date)
            
            news_list = self.parser.parse(raw_data)
            
            filtered_news = [
                news for news in news_list
                if start_date <= news.published_at <= end_date
            ]
            
            self.logger.info(f'Collected {len(filtered_news)} news items')
            return filtered_news
            
        except Exception as e:
            self.logger.error(f'Error collecting news: {e}')
            return []
    
    def validate_config(self) -> bool:
        return self.parser.validate_config()