from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Set
import re


@dataclass
class news_data:
    title: str
    content: str
    published_at: datetime
    source: str
    url: str
    categories: List[str]
    symbols: List[str]
    sentiment: Optional[float] = None
    author: Optional[str] = None
    raw_data: Optional[Dict] = None

    def __post_init__(self):
        if self.sentiment is not None:
            self.sentiment = max(-1.0, min(1.0, self.sentiment))


class base_news_parser(ABC):

    SYMBOL_DICTIONARY = {
        'AAPL': {'symbol': 'AAPL', 'name': 'Apple', 'keywords': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'tim cook']},
        'MSFT': {'symbol': 'MSFT', 'name': 'Microsoft', 'keywords': ['microsoft', 'windows', 'azure', 'office', 'satya', 'nadella', 'msft']},
        'GOOGL': {'symbol': 'GOOGL', 'name': 'Google', 'keywords': ['google', 'alphabet', 'gmail', 'chrome', 'android', 'pichai']},
        'AMZN': {'symbol': 'AMZN', 'name': 'Amazon', 'keywords': ['amazon', 'aws', 'bezos', 'prime', 'alexa', 'jassy']},
        'META': {'symbol': 'META', 'name': 'Meta', 'keywords': ['meta', 'facebook', 'instagram', 'whatsapp', 'zuckerberg']},
        'TSLA': {'symbol': 'TSLA', 'name': 'Tesla', 'keywords': ['tesla', 'elon', 'musk', 'cybertruck', 'model 3', 'model y']},
        'NVDA': {'symbol': 'NVDA', 'name': 'Nvidia', 'keywords': ['nvidia', 'jensen', 'huang', 'gpu', 'rtx', 'cuda']},
        'AMD': {'symbol': 'AMD', 'name': 'AMD', 'keywords': ['amd', 'ryzen', 'radeon', 'lisa su']},
        'INTC': {'symbol': 'INTC', 'name': 'Intel', 'keywords': ['intel', 'core i', 'xeon', 'pat gelsinger']},
        'CRM': {'symbol': 'CRM', 'name': 'Salesforce', 'keywords': ['salesforce', 'crm', 'benioff']},
        'NFLX': {'symbol': 'NFLX', 'name': 'Netflix', 'keywords': ['netflix', 'streaming', 'hastings']},
        'ADBE': {'symbol': 'ADBE', 'name': 'Adobe', 'keywords': ['adobe', 'photoshop', 'pdf', 'creative cloud']},
        'ORCL': {'symbol': 'ORCL', 'name': 'Oracle', 'keywords': ['oracle', 'ellison', 'database']},
        'IBM': {'symbol': 'IBM', 'name': 'IBM', 'keywords': ['ibm', 'watson', 'arvind krishna']},
        'JPM': {'symbol': 'JPM', 'name': 'JPMorgan', 'keywords': ['jpmorgan', 'jpm', 'jamie dimon', 'chase']},
        'BAC': {'symbol': 'BAC', 'name': 'Bank of America', 'keywords': ['bank of america', 'bofa', 'moynihan']},
        'WFC': {'symbol': 'WFC', 'name': 'Wells Fargo', 'keywords': ['wells fargo', 'charles scharf']},
        'C': {'symbol': 'C', 'name': 'Citigroup', 'keywords': ['citigroup', 'citi', 'jane fraser']},
        'GS': {'symbol': 'GS', 'name': 'Goldman Sachs', 'keywords': ['goldman', 'david solomon']},
        'MS': {'symbol': 'MS', 'name': 'Morgan Stanley', 'keywords': ['morgan stanley', 'james gorman']},
        'V': {'symbol': 'V', 'name': 'Visa', 'keywords': ['visa', 'digital payment', 'credit card']},
        'MA': {'symbol': 'MA', 'name': 'Mastercard', 'keywords': ['mastercard', 'michael miebach']},
        'AXP': {'symbol': 'AXP', 'name': 'American Express', 'keywords': ['american express', 'amex', 'stephen squeri']},
        'XOM': {'symbol': 'XOM', 'name': 'Exxon', 'keywords': ['exxon', 'exxonmobil', 'darren woods', 'oil']},
        'CVX': {'symbol': 'CVX', 'name': 'Chevron', 'keywords': ['chevron', 'mike wirth']},
        'COP': {'symbol': 'COP', 'name': 'ConocoPhillips', 'keywords': ['conoco', 'phillips', 'ryan lance']},
        'SLB': {'symbol': 'SLB', 'name': 'Schlumberger', 'keywords': ['schlumberger', 'oil services']},
        'EOG': {'symbol': 'EOG', 'name': 'EOG Resources', 'keywords': ['eog resources']},
        'OXY': {'symbol': 'OXY', 'name': 'Occidental', 'keywords': ['occidental', 'oxy', 'vicki hollub']},
        'WMT': {'symbol': 'WMT', 'name': 'Walmart', 'keywords': ['walmart', 'doug mcmillon']},
        'COST': {'symbol': 'COST', 'name': 'Costco', 'keywords': ['costco', 'craig jelinek']},
        'HD': {'symbol': 'HD', 'name': 'Home Depot', 'keywords': ['home depot', 'ted decker']},
        'MCD': {'symbol': 'MCD', 'name': 'McDonald\'s', 'keywords': ['mcdonald', 'mcdonalds', 'chris kempczinski']},
        'SBUX': {'symbol': 'SBUX', 'name': 'Starbucks', 'keywords': ['starbucks', 'howard schultz', 'laxman narasimhan']},
        'NKE': {'symbol': 'NKE', 'name': 'Nike', 'keywords': ['nike', 'john donahoe']},
        'DIS': {'symbol': 'DIS', 'name': 'Disney', 'keywords': ['disney', 'bob iger', 'marvel', 'star wars']},
        'JNJ': {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'keywords': ['johnson', 'janssen', 'joaquin duato']},
        'PFE': {'symbol': 'PFE', 'name': 'Pfizer', 'keywords': ['pfizer', 'albert bourla', 'vaccine']},
        'MRK': {'symbol': 'MRK', 'name': 'Merck', 'keywords': ['merck', 'rob davis']},
        'ABBV': {'symbol': 'ABBV', 'name': 'AbbVie', 'keywords': ['abbvie', 'richard gonzalez']},
        'UNH': {'symbol': 'UNH', 'name': 'UnitedHealth', 'keywords': ['unitedhealth', 'united health', 'andrew witty']},
        'BA': {'symbol': 'BA', 'name': 'Boeing', 'keywords': ['boeing', 'dave calhoun', '737', '787']},
        'CAT': {'symbol': 'CAT', 'name': 'Caterpillar', 'keywords': ['caterpillar', 'cat', 'jim umpleby']},
        'GE': {'symbol': 'GE', 'name': 'General Electric', 'keywords': ['ge', 'general electric', 'h lawrence culp']},
        'HON': {'symbol': 'HON', 'name': 'Honeywell', 'keywords': ['honeywell', 'darius adamczyk']},
        'UPS': {'symbol': 'UPS', 'name': 'UPS', 'keywords': ['ups', 'carol tomé']},
        'FDX': {'symbol': 'FDX', 'name': 'FedEx', 'keywords': ['fedex', 'raj subramaniam']},
        'YNDX': {'symbol': 'YNDX', 'name': 'Yandex', 'keywords': ['yandex', 'яндекс', 'yandex']},
        'SBER': {'symbol': 'SBER', 'name': 'Sberbank', 'keywords': ['sberbank', 'сбербанк', 'сбер']},
        'GAZP': {'symbol': 'GAZP', 'name': 'Gazprom', 'keywords': ['gazprom', 'газпром']},
        'LKOH': {'symbol': 'LKOH', 'name': 'Lukoil', 'keywords': ['lukoil', 'лукойл']},
        'ROSN': {'symbol': 'ROSN', 'name': 'Rosneft', 'keywords': ['rosneft', 'роснефть']},
    }

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
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
    def parse(self, raw_data: Any) -> List[news_data]:
        pass

    def validate_config(self) -> bool:
        return True

    def extract_symbols(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_symbols = set()

        for symbol, info in self.SYMBOL_DICTIONARY.items():
            if f' {symbol.lower()} ' in f' {text_lower} ':
                found_symbols.add(symbol)
                continue

            for keyword in info['keywords']:
                if keyword in text_lower:
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, text_lower):
                        found_symbols.add(symbol)
                        break

        return list(found_symbols)

    def categorize(self, text: str) -> List[str]:
        categories = []
        text_lower = text.lower()

        category_keywords = {
            'economy': {
                'keywords': ['fed', 'federal reserve', 'inflation', 'interest rate', 'gdp',
                           'unemployment', 'economic growth', 'recession', 'central bank',
                           'monetary policy', 'fiscal policy', 'stimulus', 'economy',
                           'pce', 'cpi', 'ppi', 'nonfarm payrolls', 'jobs report'],
                'weight': 1
            },
            'markets': {
                'keywords': ['stock market', 'wall street', 'dow jones', 's&p', 'nasdaq',
                           'trading', 'bull market', 'bear market', 'equities', 'bonds',
                           'yield curve', 'market rally', 'sell-off', 'correction',
                           'volatility', 'vix', 'shares', 'stocks', 'etf', 'index'],
                'weight': 1
            },
            'technology': {
                'keywords': ['tech', 'technology', 'software', 'ai', 'artificial intelligence',
                           'semiconductor', 'chip', 'cloud', 'digital', 'internet',
                           'cybersecurity', 'blockchain', 'crypto', 'bitcoin', 'ethereum',
                           '5g', 'quantum', 'robotics', 'automation', 'app', 'mobile'],
                'weight': 1
            },
            'commodities': {
                'keywords': ['oil', 'gold', 'silver', 'commodity', 'crude', 'natural gas',
                           'copper', 'wheat', 'corn', 'soybean', 'metal', 'precious metals',
                           'energy', 'brent', 'wti', 'futures', 'spot price'],
                'weight': 1
            },
            'politics': {
                'keywords': ['geopolitical', 'trade war', 'sanctions', 'election',
                           'policy', 'regulation', 'government', 'political', 'congress',
                           'senate', 'house', 'white house', 'president', 'trump', 'biden',
                           'putin', 'xi', 'ukraine', 'russia', 'china', 'iran', 'israel'],
                'weight': 1
            }
        }

        if any(word in text_lower for word in ['earnings', 'revenue', 'profit', 'loss',
                                               'quarterly', 'fiscal', 'guidance', 'forecast']):
            if 'markets' not in categories:
                categories.append('markets')

        scores = {cat: 0 for cat in category_keywords}

        for category, data in category_keywords.items():
            for keyword in data['keywords']:
                if keyword in text_lower:
                    scores[category] += data['weight']

        threshold = 1
        for category, score in scores.items():
            if score >= threshold:
                categories.append(category)

        return categories if categories else ['general']

