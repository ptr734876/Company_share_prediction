from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional

@dataclass
class stock_data:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    raw_data: Optional[Dict] = None

class base_collector(ABC):
    def __init__(self, name:str, config: Dict[str, Any]):
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
    async def fetch_data(self, symbol:str, start_date: datetime,
                        end_date: datetime) -> List[stock_data]:
        pass

    def validate_config(self) -> bool:
        pass

    async def process(self, symbol: str, start_date: datetime,
                    end_date: datetime) -> List[stock_data]:
        self.logger.info(f'Starting collection for {symbol} from {start_date} to {end_date}')
        try:
            date = await self.fetch_data(symbol, start_date, end_date)
            validated_data = self._validate_data(date)
            self.logger.info(f'Successfully collected {len(validated_data)} records')

            return validated_data

        except Exception as e:
            self.logger.error(f'Error collecting data: {e}')
            raise

    def _validate_data(self, data: List[stock_data]) -> List[stock_data]:
        valid_data = []
        for item in data:
            if self._is_valid_record(item):
                valid_data.append(item)
            else:
                self.logger.warning(f'Invalid record found: {item}')

        if len(valid_data) < len(data):
            self.logger.warning(f'Removed {len(data) - len(valid_data)} invalid records')
        return valid_data

    def _is_valid_record(self, record:stock_data) -> bool:
        if not record.timestamp:
            self.logger.debug("Record missing timestamp")
            return False
        if any(v is None for v in [record.open, record.high, record.low, record.close, record.volume]):
            self.logger.debug("Record missing numerical data")
            return False
        if record.high < record.low:
            self.logger.debug(f"High ({record.high}) < Low ({record.low})")
            return False
        if record.close < record.low or record.close > record.high:
            self.logger.debug(f"Close ({record.close}) outside range [{record.low}, {record.high}]")
            return False
        
        if record.open < record.low or record.open > record.high:
            self.logger.debug(f"Open ({record.open}) outside range [{record.low}, {record.high}]")
            return False

        if record.volume <= 0:
            self.logger.debug(f"Non-positive volume: {record.volume}") 
            return False
        return True
