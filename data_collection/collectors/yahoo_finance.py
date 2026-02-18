
import yfinance as yf 
from datetime import datetime, timedelta
from typing import List
import asyncio 
from ..base.base_collector import base_collector, stock_data
import pandas as pd
class YahooFinanceCollector(base_collector):
    
    def __init__(self, config: dict):
        super().__init__("yahoo_finance", config)
    
    def validate_config(self) -> bool:
        required_fields = ['interval', 'timeout']
        
        for field in required_fields:
            if field not in self.config:
                self.logger.error(f"Missing required config field: {field}")
                return False
        
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if self.config['interval'] not in valid_intervals:
            self.logger.error(f"Invalid interval: {self.config['interval']}")
            return False
        
        return True
    
    async def fetch_data(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> List[stock_data]:
        try:
            self.logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
            
            loop = asyncio.get_event_loop()
            
            data = await loop.run_in_executor(
                None,
                self._fetch_sync,
                symbol,  
                start_date,
                end_date
            )
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return []
    
    def _fetch_sync(self, symbol: str, start_date: datetime, 
                   end_date: datetime) -> List[stock_data]:
        try:
            ticker = yf.Ticker(symbol)
            
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=self.config.get('interval', '1d')
            )
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return []
            
            result = []
            for index, row in df.iterrows():
                try:
                    data = stock_data(
                        symbol=symbol,
                        timestamp=index.to_pydatetime(),  
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']) if not pd.isna(row['Volume']) else 0,
                        source='yahoo_finance',
                        raw_data=row.to_dict() 
                    )
                    result.append(data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing row {index}: {e}")
                    continue
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _fetch_sync: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            self.logger.error(f"Error getting company info: {e}")
            return {}
    
    async def get_realtime_price(self, symbol: str) -> float:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.fast_info.last_price
        except Exception as e:
            self.logger.error(f"Error getting realtime price: {e}")
            return 0.0
