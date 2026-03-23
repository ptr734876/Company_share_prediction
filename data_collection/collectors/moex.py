import asyncio
from datetime import datetime, timedelta
from typing import List
import pandas as pd
from moexalgo import Ticker
from ..base.base_collector import base_collector, stock_data


class moex_collector(base_collector):
    
    def __init__(self, config: dict):
        super().__init__("moex", config)
    
    def validate_config(self) -> bool:
        return super().validate_config()
    
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
            self.logger.error(f"Error fetching data from MOEX: {e}")
            return []
    

    def _fetch_sync(self, symbol: str, start_date: datetime, 
                    end_date: datetime) -> List[stock_data]:
        try:
            ticker = Ticker(symbol)
            
            interval = self.config.get('interval', '1d')
            interval_map = {
                '1d': '1D',
                '1h': '1h',
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1wk': '1W',
                '1mo': '1M'
            }
            period = interval_map.get(interval, '1D')
            
            candles = ticker.candles(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                period=period
            )
            
            if candles is None:
                return []
            
            if isinstance(candles, pd.DataFrame):
                if candles.empty:
                    return []
                candles = candles.to_dict('records')
            elif not candles:
                return []
            
            result = []
            for candle in candles:
                try:
                    timestamp = candle['begin']
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            timestamp = datetime.strptime(timestamp, '%Y-%m-%d')
                    
                    data = stock_data(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(candle['open']),
                        high=float(candle['high']),
                        low=float(candle['low']),
                        close=float(candle['close']),
                        volume=int(candle['volume']) if candle.get('volume') else 0,
                        source='moex',
                        raw_data=candle
                    )
                    result.append(data)
                except Exception as e:
                    self.logger.error(f"Error processing candle: {e}")
                    continue
            
            return result
        except Exception as e:
            self.logger.error(f"Error in _fetch_sync: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> dict:
        try:
            ticker = Ticker(symbol)
            return ticker.info()
        except Exception as e:
            self.logger.error(f"Error getting company info: {e}")
            return {}
    
    async def get_realtime_price(self, symbol: str) -> float:
        try:
            ticker = Ticker(symbol)
            
            loop = asyncio.get_event_loop()
            candles = await loop.run_in_executor(
                None,
                lambda: ticker.candles(
                    start=datetime.now().strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d'),
                    period=1,
                    latest=True
                )
            )
            
            if candles is None:
                return 0.0
            
            if isinstance(candles, pd.DataFrame):
                if candles.empty:
                    return 0.0
                return float(candles.iloc[-1]['close'])
            elif candles:
                return float(candles[-1]['close'])
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting realtime price: {e}")
            return 0.0