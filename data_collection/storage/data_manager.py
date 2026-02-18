import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict
import hashlib
from ..base.base_collector import stock_data

class data_manager:
    def __init__(self, db_path: str = 'database/stock_data.db'): 
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stock_prices(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    data_hash TEXT UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            ''')
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON stock_prices(symbol, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_hash ON stock_prices(data_hash)")

    def _generate_hash(self, data: stock_data) -> str:
        hash_string = f"{data.symbol}_{data.timestamp}_{data.source}_{data.close}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def save_data(self, data: List[stock_data]) -> Dict[str, int]:
        stats = {'new': 0, 'duplicate': 0, 'error': 0}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor() 
            for record in data:
                try:
                    data_hash = self._generate_hash(record)
                    cursor.execute('''
                        INSERT OR IGNORE INTO stock_prices
                        (symbol, timestamp, open, high, low, close, volume, source, data_hash)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    ''', (
                        record.symbol,
                        record.timestamp.isoformat(),
                        record.open,
                        record.high,
                        record.low,
                        record.close,
                        record.volume,
                        record.source,
                        data_hash
                    ))

                    if cursor.rowcount > 0: 
                        stats['new'] += 1
                    else:
                        stats['duplicate'] += 1

                except Exception as e:
                    print(f'Error saving record: {e}')
                    stats['error'] += 1
        return stats

    def get_data(self, symbol:str, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
        query = 'SELECT * FROM stock_prices WHERE symbol = ?'
        params = [symbol]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        
        return df

    def get_latest_timestamp(self, symbol:str, source: str) -> Optional[datetime]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT MAX(timestamp) FROM stock_prices
                WHERE symbol = ? AND source = ?
            ''', (symbol, source))
            
            result = cursor.fetchone()[0]

            if result:
                return datetime.fromisoformat(result)
            return None

    def delete_duplicates(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM stock_prices
                WHERE id NOT IN (
                    SELECT MIN(id) FROM stock_prices GROUP BY data_hash  -- Исправлено: data_hesh -> data_hash
                )
            ''')
            deleted = cursor.rowcount
            conn.commit()
            return deleted 

    def get_statistics(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute('SELECT COUNT(*) FROM stock_prices').fetchone()[0]
            symbols = conn.execute('SELECT COUNT(DISTINCT symbol) FROM stock_prices').fetchone()[0] 

            date_range = conn.execute('''
                SELECT MIN(timestamp), MAX(timestamp) FROM stock_prices
            ''').fetchone()

        return {
            'total_records': total,
            'unique_symbols': symbols, 
            'earliest_date': date_range[0],
            'latest_date': date_range[1]
        }