import sqlite3
import json
from datetime import datetime
from typing import List, Optional

from ..base.base_news import news_data


class SimpleNewsStorage:

    def __init__(self, db_path: str = 'database/news.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_news(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT,
                    published_at TIMESTAMP,
                    source TEXT,
                    url TEXT UNIQUE,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS news_symbols(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    FOREIGN KEY (news_id) REFERENCES raw_news(id) ON DELETE CASCADE
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS news_categories(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    FOREIGN KEY (news_id) REFERENCES raw_news(id) ON DELETE CASCADE
                )
            ''')

            conn.execute('CREATE INDEX IF NOT EXISTS idx_published ON raw_news(published_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON raw_news(source)')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_news_symbols_news_id ON news_symbols(news_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_news_symbols_symbol ON news_symbols(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_news_categories_news_id ON news_categories(news_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_news_categories_category ON news_categories(category)')

    def save_news(self, news_list: List[news_data]) -> int:
        saved = 0

        with sqlite3.connect(self.db_path) as conn:
            for news in news_list:
                try:
                    cursor = conn.execute('''
                        INSERT OR IGNORE INTO raw_news
                        (title, content, published_at, source, url, raw_data)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        news.title[:500],
                        news.content[:5000] if news.content else None,
                        news.published_at.isoformat(),
                        news.source,
                        news.url,
                        json.dumps(news.raw_data, default=str) if news.raw_data else None
                    ))

                    if conn.total_changes > 0:
                        saved += 1
                        news_id = cursor.lastrowid
                        
                        if news.symbols:
                            for symbol in news.symbols:
                                conn.execute('''
                                    INSERT INTO news_symbols (news_id, symbol)
                                    VALUES (?, ?)
                                ''', (news_id, symbol))
                        
                        if news.categories:
                            for category in news.categories:
                                conn.execute('''
                                    INSERT INTO news_categories (news_id, category)
                                    VALUES (?, ?)
                                ''', (news_id, category))

                except Exception as e:
                    print(f"Error saving news: {e}")
                    continue

            conn.commit()

        return saved

    def get_news(self,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                source: Optional[str] = None,
                limit: int = 1000,
                include_symbols: bool = True,
                include_categories: bool = True) -> List[dict]:
        query = "SELECT * FROM raw_news WHERE 1=1"
        params = []

        if start_date:
            query += " AND published_at >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND published_at <= ?"
            params.append(end_date.isoformat())

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            
            if not results or (not include_symbols and not include_categories):
                return results
            
            news_ids = [r['id'] for r in results]
            
            if include_symbols and news_ids:
                placeholders = ','.join('?' * len(news_ids))
                symbols_query = f'''
                    SELECT news_id, symbol FROM news_symbols 
                    WHERE news_id IN ({placeholders})
                '''
                cursor = conn.execute(symbols_query, news_ids)
                symbols_by_news = {}
                for row in cursor.fetchall():
                    news_id = row['news_id']
                    if news_id not in symbols_by_news:
                        symbols_by_news[news_id] = []
                    symbols_by_news[news_id].append(row['symbol'])
                
                for result in results:
                    result['symbols'] = symbols_by_news.get(result['id'], [])
            
            if include_categories and news_ids:
                placeholders = ','.join('?' * len(news_ids))
                categories_query = f'''
                    SELECT news_id, category FROM news_categories 
                    WHERE news_id IN ({placeholders})
                '''
                cursor = conn.execute(categories_query, news_ids)
                categories_by_news = {}
                for row in cursor.fetchall():
                    news_id = row['news_id']
                    if news_id not in categories_by_news:
                        categories_by_news[news_id] = []
                    categories_by_news[news_id].append(row['category'])
                
                for result in results:
                    result['categories'] = categories_by_news.get(result['id'], [])
            
            return results

    def get_symbols_for_news(self, news_id: int) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT symbol FROM news_symbols WHERE news_id = ?',
                (news_id,)
            )
            return [row[0] for row in cursor.fetchall()]

    def get_categories_for_news(self, news_id: int) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT category FROM news_categories WHERE news_id = ?',
                (news_id,)
            )
            return [row[0] for row in cursor.fetchall()]

    def get_news_by_symbol(self, symbol: str, limit: int = 100) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT n.* FROM raw_news n
                INNER JOIN news_symbols s ON n.id = s.news_id
                WHERE s.symbol = ?
                ORDER BY n.published_at DESC
                LIMIT ?
            ''', (symbol, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_news_by_category(self, category: str, limit: int = 100) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT n.* FROM raw_news n
                INNER JOIN news_categories c ON n.id = c.news_id
                WHERE c.category = ?
                ORDER BY n.published_at DESC
                LIMIT ?
            ''', (category, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_date(self, source: Optional[str] = None) -> Optional[datetime]:
        query = "SELECT MAX(published_at) FROM raw_news"
        params = []

        if source:
            query += " WHERE source = ?"
            params.append(source)

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(query, params).fetchone()[0]
            return datetime.fromisoformat(result) if result else None

