import asyncio
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.collectors.yahoo_finance import YahooFinanceCollector
from data_collection.collectors.moex import moex_collector
from data_collection.storage.data_manager import data_manager

async def test_collector(collector_class, name, symbols, interval='1d'):
    print(f"\n{'='*40}\nТестирование {name} (interval={interval})\n{'='*40}")
    
    collector = collector_class({'interval': interval, 'timeout': 30})
    db_name = f"./databases/{name}_{interval}.db"
    db = data_manager(db_name)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        
        data = await collector.fetch_data(symbol, start_date, end_date)
        print(f"  Собрано: {len(data)} записей")
        if data:
            print(f"  Пример: {data[0].timestamp.date()} - {data[0].close}")
        
        info = collector.get_company_info(symbol)
        print(f"  Инфо: {info.get('longName', info.get('shortname', 'N/A'))[:30]}")
        
        price = await collector.get_realtime_price(symbol)
        print(f"  Цена: {price:.2f}")
        
        if data:
            stats = db.save_data(data)
            print(f"  Сохранено: {stats['new']} записей")
    
    stats = db.get_statistics()
    print(f"\nИтого в {db_name}: {stats['total_records']} записей, {stats['unique_symbols']} символов")

async def main():
    await test_collector(
        YahooFinanceCollector, 
        "yahoo", 
        ["AAPL", "GOOGL"],
        "1d"
    )
    
    await test_collector(
        moex_collector, 
        "moex", 
        ["SBER", "GAZP"],
        "1d"
    )
    
    await test_collector(
        moex_collector, 
        "moex", 
        ["LKOH"],
        "1h"
    )

if __name__ == "__main__":
    asyncio.run(main())