import asyncio
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.collectors.yahoo_finance import YahooFinanceCollector
from data_collection.storage.data_manager import data_manager

async def demonstrate_incremental_collection():
    print("="*50)
    print("Демонстрация инкрементального сбора данных")
    print("="*50)

    db = data_manager("./date_bases/second.db")
    
    config = {
        'interval': '1d', 
        'timeout': 30
    }
    
    collector = YahooFinanceCollector(config)
    
    if not collector.validate_config():
        print("Ошибка в конфигурации!")
        return
    
    symbol = "GOOGL" 
    
    last_date = db.get_latest_timestamp(symbol, 'yahoo_finance')
    
    start_date = datetime.now() - timedelta(days=1000)
    
    end_date = datetime.now()
    
    print(f"\nНачинаем сбор данных для {symbol}...")
    data = await collector.process(symbol, start_date, end_date)
    
    if data:
        print(f"Собрано {len(data)} записей")
        
        stats = db.save_data(data)
        print(f"\nРезультат сохранения:")
        print(f"  - Новых записей: {stats['new']}")
        print(f"  - Дубликатов: {stats['duplicate']}")
        print(f"  - Ошибок: {stats['error']}")
        
        print(f"\nПримеры собранных данных:")
        for i, record in enumerate(data[:3]): 
            print(f"  {i+1}. {record.timestamp.date()}: Open={record.open:.2f}, "
                  f"Close={record.close:.2f}, Volume={record.volume}")
        
        stats = db.get_statistics()
        print(f"\nСтатистика базы данных:")
        print(f"  - Всего записей: {stats['total_records']}")
        print(f"  - Уникальных символов: {stats['unique_symbols']}")
        print(f"  - Диапазон дат: {stats['earliest_date']} - {stats['latest_date']}")
        
    else:
        print("Не удалось собрать данные")

async def demonstrate_full_history(symbol: str):
    print("\n" + "="*50)
    print("Демонстрация сбора полной истории")
    print("="*50)
    
    db = data_manager("./date_bases/by_days.db")
    collector = YahooFinanceCollector({'interval': '1d', 'timeout': 100})
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    print(f"Собираем историю {symbol} с {start_date.date()} по {end_date.date()}")
    
    data = await collector.process(symbol, start_date, end_date)
    
    if data:
        stats = db.save_data(data)
        print(f"Сохранено {stats['new']} записей")
        
        df = db.get_data(symbol, start_date, end_date)
        print(f"В БД теперь {len(df)} записей для {symbol}")
        
        print(f"\nСтатистика по данным:")
        print(f"  - Средняя цена: ${df['close'].mean():.2f}")
        print(f"  - Максимум: ${df['high'].max():.2f}")
        print(f"  - Минимум: ${df['low'].min():.2f}")
        print(f"  - Общий объем торгов: {df['volume'].sum():,}")
    else:
        print("Не удалось собрать данные")

async def main():
    """
    Главная функция
    """
    await demonstrate_incremental_collection()
    print("\n" + "="*50)
    print("Проверка сохраненных данных")
    print("="*50)
    
    db = data_manager("./date_bases/second.db")
    
    df = db.get_data("GOOGL")
    
    if not df.empty:
        print(f"Данные для AAPL:")
        print(f"  - Записей: {len(df)}")
        print(f"  - Период: {df['timestamp'].min().date()} - {df['timestamp'].max().date()}")
        print(f"  - Колонки: {list(df.columns)}")
        
        print(f"\nПоследние 5 записей:")
        print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(100))
    else:
        print("Нет данных для AAPL")

if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(demonstrate_full_history('YNDX'))
    # asyncio.run(demonstrate_full_history('AAPL'))
    # asyncio.run(demonstrate_full_history('GOOGL'))
    # asyncio.run(demonstrate_full_history('TSLA'))
    # asyncio.run(demonstrate_full_history('AMZN'))
