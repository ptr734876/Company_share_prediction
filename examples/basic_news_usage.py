import asyncio
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.collectors.simple_rss import SimpleRSSCollector
from data_collection.storage.simple_news_storage import SimpleNewsStorage


async def collect_from_all_sources():
    print("=" * 60)
    print("COLLECTING FROM ALL AVAILABLE SOURCES")
    print("=" * 60)

    config = {
        'feeds': [
            'bloomberg', 'ft', 'wsj', 'yahoo', 'economist',
            'marketwatch', 'investing', 'zerohedge', 'techcrunch',
            'businessinsider', 'fortune', 'barrons', 'federalreserve'
        ],
        'max_per_source': 20000
    }

    storage = SimpleNewsStorage("./date_bases/all_news.db")
    collector = SimpleRSSCollector(config)

    end = datetime.now()
    start = end - timedelta(days=1000)

    print(f"\nPeriod: {start} - {end}")
    print(f"Sources: {len(config['feeds'])}")

    news = await collector.collect(start, end)

    if news:
        saved = storage.save_news(news)

        print(f"\nCollected: {len(news)} news items")
        print(f"Saved: {saved}")

        sources = {}
        for item in news:
            sources[item.source] = sources.get(item.source, 0) + 1

        print("\nBY SOURCE:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} news")

        print("\nNEWS SAMPLES:")
        for i, item in enumerate(news[:10]):
            print(f"\n{i+1}. [{item.source}] {item.title}")
            print(f"   {item.published_at.strftime('%H:%M %d.%m')}")

    else:
        print("No news found")


async def incremental_collect():
    print("\n" + "=" * 60)
    print("INCREMENTAL COLLECTION")
    print("=" * 60)

    storage = SimpleNewsStorage("./date_bases/all_news.db")
    collector = SimpleRSSCollector({'feeds': ['bloomberg', 'ft', 'wsj']})

    last_date = storage.get_latest_date()

    if last_date:
        print(f"Last news: {last_date}")
        start = last_date - timedelta(hours=1)
    else:
        print("Initial collection")
        start = datetime.now() - timedelta(days=3)

    end = datetime.now()

    print(f"Collecting for last {(end - start).total_seconds() / 3600:.1f} hours")

    news = await collector.collect(start, end)

    if news:
        saved = storage.save_news(news)
        print(f"New items saved: {saved}")

        hours = {}
        for item in news:
            hour = item.published_at.strftime('%Y-%m-%d %H:00')
            hours[hour] = hours.get(hour, 0) + 1

        print("\nBy hour:")
        for hour, count in sorted(hours.items()):
            print(f"  {hour}: {count} news")
    else:
        print("No new news")


if __name__ == "__main__":
    asyncio.run(collect_from_all_sources())
