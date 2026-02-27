import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron_network.predictor import StockPredictor, PredictionResult

DB_NAME = "by_days.db"

def get_db_path(db_name: str = DB_NAME) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, 'date_bases', db_name)
    
    if not os.path.exists(db_path):
        alt_paths = [
            os.path.join(project_root, 'date_bases', 'second.db'),
            os.path.join(current_dir, '..', 'date_bases', db_name),
            os.path.join(current_dir, 'date_bases', db_name),
            './date_bases/' + db_name,
            '../date_bases/' + db_name,
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
    
    return db_path

async def test_training(symbol: str = 'GOOGL', 
                        train_start: datetime = datetime(2025, 1, 1),
                        train_end: datetime = datetime(2026, 2, 27)):
    print("="*60)
    print("Testing model training")
    print("="*60)
    
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    predictor = StockPredictor(db_path)
    
    print(f"\nTraining {symbol} {train_start.date()} - {train_end.date()}")
    
    try:
        model_name = f"{symbol}_{train_start.strftime('%Y%m%d')}-{train_end.strftime('%Y%m%d')}_{DB_NAME[:-3]}"
        
        history = predictor.train(
            symbol=symbol,
            start_date=train_start,
            end_date=train_end,
            epochs=10000,
            model_name=model_name
        )
        
        print(f"Training completed. Final loss: {history['train_loss'][-1]:.6f}")
        print(f"Best validation loss: {min(history['val_loss']):.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

async def test_prediction():
    print("\n" + "="*60)
    print("Testing predictions")
    print("="*60)
    
    db_path = get_db_path()
    predictor = StockPredictor(db_path)
    
    print("Loading model...")
    predictor.load_model("GOOGL_20250101-20260227_by_days")
    
    # Predict for a period that exists in database
    future_start = datetime(2026, 2, 26)
    future_end = datetime(2026, 2, 27)
    
    predictions = predictor.predict(
        symbol="GOOGL",
        start=future_start,
        end=future_end,
        compare_with_real=True
    )
    
    print(f"\nPredictions made: {len(predictions)}")
    print("\n" + "-"*90)
    print(f"{'Date':<12} {'Predicted':<12} {'Confidence':<15} {'Real':<10} {'Error %':<10} {'Status':<10}")
    print("-"*90)
    
    total_error = 0
    error_count = 0
    
    for pred in predictions:
        status = "✓ HAS DATA" if pred.has_real_data else "✗ NO DATA"
        
        real_str = f"${pred.real_close:.2f}" if pred.real_close else "N/A"
        error_str = f"{pred.error_percentage:.2f}%" if pred.error_percentage else "N/A"
        
        print(f"{pred.timestamp.date()}  "
            f"${pred.predicted_close:<10.2f} "
            f"[{pred.confidence_lower:<6.2f}-{pred.confidence_upper:<6.2f}]  "
            f"{real_str:<9} "
            f"{error_str:<9} "
            f"{status}")
        
        if pred.has_real_data and pred.error_percentage:
            total_error += pred.error_percentage
            error_count += 1
    
    print("-"*90)
    if error_count > 0:
        avg_error = total_error / error_count
        print(f"Average error: {avg_error:.2f}% (based on {error_count} days with real data)")
    else:
        print("No real data available for comparison")
            
async def test_resume_training():
    print("\n" + "="*60)
    print("Testing model fine-tuning")
    print("="*60)
    
    db_path = get_db_path()
    predictor = StockPredictor(db_path)
    
    history, new_model_name = predictor.resume_training(
        symbol="GOOGL",
        model_name="GOOGL_20250101-20260227_by_days",
        new_start_date=datetime(2024, 6, 15),
        new_end_date=datetime(2025, 1, 1),
        additional_epochs=5000
    )
    
    print(f"Fine-tuning completed. New model: {new_model_name}")

async def test_multiple_symbols():
    print("\n" + "="*60)
    print("Testing multiple symbols")
    print("="*60)
    
    db_path = get_db_path()
    symbols = ["AAPL", "TSLA", "AMZN"]
    
    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        
        predictor = StockPredictor(db_path)
        
        train_start = datetime(2022, 1, 1)
        train_end = datetime(2023, 6, 30)
        
        try:
            model_name = f"{symbol}_{train_start.strftime('%Y%m%d')}-{train_end.strftime('%Y%m%d')}_by_days"
            
            history = predictor.train(
                symbol=symbol,
                start_date=train_start,
                end_date=train_end,
                epochs=30,
                model_name=model_name
            )
            
            test_start = datetime(2023, 7, 1)
            test_end = datetime(2023, 7, 31)
            
            predictions = predictor.predict(
                symbol=symbol,
                start=test_start,
                end=test_end
            )
            
            print(f"Predictions: {len(predictions)}")
            print(f"Range: ${min(p.predicted_close for p in predictions):.2f} - "
                f"${max(p.predicted_close for p in predictions):.2f}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # asyncio.run(test_training())
    # asyncio.run(test_resume_training())
    asyncio.run(test_prediction())
    pass