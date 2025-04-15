from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta

app = Flask(__name__)

# Load model and data
model = joblib.load('aissmspricemodel.pkl')
raw_data = pd.read_csv('EngineeredData.csv')

# Define the expected column order
EXPECTED_FEATURE_ORDER = [
    'Market',
    'Arrivals (Tonnes)',
    'Variety',
    'Year',
    'Month',
    'Day',
    'Quarter',
    'DayOfWeek',
    'Lag_1',
    'Lag_2',
    'Lag_3',
    'Rolling_Mean_7',
    'Rolling_Mean_14',
    'Rolling_Mean_30',
    'Rolling_Std_7',
    'Rolling_Std_14',
    'Rolling_Std_30',
    'Arrival_Rolling_mean_3'
]

def engineer_features(market, variety):
    """Engineer features for a specific market and variety combination"""
    historical_data = raw_data[
        (raw_data['Market'] == market) & 
        (raw_data['Variety'] == variety)
    ].copy()
    
    if historical_data.empty:
        raise ValueError(f"No historical data found for {market} - {variety}")
        
    # Sort by date to ensure correct temporal order
    historical_data['Arrival Date'] = pd.to_datetime(historical_data['Arrival Date'])
    historical_data = historical_data.sort_values('Arrival Date')
    
    # Get the latest values for feature initialization
    latest_data = historical_data.iloc[-1].copy()
    latest_date = historical_data['Arrival Date'].max()
    
    # Calculate rolling statistics first
    prices = historical_data['Minimum Price(Rs./Quintal)']
    rolling_stats = {}
    for window in [7, 14, 30]:
        rolling_stats[f'Rolling_Mean_{window}'] = prices.tail(window).mean()
        rolling_stats[f'Rolling_Std_{window}'] = prices.tail(window).std()
    
    # Initialize all features in the expected order
    features = {
        'Market': market,
        'Arrivals (Tonnes)': latest_data['Arrivals (Tonnes)'],
        'Variety': variety,
        'Year': latest_date.year,
        'Month': latest_date.month,
        'Day': latest_date.day,
        'Quarter': (latest_date.month - 1) // 3 + 1,
        'DayOfWeek': latest_date.weekday(),
        'Lag_1': historical_data['Minimum Price(Rs./Quintal)'].iloc[-1],
        'Lag_2': historical_data['Minimum Price(Rs./Quintal)'].iloc[-2],
        'Lag_3': historical_data['Minimum Price(Rs./Quintal)'].iloc[-3],
        'Rolling_Mean_7': rolling_stats['Rolling_Mean_7'],
        'Rolling_Mean_14': rolling_stats['Rolling_Mean_14'],
        'Rolling_Mean_30': rolling_stats['Rolling_Mean_30'],
        'Rolling_Std_7': rolling_stats['Rolling_Std_7'],
        'Rolling_Std_14': rolling_stats['Rolling_Std_14'],
        'Rolling_Std_30': rolling_stats['Rolling_Std_30'],
        'Arrival_Rolling_mean_3': historical_data['Arrivals (Tonnes)'].tail(3).mean()
    }
    
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        required_fields = ['market', 'variety', 'days']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields. Please provide: {required_fields}'}), 400
            
        market = data['market']
        variety = data['variety']
        n_days = int(data['days'])
        
        if n_days < 1:
            return jsonify({'error': 'Number of days must be positive'}), 400
        
        # Engineer initial features
        try:
            current_features = engineer_features(market, variety)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': f'Error engineering features: {str(e)}'}), 500
        
        # Initialize rolling windows
        historical_data = raw_data[
            (raw_data['Market'] == market) & 
            (raw_data['Variety'] == variety)
        ]
        prices = historical_data['Minimum Price(Rs./Quintal)'].tail(30).tolist()
        arrivals = deque(historical_data['Arrivals (Tonnes)'].tail(3).tolist(), maxlen=3)
        
        # Make predictions
        predictions = []
        dates = []
        current_date = datetime.strptime(
            historical_data['Arrival Date'].max(),
            '%Y-%m-%d' if isinstance(historical_data['Arrival Date'].iloc[-1], str) else None
        )
        
        for i in range(n_days):
            # Update date for next prediction
            current_date += timedelta(days=1)
            
            # Update date-related features
            current_features['Year'] = current_date.year
            current_features['Month'] = current_date.month
            current_features['Day'] = current_date.day
            current_features['Quarter'] = (current_date.month - 1) // 3 + 1
            current_features['DayOfWeek'] = current_date.weekday()
            
            # Create DataFrame for prediction with correct column order
            df = pd.DataFrame([current_features])[EXPECTED_FEATURE_ORDER]
            df['Market'] = df['Market'].astype('category')
            df['Variety'] = df['Variety'].astype('category')
            
            # Make prediction
            price = float(model.predict(df)[0])
            predictions.append(round(price, 2))
            dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Update features for next prediction
            current_features['Lag_3'] = current_features['Lag_2']
            current_features['Lag_2'] = current_features['Lag_1']
            current_features['Lag_1'] = price
            
            # Update arrival features
            next_arrival = np.mean(arrivals)
            arrivals.append(next_arrival)
            current_features['Arrivals (Tonnes)'] = next_arrival
            current_features['Arrival_Rolling_mean_3'] = np.mean(arrivals)
            
            # Update rolling windows
            prices.append(price)
            if len(prices) > 30:
                prices.pop(0)
            
            # Update rolling statistics
            for window in [7, 14, 30]:
                current_features[f'Rolling_Mean_{window}'] = np.mean(prices[-window:])
                current_features[f'Rolling_Std_{window}'] = np.std(prices[-window:])
        
        return jsonify({
            'market': market,
            'variety': variety,
            'predictions': [
                {'date': date, 'price': price} 
                for date, price in zip(dates, predictions)
            ]
        })
        
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/markets', methods=['GET'])
def get_markets():
    """Get list of available markets"""
    return jsonify({'markets': raw_data['Market'].unique().tolist()})

@app.route('/varieties', methods=['GET'])
def get_varieties():
    """Get list of available varieties for a given market"""
    market = request.args.get('market')
    if not market:
        return jsonify({'error': 'Market parameter is required'}), 400
        
    varieties = raw_data[raw_data['Market'] == market]['Variety'].unique().tolist()
    return jsonify({'varieties': varieties})

if __name__ == '__main__':
    app.run(debug=True)