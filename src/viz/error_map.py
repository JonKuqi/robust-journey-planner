import folium
from folium.plugins import HeatMap
import pandas as pd
from typing import List, Dict, Optional

def plot_error_heatmap(
    test_predictions: pd.DataFrame, 
    stop_metadata: Dict[int, Dict], 
    title: str = "Geographic Prediction Errors"
):
    """Plot a heatmap or markers showing where the model most under-predicts delays.
    
    Args:
        test_predictions: DataFrame with ['to_stop', 'actual_delay_sec', 'pred_delay_sec']
        stop_metadata: Dict mapping stop_id -> {'stop_lat', 'stop_lon'}
    """
    # Calculate error (residuals)
    test_predictions['error'] = test_predictions['actual_delay_sec'] - test_predictions['pred_delay_sec']
    
    # Filter for under-predictions (risk areas)
    under_preds = test_predictions[test_predictions['error'] > 0].copy()
    
    # Map stop metadata
    def get_coords(stop_id):
        meta = stop_metadata.get(int(stop_id), {})
        lat = meta.get('stop_lat')
        lon = meta.get('stop_lon')
        if lat is not None and lon is not None:
            return lat, lon
        return None, None

    under_preds[['lat', 'lon']] = under_preds['to_stop'].apply(lambda x: pd.Series(get_coords(x)))
    
    # Drop stops with missing coordinates gracefully
    valid_points = under_preds.dropna(subset=['lat', 'lon'])
    
    if valid_points.empty:
        print("Warning: No valid coordinates found for error heatmap.")
        return None

    # Center map on the average location
    center = [valid_points['lat'].mean(), valid_points['lon'].mean()]
    m = folium.Map(location=center, zoom_start=11)
    
    # Add HeatMap of errors
    heat_data = [[row['lat'], row['lon'], row['error']] for _, row in valid_points.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.3).add_to(m)
    
    # Add some markers for the worst offenders
    worst_offenders = valid_points.nlargest(10, 'error')
    for _, row in worst_offenders.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"Stop: {row['to_stop']}<br>Error: {row['error']:.1f}s",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    return m
