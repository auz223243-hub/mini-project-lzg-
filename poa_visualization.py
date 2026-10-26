import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_poa_single_day(poa_results, date_str):
    """
    Plot Plane of Array (POA) irradiance components for a single day.
    
    Parameters:
    -----------
    poa_results : DataFrame
        DataFrame containing POA irradiance components with datetime index
    date_str : str
        Date string in 'YYYY-MM-DD' format
    """
    # Filter data for the specified date
    day_data = poa_results[poa_results.index.strftime('%Y-%m-%d') == date_str]
    if day_data.empty:
        print("No data for the specified date!")
        return
        
    # Create figure and plot each POA component
    plt.figure(figsize=(12,6))
    plt.plot(day_data.index, day_data['poa_direct'], label='Direct', linewidth=2)
    plt.plot(day_data.index, day_data['poa_sky_diffuse'], label='Sky Diffuse', linewidth=2)
    plt.plot(day_data.index, day_data['poa_ground'], label='Ground Reflected', linewidth=2)
    plt.plot(day_data.index, day_data['total_poa'], label='Total POA', linewidth=2)
    
    # Set labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('POA Irradiance (W/m²)')
    plt.title(f'POA Components on {date_str}')
    plt.legend(fontsize=10)
    
    # Format x-axis with hourly ticks
    hours = pd.date_range(day_data.index[0], day_data.index[-1], freq='h')
    plt.xticks(hours, [h.strftime('%H:00') for h in hours], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_poa_daily_avg(poa_results):
    """
    Plot average daily profile of POA irradiance components across the entire dataset.
    
    Parameters:
    -----------
    poa_results : DataFrame
        DataFrame containing POA irradiance components with datetime index
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = poa_results.copy()
    # Extract hour from the datetime index
    df['hour'] = df.index.hour
    # Define columns to plot
    columns = ['poa_direct', 'poa_sky_diffuse', 'poa_ground', 'total_poa']
    # Group by hour and calculate mean values for each component
    grouped = df.groupby('hour')[columns].mean().reset_index()

    # Create figure and plot each component
    plt.figure(figsize=(12,6))
    for col in columns:
        plt.plot(grouped['hour'], grouped[col], 'o-', label=col.replace('poa_', '').title(), 
                linewidth=2, markersize=5)
    
    # Set labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('Average POA Irradiance (W/m²)')
    plt.title('Average Daily POA Components Throughout the Year')
    plt.xticks(np.arange(0,24,1))
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_poa_monthly_avg(poa_results):
    """
    Plot monthly average POA irradiance components across the year.
    
    Parameters:
    -----------
    poa_results : DataFrame
        DataFrame containing POA irradiance components with datetime index
    """
    # Create a copy and extract month from the datetime index
    df = poa_results.copy()
    df['month'] = df.index.month
    # Define columns to plot
    columns = ['poa_direct', 'poa_sky_diffuse', 'poa_ground', 'total_poa']
    # Group by month and calculate mean values
    monthly_avg = df.groupby('month')[columns].mean().reset_index()
    
    # Define month labels for x-axis
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create figure and plot each component
    plt.figure(figsize=(12,6))
    for col in columns:
        plt.plot(month_labels, monthly_avg[col], 'o-', 
                label=col.replace('poa_', '').title(), linewidth=2, markersize=5)
    
    # Set labels and title
    plt.xlabel('Month')
    plt.ylabel('Average POA Irradiance (W/m²)')
    plt.title('Monthly Average POA Components')
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_poa_seasonal(poa_results):
    """
    Plot seasonal average POA irradiance components as a bar chart.
    
    Parameters:
    -----------
    poa_results : DataFrame
        DataFrame containing POA irradiance components with datetime index
    """
    # Create a copy and extract month from the datetime index
    df = poa_results.copy()
    df['month'] = df.index.month
    # Assign seasons based on month
    df['season'] = df['month'].apply(lambda m: 'Spring' if m in [3,4,5] else
                                            'Summer' if m in [6,7,8] else
                                            'Autumn' if m in [9,10,11] else
                                            'Winter')
    
    # Define columns to plot
    columns = ['poa_direct', 'poa_sky_diffuse', 'poa_ground', 'total_poa']
    # Group by season and calculate mean values
    grouped = df.groupby('season')[columns].mean().reset_index()
    
    # Ensure consistent season order
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    grouped['season'] = pd.Categorical(grouped['season'], categories=season_order, ordered=True)
    grouped = grouped.sort_values('season')
    
    # Create bar chart with grouped bars for each component
    x = np.arange(len(grouped))
    width = 0.2
    plt.figure(figsize=(12,6))
    
    for i, col in enumerate(columns):
        bars = plt.bar(x + i*width, grouped[col], width, 
                      label=col.replace('poa_', '').title(), alpha=0.8)
        # Add value labels above each bar
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{int(bar.get_height())}',
                    ha='center', va='bottom')
    
    # Set labels and title
    plt.xlabel('Season')
    plt.ylabel('Average POA Irradiance (W/m²)')
    plt.title('Seasonal Average POA Components')
    plt.xticks(x + width*1.5, grouped['season'])
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def compare_fixed_tracked_daily(poa_fixed, poa_tracked):
    """
    Compare daily average total POA irradiance between fixed and tracked PV systems.
    
    Parameters:
    -----------
    poa_fixed : DataFrame
        DataFrame containing POA irradiance data for fixed system with datetime index
    poa_tracked : DataFrame
        DataFrame containing POA irradiance data for tracking system with datetime index
    """
    # Prepare data
    df_fixed = poa_fixed.copy()
    df_tracked = poa_tracked.copy()
    
    # Extract hour from datetime index
    df_fixed['hour'] = df_fixed.index.hour
    df_tracked['hour'] = df_tracked.index.hour
    
    # Calculate hourly averages
    fixed_hourly = df_fixed.groupby('hour')['total_poa'].mean()
    tracked_hourly = df_tracked.groupby('hour')['total_poa'].mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(fixed_hourly.index, fixed_hourly.values, 'o-', 
             label='Fixed Installation', linewidth=2, markersize=6)
    plt.plot(tracked_hourly.index, tracked_hourly.values, 's-', 
             label='Solar Tracking', linewidth=2, markersize=6)
    
    # Set labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('Average POA Irradiance (W/m²)')
    plt.title('Comparison of Daily Average POA: Fixed vs Tracking')
    plt.xticks(np.arange(0, 24, 1))
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add percentage gain annotations
    for hour in range(24):
        if fixed_hourly[hour] > 50:  # Only add annotations for meaningful irradiance values
            gain = ((tracked_hourly[hour] - fixed_hourly[hour]) / fixed_hourly[hour] * 100)
            plt.text(hour, max(tracked_hourly[hour], fixed_hourly[hour]), 
                    f'{gain:+.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()

def plot_pv_power_daily(pvout_fixed, pvout_tracked):
    """
    Compare daily average power output between fixed and tracked PV systems.
    
    Parameters:
    -----------
    pvout_fixed : DataFrame
        DataFrame containing power output data for fixed system with datetime index
    pvout_tracked : DataFrame
        DataFrame containing power output data for tracking system with datetime index
    """
    # Prepare data
    df_fixed = pvout_fixed.copy()
    df_tracked = pvout_tracked.copy()
    
    # Extract hour and calculate hourly averages
    df_fixed['hour'] = df_fixed.index.hour
    df_tracked['hour'] = df_tracked.index.hour
    
    fixed_hourly = df_fixed.groupby('hour')['power'].mean()
    tracked_hourly = df_tracked.groupby('hour')['power'].mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(fixed_hourly.index, fixed_hourly.values, 'o-', 
             label='Fixed System', linewidth=2, markersize=6)
    plt.plot(tracked_hourly.index, tracked_hourly.values, 's-', 
             label='Tracking System', linewidth=2, markersize=6)
    
    # Set labels and title
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Power Output (W)')
    plt.title('Daily Average Power Output Comparison: Fixed vs Tracking')
    plt.xticks(np.arange(0, 24, 1))
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add percentage gain annotations
    for hour in range(24):
        if fixed_hourly[hour] > 5:  # Only add annotations for meaningful power values
            gain = ((tracked_hourly[hour] - fixed_hourly[hour]) / fixed_hourly[hour] * 100)
            plt.text(hour, max(tracked_hourly[hour], fixed_hourly[hour]), 
                    f'{gain:+.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()

# Usage example
if __name__ == '__main__':
    from poa_calculation import calculate_poa
    from shenzhen_weather_visualization import read_weather
    
    # Read weather data and calculate POA
    weather_df = read_weather('shenzhen_weather.csv')
    poa_results = calculate_poa(weather_df, 30, 180, 22.54554, 114.0683)
    
    # Create all visualizations
    plot_poa_single_day(pd.DataFrame(poa_results), '2023-01-01')
    plt.show()
    
    plot_poa_daily_avg(pd.DataFrame(poa_results))
    plt.show()
    
    plot_poa_monthly_avg(pd.DataFrame(poa_results))
    plt.show()
    
    plot_poa_seasonal(pd.DataFrame(poa_results))
    plt.show()
