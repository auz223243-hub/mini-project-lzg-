import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_weather(csv_path):
    """
    Read weather data from a CSV file and format the datetime index.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing weather data
        
    Returns:
    --------
    df : DataFrame
        Weather data with datetime index
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    # Combine Year, Month, Day, Hour, Minute columns into a datetime object
    df['datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)
    # Sort the dataframe by the datetime index
    df = df.sort_index()
    return df

def plot_single_day(df, date_str):
    """
    Plot solar radiation components for a specific day.
    
    Parameters:
    -----------
    df : DataFrame
        Weather data with datetime index
    date_str : str
        Date string in 'YYYY-MM-DD' format
    """
    # Filter data for the specified date using the datetime index
    day_data = df[df.index.strftime('%Y-%m-%d')== date_str]
    if day_data.empty:
        print("No data for the specified date!")
        return
    
    # Create the figure
    plt.figure(figsize=(12,6))
    # Plot the three solar radiation components
    plt.plot(day_data.index, day_data['GHI'], label='GHI', linewidth=2)
    plt.plot(day_data.index, day_data['DNI'], label='DNI', linewidth=2)
    plt.plot(day_data.index, day_data['DHI'], label='DHI', linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel('Solar Radiation (W/m²)')
    plt.title(f'Solar Radiation Components on {date_str}')
    plt.legend(fontsize=10)
    
    # Format x-axis with hourly ticks
    hours = pd.date_range(day_data.index[0], day_data.index[-1], freq='h')
    plt.xticks(hours, [h.strftime('%H:00') for h in hours], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_daily_avg(df):
    """
    Plot average daily profile of solar radiation components across the entire dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Weather data with datetime index
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    # Extract hour from the datetime index
    df['hour'] = df.index.hour
    # Group by hour and calculate mean values for each radiation component
    grouped = df.groupby('hour')[['GHI','DNI','DHI']].mean().reset_index()

    plt.figure(figsize=(12,6))
    # Plot line charts with markers for each component
    plt.plot(grouped['hour'], grouped['GHI'], 'o-', label='GHI', linewidth=2, markersize=5)
    plt.plot(grouped['hour'], grouped['DNI'], 's-', label='DNI', linewidth=2, markersize=5)
    plt.plot(grouped['hour'], grouped['DHI'], '^-', label='DHI', linewidth=2, markersize=5)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Solar Radiation (W/m²)')
    plt.title('Average Daily Solar Radiation Components Throughout the Year')
    plt.xticks(np.arange(0,24,1))
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_monthly_avg(df):
    """
    Create a grid of subplots showing average daily profiles for each month.
    
    Parameters:
    -----------
    df : DataFrame
        Weather data with datetime index
    """
    # Create a copy and extract month and hour from the datetime index
    df = df.copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    # Group by month and hour, then calculate mean values
    grouped = df.groupby(['month','hour'])[['GHI','DNI','DHI']].mean().reset_index()
    
    # Create a 3x4 grid of subplots (one for each month)
    fig, axes = plt.subplots(3,4, figsize=(16,10), sharey=True)
    for i, ax in enumerate(axes.flat, start=1):
        # Filter data for the current month
        sub = grouped[grouped['month']==i]
        # Plot the three components for this month
        ax.plot(sub['hour'], sub['GHI'], marker='o', label='GHI', linewidth=2, markersize=4)
        ax.plot(sub['hour'], sub['DNI'], marker='o', label='DNI', linewidth=2, markersize=4)
        ax.plot(sub['hour'], sub['DHI'], marker='o', label='DHI', linewidth=2, markersize=4)
        ax.set_title(f'Month {i}')
        ax.set_xticks(range(0,24,3))
        ax.grid(True, linestyle='--', alpha=0.3)
        # Add x-axis labels only to bottom row subplots
        if i in [9,10,11,12]:
            ax.set_xlabel('Hour of Day')
        # Add y-axis labels only to leftmost column subplots
        if i in [1,5,9]:
            ax.set_ylabel('Solar Radiation (W/m²)')
            
    # Add a title and legend to the entire figure
    fig.suptitle('Monthly Average Daily Solar Radiation Profiles', fontsize=16, y=1.02)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    plt.tight_layout()

def plot_monthly_hourly(df):
    """
    Plot monthly average solar radiation components across the year.
    
    Parameters:
    -----------
    df : DataFrame
        Weather data with datetime index
    """
    df = df.copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    
    # Calculate monthly averages for each radiation component
    monthly_avg = df.groupby('month')[['GHI','DNI','DHI']].mean().reset_index()
    
    # Define month labels for x-axis
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(12,6))
    # Plot line charts with markers for each component
    plt.plot(month_labels, monthly_avg['GHI'], 'o-', label='GHI', linewidth=2, markersize=5)
    plt.plot(month_labels, monthly_avg['DNI'], 's-', label='DNI', linewidth=2, markersize=5)
    plt.plot(month_labels, monthly_avg['DHI'], '^-', label='DHI', linewidth=2, markersize=5)
    
    plt.xlabel('Month')
    plt.ylabel('Average Solar Radiation (W/m²)')
    plt.title('Monthly Average Solar Radiation Components')
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def get_season(month):
    """
    Map month number to season name.
    
    Parameters:
    -----------
    month : int
        Month number (1-12)
        
    Returns:
    --------
    str
        Season name in English
    """
    # Define seasons in English
    if month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    elif month in [9,10,11]:
        return 'Autumn'
    else:
        return 'Winter'

def plot_seasonal(df):
    """
    Create a bar chart showing seasonal average solar radiation components.
    
    Parameters:
    -----------
    df : DataFrame
        Weather data with datetime index
    """
    df = df.copy()
    df['month'] = df.index.month
    # Apply the get_season function to map months to seasons
    df['season'] = df['month'].apply(get_season)
    # Group by season and calculate mean values
    grouped = df.groupby('season')[['GHI','DNI','DHI']].mean().reset_index()
    
    # Sort seasons in chronological order
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    grouped['season'] = pd.Categorical(grouped['season'], categories=season_order, ordered=True)
    grouped = grouped.sort_values('season')
    
    # Set up the bar chart
    x = np.arange(len(grouped))
    width = 0.25
    plt.figure(figsize=(10,6))
    
    # Create grouped bars for each component
    bars1 = plt.bar(x - width, grouped['GHI'], width, label='GHI', alpha=0.8)
    bars2 = plt.bar(x, grouped['DNI'], width, label='DNI', alpha=0.8)
    bars3 = plt.bar(x + width, grouped['DHI'], width, label='DHI', alpha=0.8)
    
    plt.xlabel('Season')
    plt.ylabel('Average Solar Radiation (W/m²)')
    plt.title('Seasonal Average Solar Radiation Components')
    plt.xticks(x, grouped['season'])
    plt.legend(fontsize=10)
    
    # Add value labels above each bar
    def autolabel(bars):
        """Add text labels above each bar showing its height"""
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

# Example usage
if __name__ == '__main__':
    # Read weather data from CSV file
    csv_file = 'shenzhen_weather.csv'
    weather = read_weather(csv_file)
    
    # Generate and display various visualizations
    # Plot data for a specific day
    plot_single_day(weather, '2023-01-01')
    plt.show()
    
    # Plot average daily profile across the year
    plot_daily_avg(weather)
    plt.show()
    
    # Plot monthly average daily profiles
    plot_monthly_avg(weather)
    plt.show()
    
    # Plot seasonal averages as bar chart
    plot_seasonal(weather)
    plt.show()
    
    # Plot monthly averages across the year
    plot_monthly_hourly(weather)
    plt.show()
