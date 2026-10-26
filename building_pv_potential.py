import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
import pandas as pd
import calendar

# Basemap tiles: Esri satellite imagery needs no API key and covers all four cities
# (CartoDB now requires a key; Esri gray/street maps show "Map data not yet available" in Seoul and Shenzhen)
BASEMAP = ctx.providers.Esri.WorldImagery


def add_basemap(ax, buildings):
    """
    Add a basemap, capping the zoom level at the provider's maximum
    (otherwise small areas request tiles that show "Map data not yet available")
    
    Parameters:
    -----------
    ax : matplotlib Axes
        Axes in Web Mercator (EPSG:3857) coordinates
    buildings : GeoDataFrame
        Plotted buildings, used to choose the zoom level
    """
    west, south, east, north = buildings.to_crs(epsg=4326).total_bounds
    zoom = min(ctx.tile._calculate_zoom(west, south, east, north), BASEMAP.get('max_zoom', 19))
    ctx.add_basemap(ax, source=BASEMAP, zoom=zoom)


def visualize_buildings(geojson_path):
    """
    Visualize building footprints on a map
    
    Parameters:
    -----------
    geojson_path : str
        Path to the GeoJSON file containing building geometries
        
    Returns:
    --------
    buildings : GeoDataFrame
        GeoDataFrame containing the building data
    """
    # Load building data from GeoJSON file
    buildings = gpd.read_file(geojson_path)
    
    # Ensure the data is in the correct projection for web mapping (EPSG:3857)
    buildings = buildings.to_crs(epsg=3857)
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot buildings with semi-transparent fill and black outlines
    buildings.plot(ax=ax, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add a basemap for geographic context
    add_basemap(ax, buildings)
    
    # Set title and layout properties
    plt.title('Building Footprints', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    return buildings

def energy_per_m2(pvout_data, module_area):
    """
    Convert module power output (W per module) into energy per square meter
    of module (Wh/m²) for each time step

    Parameters:
    -----------
    pvout_data : DataFrame or Series
        Power output of ONE PV module (W) with datetime index;
        a DataFrame must contain a 'power' column
    module_area : float
        Area of one PV module (m²)

    Returns:
    --------
    Series : Energy per m² of module for each time step (Wh/m²)
    """
    # Handle different input data formats
    if isinstance(pvout_data, pd.DataFrame):
        if 'power' not in pvout_data.columns:
            raise ValueError("pvout_data must contain a 'power' column")
        power = pvout_data['power']
    else:
        power = pvout_data

    # Time step length in hours (10-min data -> 1/6 h), inferred from the index
    step_hours = pd.Series(power.index).diff().median() / pd.Timedelta(hours=1)

    # W per module x hours / m² per module = Wh per m²
    return power * step_hours / module_area

def calculate_pv_potential(buildings, pvout_data, roof_efficiency=0.8, module_area=1.69):
    """
    Calculate the photovoltaic potential for each building

    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame containing building geometries
    pvout_data : DataFrame or Series
        Power output of ONE PV module (W) with datetime index
    roof_efficiency : float, default=0.8
        Efficiency factor for usable roof area (accounting for HVAC equipment, spacing, etc.)
    module_area : float, default=1.69
        Area of one PV module (m²), must match the value used in the PV output calculation

    Returns:
    --------
    buildings : GeoDataFrame
        Updated GeoDataFrame with PV potential calculations
    """
    # Calculate the roof area (m²) in a local UTM projection.
    # Web Mercator (EPSG:3857) is only for map display: it inflates areas by 1/cos²(latitude)
    buildings['roof_area'] = buildings.to_crs(buildings.estimate_utm_crs()).geometry.area

    # Calculate usable area considering efficiency factor
    buildings['usable_area'] = buildings['roof_area'] * roof_efficiency

    # Annual energy yield per m² of PV module (Wh/m²)
    annual_yield = energy_per_m2(pvout_data, module_area).sum()

    # Calculate annual energy production in kWh for each building
    buildings['annual_energy'] = buildings['usable_area'] * annual_yield / 1000  # Convert from Wh to kWh

    return buildings

def visualize_pv_potential(buildings):
    """
    Visualize the solar PV potential of buildings on a map
    
    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame containing building geometries and calculated PV potential
    """
    # Create a custom color map for visualizing energy production
    colors = ['#FEF001', '#FFB403', '#FD6104', '#FD2008', '#BC0D0F']  # Yellow to red gradient
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot buildings colored by annual energy production
    buildings.plot(column='annual_energy', 
                  cmap=custom_cmap,
                  ax=ax,
                  legend=True,
                  legend_kwds={'label': 'Annual Energy Production (kWh)',
                              'orientation': 'horizontal',
                              'shrink': 0.8})
    
    # Add a basemap for geographic context
    add_basemap(ax, buildings)
    
    # Set title and layout properties
    plt.title('Solar PV Potential on Building Rooftops', fontsize=16)
    plt.axis('off')
    plt.tight_layout()

def create_statistics_plot(buildings):
    """
    Create statistical plots to analyze building PV potential
    
    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame containing building data with calculated PV potential
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram: Distribution of roof areas
    buildings['roof_area'].hist(ax=ax1, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Roof Areas')
    ax1.set_xlabel('Roof Area (m²)')
    ax1.set_ylabel('Number of Buildings')
    
    # Scatter plot: Roof area vs annual energy production
    ax2.scatter(buildings['roof_area'], buildings['annual_energy'], 
               alpha=0.5, c='orange')
    ax2.set_title('Roof Area vs Annual Energy Production')
    ax2.set_xlabel('Roof Area (m²)')
    ax2.set_ylabel('Annual Energy (kWh)')
    
    plt.tight_layout()

def visualize_building_annual_pv(buildings, building_id, pvout_data, module_area=1.69):
    """
    Visualize the annual PV potential variation for a specific building

    Parameters:
    -----------
    buildings : GeoDataFrame
        GeoDataFrame containing building information (after calculate_pv_potential)
    building_id : str or int
        ID of the building to visualize (value in the 'building_id' column;
        falls back to the row index if the column does not exist)
    pvout_data : DataFrame
        Power output of ONE PV module (W) with datetime index and a 'power' column
    module_area : float, default=1.69
        Area of one PV module (m²), must match the value used in calculate_pv_potential

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
        The created figure and axes objects
    """
    # Look up the building by its 'building_id' column (or by row index if absent)
    if 'building_id' in buildings.columns:
        match = buildings[buildings['building_id'] == building_id]
    else:
        match = buildings[buildings.index == building_id]

    # Check if the specified building ID exists in the dataset
    if match.empty:
        print(f"Building ID {building_id} not found!")
        return

    # Get the data for the specified building
    building = match.iloc[0]

    # Daily energy yield per m² of PV module (Wh/m²);
    # min_count=1 leaves days without data (e.g. a missing Feb 29) empty instead of 0
    daily_yield = energy_per_m2(pvout_data, module_area).resample('D').sum(min_count=1)

    # Calculate daily energy production for the building in kWh
    daily_energy = daily_yield * building['usable_area'] / 1000
    
    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot daily energy production
    ax.plot(daily_energy.index, daily_energy.values, 
            color='#FD6104', linewidth=2, marker='', alpha=0.8)
    
    # Set x-axis ticks to show month labels
    month_ticks = pd.date_range(start=daily_energy.index[0], 
                               end=daily_energy.index[-1], 
                               freq='MS')  # Month start frequency
    ax.set_xticks(month_ticks)
    ax.set_xticklabels([calendar.month_abbr[date.month] for date in month_ticks])
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set title and axis labels
    ax.set_title(f'Annual PV Potential for Building {building_id}', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Daily Energy Production (kWh)', fontsize=12)
    
    # Create text with building information
    info_text = (f"Roof Area: {building['roof_area']:.1f} m²\n"
                f"Usable Area: {building['usable_area']:.1f} m²\n"
                f"Annual Energy: {building['annual_energy']:.1f} kWh")
    
    # Add a text box with building information in the top-right corner
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

def main(geojson_path, pvout_data, building_id=None):
    """
    Main function to run the building PV potential analysis
    
    Parameters:
    -----------
    geojson_path : str
        Path to the GeoJSON file containing building geometries
    pvout_data : DataFrame or Series
        Power output of ONE PV module (W) with datetime index
    building_id : str or int, optional
        ID of a specific building to analyze
    """
    # Visualize building footprints
    buildings = visualize_buildings(geojson_path)
    plt.show()
    
    # Calculate PV potential for all buildings
    buildings = calculate_pv_potential(buildings, pvout_data)
    
    # If a specific building ID is provided, visualize its annual PV potential
    if building_id is not None:
        visualize_building_annual_pv(buildings, building_id, pvout_data)
        plt.show()
    else:
        # Otherwise, visualize PV potential for all buildings
        visualize_pv_potential(buildings)
        plt.show()
        
        # Create statistical plots for analysis
        create_statistics_plot(buildings)
        plt.show()
    
    # Print summary statistics
    print("\nSolar PV Potential Summary:")
    print(f"Total number of buildings: {len(buildings)}")
    print(f"Total roof area: {buildings['roof_area'].sum():,.0f} m²")
    print(f"Total annual energy production: {buildings['annual_energy'].sum():,.0f} kWh")
    print(f"Average annual energy per building: {buildings['annual_energy'].mean():,.0f} kWh")

if __name__ == "__main__":
    # Example usage
    from poa_calculation import calculate_poa
    from shenzhen_weather_visualization import read_weather
    
    # Read input data
    geojson_path = "path_to_your_buildings.geojson"
    weather_df = read_weather('shenzhen_weather.csv')
    poa_results = calculate_poa(weather_df, 30, 180, 22.54554, 114.0683)
    
    # Example 1: Show PV potential for all buildings
    main(geojson_path, poa_results)
    
    # Example 2: Show annual PV potential variation for a specific building
    # Note: Replace with an actual building ID that exists in your data
    # main(geojson_path, poa_results, building_id=1)  # Assuming building ID 1 exists
