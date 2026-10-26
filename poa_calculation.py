import pvlib
import pandas as pd
import numpy as np
from timezonefinder import TimezoneFinder
import pytz

def get_timezone(latitude, longitude):
    """Get the timezone for the specified latitude and longitude"""
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
    return pytz.timezone(timezone_str)

def calculate_poa(weather_data, surface_tilt, surface_azimuth, latitude, longitude):
    """
    Calculate Plane of Array (POA) irradiance
    
    Parameters:
    -----------
    weather_data : pd.DataFrame
        DataFrame containing GHI, DNI, DHI with datetime index
    surface_tilt : float
        Surface tilt angle in degrees (0-90)
    surface_azimuth : float
        Surface azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)
    latitude : float
        Geographic latitude
    longitude : float
        Geographic longitude
    """
    # Get timezone for the location
    tz = get_timezone(latitude, longitude)
    
    # Ensure datetime index has correct timezone information
    if weather_data.index.tz is None:
        weather_data.index = weather_data.index.tz_localize(tz)
    elif weather_data.index.tz != tz:
        weather_data.index = weather_data.index.tz_convert(tz)
    
    # 1. Get solar position
    solar_position = pvlib.solarposition.get_solarposition(
        time=weather_data.index,
        latitude=latitude,
        longitude=longitude
    )
    
    # 2. Calculate POA irradiance components
    # Calculate sky diffuse irradiance using Perez model
    poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dhi=weather_data['DHI'],
        dni=weather_data['DNI'],
        ghi=weather_data['GHI'],
        dni_extra=pvlib.irradiance.get_extra_radiation(weather_data.index),
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model='perez'
    )
    
    # Calculate direct beam irradiance
    poa_direct = pvlib.irradiance.beam_component(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=weather_data['DNI']
    )
    
    # Calculate ground reflected irradiance (assuming albedo of 0.2)
    poa_ground = pvlib.irradiance.get_ground_diffuse(
        surface_tilt=surface_tilt,
        ghi=weather_data['GHI'],
        albedo=0.2
    )
    
    # 3. Calculate total POA irradiance
    total_poa = poa_direct + poa_sky_diffuse + poa_ground
    
    return {
        'poa_direct': poa_direct,
        'poa_sky_diffuse': poa_sky_diffuse,
        'poa_ground': poa_ground,
        'total_poa': total_poa,
        'solar_zenith': solar_position['apparent_zenith'],
        'solar_azimuth': solar_position['azimuth']
    }

def calculate_poa_track(weather_data, latitude, longitude):
    """
    Calculate POA irradiance for a dual-axis tracking system
    
    Parameters:
    -----------
    weather_data : pd.DataFrame
        DataFrame containing GHI, DNI, DHI with datetime index
    latitude : float
        Geographic latitude
    longitude : float
        Geographic longitude
        
    Returns:
    --------
    dict : Dictionary containing POA irradiance components and solar position information
    """
    # Get timezone for the location
    tz = get_timezone(latitude, longitude)
    
    # Ensure datetime index has correct timezone information
    if weather_data.index.tz is None:
        weather_data.index = weather_data.index.tz_localize(tz)
    elif weather_data.index.tz != tz:
        weather_data.index = weather_data.index.tz_convert(tz)
    
    # Get solar position
    solar_position = pvlib.solarposition.get_solarposition(
        time=weather_data.index,
        latitude=latitude,
        longitude=longitude
    )
    
    # For dual-axis tracking system:
    # surface_tilt = solar_zenith (panel always perpendicular to sun rays)
    # surface_azimuth = solar_azimuth (panel always facing the sun's azimuth)
    surface_tilt = solar_position['apparent_zenith']
    surface_azimuth = solar_position['azimuth']
    
    # Calculate POA irradiance components
    # Direct irradiance - for tracking systems, this equals DNI as incidence is normal
    poa_direct = weather_data['DNI']  # With dual-axis tracking, direct component equals DNI
    
    # Sky diffuse irradiance
    poa_sky_diffuse = pvlib.irradiance.get_sky_diffuse(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dhi=weather_data['DHI'],
        dni=weather_data['DNI'],
        ghi=weather_data['GHI'],
        dni_extra=pvlib.irradiance.get_extra_radiation(weather_data.index),
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model='perez'
    )
    
    # Ground reflected irradiance
    poa_ground = pvlib.irradiance.get_ground_diffuse(
        surface_tilt=surface_tilt,
        ghi=weather_data['GHI'],
        albedo=0.2
    )
    
    # Total POA irradiance
    total_poa = poa_direct + poa_sky_diffuse + poa_ground
    
    return {
        'poa_direct': poa_direct,
        'poa_sky_diffuse': poa_sky_diffuse,
        'poa_ground': poa_ground,
        'total_poa': total_poa,
        'solar_zenith': solar_position['apparent_zenith'],
        'solar_azimuth': solar_position['azimuth'],
        'surface_tilt': surface_tilt,
        'surface_azimuth': surface_azimuth
    }

def optimize_tilt_azimuth(weather_data, latitude, longitude, tilt_range=(0, 90), azimuth_range=(90, 270)):
    """
    Optimize surface tilt and azimuth angles to maximize annual irradiance
    
    Parameters:
    -----------
    weather_data : pd.DataFrame
        DataFrame containing weather data
    latitude : float
        Geographic latitude
    longitude : float
        Geographic longitude
    tilt_range : tuple
        Range of tilt angles to evaluate (min, max)
    azimuth_range : tuple
        Range of azimuth angles to evaluate (min, max)
        
    Returns:
    --------
    tuple : Optimal tilt, optimal azimuth, and maximum radiation
    """
    max_radiation = 0
    optimal_tilt = 0
    optimal_azimuth = 180  # Default south-facing
    
    for tilt in range(tilt_range[0], tilt_range[1], 5):
        for azimuth in range(azimuth_range[0], azimuth_range[1], 5):
            poa_results = calculate_poa(weather_data, tilt, azimuth, latitude, longitude)
            annual_radiation = poa_results['total_poa'].mean()
            
            if annual_radiation > max_radiation:
                max_radiation = annual_radiation
                optimal_tilt = tilt
                optimal_azimuth = azimuth
    
    return optimal_tilt, optimal_azimuth, max_radiation

# Example usage:
if __name__ == "__main__":
    # Read weather data
    weather_data = pd.read_csv('shenzhen_weather.csv')
    weather_data['datetime'] = pd.to_datetime(weather_data[['Year','Month','Day','Hour','Minute']])
    weather_data.set_index('datetime', inplace=True)
    
    # Shenzhen coordinates
    latitude = 22.54554
    longitude = 114.0683
    
    # Calculate optimal tilt and azimuth angles
    optimal_tilt, optimal_azimuth, max_radiation = optimize_tilt_azimuth(weather_data, latitude, longitude)
    print(f"Optimal tilt angle: {optimal_tilt}°")
    print(f"Optimal azimuth angle: {optimal_azimuth}°")
    print(f"Maximum average irradiance: {max_radiation:.2f} W/m²")
    
    # Calculate POA using optimal angles
    poa_results = calculate_poa(weather_data, optimal_tilt, optimal_azimuth, latitude, longitude)
    
    # Test tracking system
    poa_track_results = calculate_poa_track(weather_data, latitude, longitude)
    print("\nDual-axis tracking system average irradiance:", poa_track_results['total_poa'].mean().round(2), "W/m²")
