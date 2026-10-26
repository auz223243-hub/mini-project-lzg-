import pandas as pd
import pvlib
import numpy as np

def calculate_pv_efficiency_round(poa_data, weather_data, module_parameters, module_area, PV_round=0, Tem_round=1, wind_round=2):
    """
    Calculate PV panel efficiency and power output
    
    Parameters:
    -----------
    poa_data (DataFrame): DataFrame containing POA irradiance data
    weather_data (DataFrame): DataFrame containing temperature and wind speed data
    module_parameters (dict): PV module parameters
    module_area (float): PV module area (square meters)
    PV_round (int): Decimal places to round irradiance values
    Tem_round (int): Decimal places to round temperature values
    wind_round (int): Decimal places to round wind speed values
    
    Returns:
    --------
    DataFrame: DataFrame containing timestamps and calculated efficiency and power values
    """
    # Prepare data, ensuring indices are aligned
    irr = pd.DataFrame({
        'pv_potential': poa_data['total_poa'],
        'Tem': weather_data['Temperature'],
        'Wind': weather_data['Wind Speed']
    }, index=poa_data.index)
    
    # Round values to reduce computational complexity
    irr['pv_potential_round'] = irr['pv_potential'].round(PV_round)
    irr['Tem_round'] = irr['Tem'].round(Tem_round)
    irr['WinSpeed_round'] = irr['Wind'].round(wind_round)
    
    def calculate_pv_performance(irr_data, module_parameters):
        """
        Calculate PV performance metrics based on irradiance and weather data
        
        Parameters:
        -----------
        irr_data (DataFrame): DataFrame with irradiance and weather data
        module_parameters (dict): PV module parameters
        
        Returns:
        --------
        DataFrame: DataFrame with calculated efficiency and power values
        """
        # Define SAPM temperature model parameters for open rack glass/polymer modules
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
        
        # Calculate cell temperature using SAPM model
        cell_temperature = pvlib.temperature.sapm_cell(
            irr_data['pv_potential'], 
            irr_data['Tem'],
            irr_data['Wind'], 
            **temperature_model_parameters
        )
        
        # Calculate PVsyst parameters for the single diode model
        pvsyst_params = pvlib.pvsystem.calcparams_pvsyst(
            effective_irradiance=irr_data['pv_potential'],
            temp_cell=cell_temperature,
            **module_parameters
        )
        
        # Calculate maximum power point using single diode model
        mpp = pvlib.pvsystem.singlediode(
            photocurrent=pvsyst_params[0],
            saturation_current=pvsyst_params[1],
            resistance_series=pvsyst_params[2],
            resistance_shunt=pvsyst_params[3],
            nNsVth=pvsyst_params[4],
            method='lambertw'
        )
        
        # Calculate efficiency (power output / incident irradiance / module area)
        efficiency = mpp['p_mp'] / irr_data['pv_potential'] / module_area
        efficiency = efficiency.where(irr_data['pv_potential'] > 0, 0)  # Set efficiency to 0 when irradiance is 0
        
        return pd.DataFrame({
            'efficiency': efficiency,
            'power': mpp['p_mp']  # Actual power output (W)
        }, index=irr_data.index)
    
    # Method 1: For computational efficiency, calculate results for unique combinations first,
    # then map back to the original time series
    # Get unique combinations of rounded values
    unique_combinations = irr[['pv_potential_round', 'Tem_round', 'WinSpeed_round']].drop_duplicates()
    
    # Create a dictionary to store the index of each unique combination
    unique_indices = {}
    for idx, row in unique_combinations.iterrows():
        key = (row['pv_potential_round'], row['Tem_round'], row['WinSpeed_round'])
        unique_indices[key] = idx
    
    # Create a DataFrame with original values for each unique combination
    unique_data = pd.DataFrame(index=unique_combinations.index)
    unique_data['pv_potential'] = [irr.loc[idx, 'pv_potential'] for idx in unique_data.index]
    unique_data['Tem'] = [irr.loc[idx, 'Tem'] for idx in unique_data.index]
    unique_data['Wind'] = [irr.loc[idx, 'Wind'] for idx in unique_data.index]
    
    # Calculate efficiency and power for unique combinations
    unique_results = calculate_pv_performance(unique_data, module_parameters)
    
    # Create results DataFrame, preserving original time index
    final_results = pd.DataFrame(index=irr.index)
    final_results['pv_potential'] = irr['pv_potential']
    final_results['Tem'] = irr['Tem']
    final_results['Wind'] = irr['Wind']
    
    # Assign efficiency and power values for each timestamp
    efficiency_values = []
    power_values = []
    
    for idx, row in irr.iterrows():
        key = (row['pv_potential_round'], row['Tem_round'], row['WinSpeed_round'])
        unique_idx = unique_indices.get(key)
        if unique_idx is not None:
            efficiency_values.append(unique_results.loc[unique_idx, 'efficiency'])
            power_values.append(unique_results.loc[unique_idx, 'power'])
        else:
            # If no matching combination is found (shouldn't happen), use default values
            efficiency_values.append(0)
            power_values.append(0)
    
    final_results['efficiency'] = efficiency_values
    final_results['power'] = power_values


    final_results.drop(columns=['pv_potential', 'Tem', 'Wind'], inplace=True)
    
    # Add timestamp column (if index is DatetimeIndex)
    if isinstance(final_results.index, pd.DatetimeIndex):
        final_results['timestamp'] = final_results.index
    
    return final_results