# Building Energy Prediction Hands-On

This project provides a hands-on demonstration of building a machine learning model to predict building energy consumption.

## Steps to Run the Experiment

1.  **Set up the Conda Environment**

    Open your terminal and navigate to the project directory. Create the Conda environment using the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the Environment**

    Activate the newly created environment:

    ```bash
    conda activate building_energy
    ```

3.  **Launch Jupyter Notebook**

    Start Jupyter Notebook from your terminal:

    ```bash
    jupyter notebook
    ```

4.  **Run the Notebook**

    In the Jupyter interface, open the `building_energy_prediction.ipynb` notebook and run the cells sequentially to see the entire workflow from data loading to model interpretation.

## File Descriptions

-   `environment.yml`: Defines the Conda environment with all necessary Python packages.
-   `generate_data.py`: A script to generate the synthetic `building_data.csv` dataset.
-   `building_data.csv`: A synthetic dataset containing hourly energy consumption and weather data for one year.
-   `building_energy_prediction.ipynb`: The main Jupyter Notebook that walks through the entire prediction task.
-   `README.md`: This file, providing setup and usage instructions.