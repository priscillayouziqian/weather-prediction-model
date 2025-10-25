# 🌤️ Weather Prediction Project

This project uses historical weather data to train a machine learning model (Ridge Regression) to predict the next day's maximum temperature. It includes a Jupyter notebook detailing the data science workflow and a Streamlit web application for interactive predictions.

## 📂 Project Components

- **`predicting_weather_JOS.ipynb`**: The main Jupyter notebook that covers the end-to-end machine learning process, from data cleaning and feature engineering to model training and evaluation.
- **`weather.py`**
- **`local_weather.csv`**
- **`weather_prediction_model.joblib`**: The serialized, trained Ridge regression model, ready for use in the Streamlit application.

##  workflow-summary Workflow Summary

The machine learning model was developed following these key steps in the `predicting_weather_JOS.ipynb` notebook:

1.  **Data Cleaning**: The raw data was loaded, and columns with a high percentage of missing values were dropped. The remaining missing values in the core features (`PRCP`, `TMAX`, `TMIN`) were imputed using the forward-fill (`ffill`) method, which is suitable for time-series data.

2.  **Feature Engineering**: To improve the model's predictive power, several new features were created from the existing data. This included:
    - A `target` column representing the next day's maximum temperature.
    - Rolling 30-day averages to capture recent trends.
    - Ratio features like `month_day_max` and `max_min` to provide context.
    - Historical monthly and daily averages to account for seasonality.

3.  **Model Training**: A **Ridge Regression** model was chosen. This model is effective for datasets where features might be correlated (like `temp_max` and `temp_min`) and helps prevent overfitting. The model was trained on data from 1960 to early 2021.

4.  **Evaluation**: The model's performance was evaluated on a test set (data from 2021 onwards) using the **Mean Absolute Error (MAE)** metric, which showed the model's predictions were, on average, about 3.3°F off from the actual temperature.

## 🚀 Getting Started

### Dependencies

This project requires Python and the following libraries. You can install them using pip.

```
streamlit>=1.33.0
pandas>=2.0.0
joblib>=1.3.0
scikit-learn>=1.3.0
```

You can also run:
```bash
pip install -r requirements.txt
```

### How to Run the Application

The Streamlit app allows you to get predictions from the trained model through a simple web interface.

1.  **Navigate to the project directory** in your terminal:

2.  **Run the Streamlit app**:
    ```bash
    streamlit run weather.py
    ```


### Exploring the Analysis

To understand how the model was built, you can explore the Jupyter notebook:

1.  Ensure you have Jupyter Notebook or JupyterLab installed (`pip install notebook`).
2.  Run the following command in your terminal:
    ```bash
    jupyter notebook predicting_weather_JOS.ipynb
    ```

This will allow you to see the code, visualizations, and step-by-step logic behind the project.

## 🙏 Acknowledgements

A special thanks to my tutor, Charles-Owolabi, for his reference on the project.