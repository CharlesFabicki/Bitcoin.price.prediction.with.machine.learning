# Bitcoin Price Prediction with Random Forest

This project predicts Bitcoin prices using a Random Forest regression model based on historical data. 
The Python script employs the scikit-learn library for machine learning, numpy for numerical operations, 
and matplotlib for data visualization.

## Screenshot

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib

Install the dependencies using:
```
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone the repository
2. Run the script
```
python RandomForestRegressor.py
```
The script will train the Random Forest model on historical data, generate predictions for Bitcoin prices 
in 2024, and display the results in the console along with a matplotlib plot.

## Model Training and Evaluation

The script splits the data into training and test sets, standardizes features, creates a Random Forest model, 
and evaluates performance using mean squared error on the test data.

## Results
Predictions for each month in 2024 are displayed in the console, and a plot comparing actual and predicted prices is saved.

## Author
Charles Fabicki

Feel free to modify the code.