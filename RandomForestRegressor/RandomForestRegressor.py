import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Bitcoin price data
data = {
    'Date': [
        '11/01/2023', '10/01/2023', '09/01/2023', '08/01/2023', '07/01/2023',
        '06/01/2023', '05/01/2023', '04/01/2023', '03/01/2023', '02/01/2023',
        '01/01/2023', '12/01/2022', '11/01/2022', '10/01/2022', '09/01/2022',
        '08/01/2022', '07/01/2022', '06/01/2022', '05/01/2022', '04/01/2022',
        '03/01/2022', '02/01/2022', '01/01/2022', '12/01/2021', '11/01/2021',
        '10/01/2021', '09/01/2021', '08/01/2021', '07/01/2021', '06/01/2021',
        '05/01/2021', '04/01/2021', '03/01/2021', '02/01/2021', '01/01/2021',
        '12/01/2020', '11/01/2020', '10/01/2020', '09/01/2020', '08/01/2020',
        '07/01/2020', '06/01/2020', '05/01/2020', '04/01/2020', '03/01/2020',
        '02/01/2020', '01/01/2020', '12/01/2019', '11/01/2019', '10/01/2019',
        '09/01/2019', '08/01/2019', '07/01/2019', '06/01/2019', '05/01/2019',
        '04/01/2019', '03/01/2019', '02/01/2019', '01/01/2019', '12/01/2018',
        '11/01/2018', '10/01/2018', '09/01/2018', '08/01/2018', '07/01/2018',
        '06/01/2018', '05/01/2018', '04/01/2018', '03/01/2018', '02/01/2018',
        '01/01/2018', '12/01/2017', '11/01/2017', '10/01/2017', '09/01/2017',
        '08/01/2017', '07/01/2017', '06/01/2017', '05/01/2017', '04/01/2017',
        '03/01/2017', '02/01/2017', '01/01/2017', '12/01/2016', '11/01/2016',
        '10/01/2016', '09/01/2016', '08/01/2016', '07/01/2016', '06/01/2016',
        '05/01/2016', '04/01/2016', '03/01/2016', '02/01/2016', '01/01/2016',
        '12/01/2015', '11/01/2015', '10/01/2015', '09/01/2015', '08/01/2015',
        '07/01/2015', '06/01/2015', '05/01/2015', '04/01/2015', '03/01/2015',
        '02/01/2015', '01/01/2015', '12/01/2014', '11/01/2014', '10/01/2014',
        '09/01/2014', '08/01/2014', '07/01/2014', '06/01/2014', '05/01/2014',
        '04/01/2014', '03/01/2014', '02/01/2014', '01/01/2014', '12/01/2013',
        '11/01/2013', '10/01/2013', '09/01/2013', '08/01/2013', '07/01/2013',
        '06/01/2013', '05/01/2013', '04/01/2013', '03/01/2013', '02/01/2013',
        '01/01/2013'
    ],
    'Price': [
        '35,454.40', '34,650.60', '26,962.70', '25,937.30', '29,232.40', '30,472.90', '27,216.10',
        '29,252.10', '28,473.70', '23,130.50', '23,125.10', '16,537.40', '17,163.90', '20,496.30',
        '19,423.00', '20,043.90', '23,303.40', '19,926.60', '31,793.40', '37,650.00', '45,525.00',
        '43,188.20', '38,498.60', '46,219.50', '56,882.90', '61,309.60', '43,823.30', '47,130.40',
        '41,553.70', '35,026.90', '37,298.60', '57,720.30', '58,763.70', '45,164.00', '33,108.10',
        '28,949.40', '19,698.10', '13,797.30', '10,776.10', '11,644.20', '11,333.40', '9,135.40',
        '9,454.80', '8,629.00', '6,412.50', '8,543.70', '9,349.10', '7,196.40', '7,546.60', '9,152.60',
        '8,284.30', '9,594.40', '10,082.00', '10,818.60', '8,558.30', '5,320.80', '4,102.30', '3,816.60',
        '3,437.20', '3,709.40', '4,039.70', '6,365.90', '6,635.20', '7,033.80', '7,729.40', '6,398.90',
        '7,502.60', '9,245.10', '6,938.20', '10,333.90', '10,265.40', '13,850.40', '9,946.80', '6,451.20',
        '4,360.60', '4,735.10', '2,883.30', '2,480.60', '2,303.30', '1,351.90', '1,079.10', '1,189.30',
        '965.5', '963.4', '742.5', '698.7', '608.1', '573.9', '621.9', '670', '528.9', '448.5', '415.7',
        '436.2', '369.8', '430', '378', '311.2', '235.9', '229.5', '283.7', '264.1', '229.8', '235.8',
        '244.1', '254.1', '218.5', '318.2', '374.9', '337.9', '388.2', '481.8', '589.5', '635.1',
        '627.9', '445.6', '444.7', '573.9', '938.8', '805.9', '1,205.70', '211.2', '141.9', '141',
        '106.2', '97.5', '128.8', '139.2', '93', '33.4', '20.4'
    ]
}
# Creating a DataFrame with data
df = pd.DataFrame(data)

# Converting the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Setting the date as the index
df.set_index('Date', inplace=True)

# Extracting data from 2013 onwards
df = df.loc['01/01/2013':]

# Removing commas and spaces from the 'Price' column and converting to float
df['Price'] = df['Price'].str.replace(',', '').str.replace(' ', '').astype(float)

# Creating a new column with the day of the year
df['Day'] = df.index.dayofyear

# Creating a new column with the year
df['Year'] = df.index.year

# Creating a new column with the month
df['Month'] = df.index.month

# Shifting the 'Price' values down by one month to create the 'Prediction' column
df['Prediction'] = df['Price'].shift(-1)

# Preparing data for the model
X = df[['Day', 'Year', 'Month']]
y = df['Prediction']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating and training a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predicting for the test data
y_pred = model.predict(X_test)

# Handling missing values in y_test and y_pred by replacing them with the mean price
mean_price = df['Price'].mean()
y_test[np.isnan(y_test)] = mean_price
y_pred[np.isnan(y_pred)] = mean_price

# Calculating the mean squared error on the test data
mse_test = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on test data: {mse_test:.2f}')

# Predicting Bitcoin prices in 2024
predictions_2024 = []

# Starting from January 2024
for month in range(1, 13):
    prediction_date = datetime(2024, month, 1)
    prediction_day = prediction_date.timetuple().tm_yday
    prediction_year = prediction_date.year
    prediction_month = prediction_date.month

    prediction_data = np.array([[prediction_day, prediction_year, prediction_month]])
    prediction_data = scaler.transform(prediction_data)
    prediction = model.predict(prediction_data)
    predictions_2024.append((prediction_date, prediction[0]))

# Displaying the predicted Bitcoin prices for each month in 2024
for date, price in predictions_2024:
    print(f"Predicted Bitcoin price on {date.strftime('%Y/%m/%d')} is: {price:.2f} USD")

# Extracting actual Bitcoin prices and dates
actual_prices = df['Price']
dates_actual = df.index

# Extracting dates for predicted prices in 2024
dates_2024 = [date for date, _ in predictions_2024]

# Creating a line plot
plt.figure(figsize=(12, 6))
plt.plot(dates_actual, actual_prices, label='Actual Prices', color='blue')
predicted_prices = [price for _, price in predictions_2024]
plt.plot(dates_2024, predicted_prices, label='Predicted Prices (2024)', color='red')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Bitcoin Price (USD)')
plt.title('Actual vs. Predicted Bitcoin Prices')

# Adding a legend
plt.legend()

# Rotating x-axis labels for readability
plt.xticks(rotation=45)

# Setting the x-axis range from 2013 to 2025
start_date = datetime(2013, 1, 1)
end_date = datetime(2025, 1, 1)
plt.xlim(start_date, end_date)

# Adding vertical gridlines for each year
years = mdates.YearLocator()
ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Adding a grid to the plot
ax.grid(which='both', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator(base=1))  # Step every year
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))  # Step every month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the year on the x-axis

# Scaling the y-axis range to 70,000 with increments of 5,000
plt.ylim(0, 70000)
plt.yticks(range(0, 70001, 5000))

# Displaying the plot
plt.tight_layout()
plt.show()
