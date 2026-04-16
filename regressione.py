import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

df = pd.read_csv('data.csv')

df_encoded = df[(df['Status'] != 'Cancelled') & (df['Qty'] != 0)]
df_encoded = pd.get_dummies(df_encoded, columns = ['Category','promotion_code'], drop_first = True)
df_encoded = df_encoded[df_encoded['Amount_new'] != 0]
encoded_columns = [col for col in df_encoded.columns if col.startswith(('Category_', 'promotion_code_','Qty'))]

X = df_encoded[encoded_columns]
y = df_encoded['Amount_new']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, shuffle=True, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

#Coefficienti
print(f'\nIntercetta : {model.intercept_}')
print(f'Coefficienti : {model.coef_}')

#Valutazione
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2:.2f}')
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Mean absolute percentage error: {mape:.2f}')

amount_pred = model.predict(X_train)

