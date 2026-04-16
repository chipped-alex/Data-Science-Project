import pandas as pd

df = pd.read_csv('Amazon_Sale_Report.csv', low_memory=False)

df.columns = df.columns.str.strip() #rimuove tutti gli spazi bianchi all'inizio o alla fine dei nomi delle colonne

#MISSING/NULL VALUES MANAGEMENT
df['promotion-ids'] = df['promotion-ids'].fillna('No promotion')
df['Courier Status'] = df['Courier Status'].fillna('Cancelled')
df['fulfilled-by'] = df['fulfilled-by'].fillna('FBA')

#UPPER & LOWERCASE
df['ship-state'] = df['ship-state'].astype(str).str.title()
df['Category'] = df['Category'].astype(str).str.title()

#SHIP-STATE MANAGEMENT
replacements = {
    "Ar": "Arunachal Pradesh",
    "Nl": "Nagaland",
    "Pb": "Punjab",
    "Pondicherry": "Puducherry",
    "Punjab/Mohali/Zirakpur": "Punjab",
    "Rajshthan": "Rajasthan",
    "Rajsthan": "Rajasthan",
    "Rj": "Rajasthan",
    "Nan": "Unknown"
}

df['ship-state'] = df['ship-state'].replace(replacements)

#PROMOTION MANAGEMENT
df['promotion_code'] = df['promotion-ids'].str.split(',').str[0] #prende la prima parte prima della virgola
df['promotion_code'] = df['promotion_code'].str.replace(r'-\d+\s', '-', regex=True) #elimina il codice numerico tra '-' e 'coupon'

#DATA E AMOUNT
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')

df['Amount_new'] = df['Amount']
df.loc[(df['Qty'] == 0) & (df['Status'] == 'Cancelled'), 'Amount_new'] = 0

df = df.dropna(subset=['Amount_new'])
df['Amount_new'] = df['Amount_new'].astype(int)

#FROM COLUMN OF STRING TO BOOLEAN COLUMNS
df['Expedited Shipment'] = df['ship-service-level'].replace({'Expedited':True, 'Standard':False})
df['Amazon Fulfilment'] = df['Fulfilment'].replace({'Amazon':True, 'Merchant':False})
df['Easy Ship'] = df['fulfilled-by'].replace({'Easy Ship':True, 'FBA':False})

#FINAL DATAFRAME
final_columns = ['Order ID', 'Date', 'Style', 'SKU', 'ASIN', 'Category', 'Size', 'Qty',
    'Amount_new', 'ship-state', 'Status', 'Courier Status', 'Expedited Shipment',
    'promotion_code', 'Amazon Fulfilment', 'Easy Ship', 'B2B']

df = df[final_columns]

#df.to_csv('data.csv', index=False)