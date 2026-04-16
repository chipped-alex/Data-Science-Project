import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('amazon_data_set.csv', parse_dates=['Date'])
sns.set_theme(style="whitegrid")

#1: ANALISI TEMPORALE
plt.figure(figsize=(12, 5))
vendite_settimanali = df.groupby(pd.Grouper(key='Date', freq='W'))['Amount_new'].sum().reset_index()
sns.lineplot(data=vendite_settimanali, x='Date', y='Amount_new', color='royalblue', linewidth=2.5, marker='o')
plt.title('1. Andamento dei Ricavi (Trend Settimanale)', fontsize=15, fontweight='bold')
plt.xlabel('Data (Settimane)', fontsize=12)
plt.ylabel('Ricavo Totale (INR)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('01_Trend_Temporale.png', dpi=300)


#2: ANALISI DI PRODOTTO E PREZZO
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ricavi_cat = df.groupby('Category')['Amount_new'].sum().reset_index().sort_values('Amount_new', ascending=False)
qty_cat = df.groupby('Category')['Qty'].sum().reset_index().sort_values('Qty', ascending=False)
sns.barplot(data=ricavi_cat, x='Category', y='Amount_new', ax=axes[0], hue='Category', legend=False, palette='viridis')
axes[0].set_title('Ricavo Totale per Categoria', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Categoria', fontsize=12)
axes[0].set_ylabel('Ricavo Totale (INR)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(data=qty_cat, x='Category', y='Qty', ax=axes[1], hue='Category', legend=False, palette='viridis')
axes[1].set_title('Quantità Totale Venduta per Categoria', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Categoria', fontsize=12)
axes[1].set_ylabel('Numero di Pezzi Venduti', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
fig.suptitle('Performance delle Categorie: Ricavi a confronto con i Volumi', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_Subplots_Ricavi_Quantita.png', dpi=300)

#Calcolo del rapporto per ogni categoria
prezzo_medio_cat = df.groupby('Category').agg({
    'Amount_new': 'sum',
    'Qty': 'sum'
}).reset_index()
prezzo_medio_cat['PM'] = prezzo_medio_cat['Amount_new'] / prezzo_medio_cat['Qty'].replace(0, 1)  #Normalizzazione: Ricavi / Quantità
prezzo_medio_cat = prezzo_medio_cat.sort_values(by='PM', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(data=prezzo_medio_cat, x='Category', y='PM', hue='Category', palette='coolwarm', legend=False)
for i, v in enumerate(prezzo_medio_cat['PM']):
    plt.text(i, v + 10, f"{int(v)}", ha='center', fontweight='bold')
plt.title('Prezzo Medio per Singolo Articolo', fontsize=15, fontweight='bold')
plt.xlabel('Categoria Prodotto')
plt.ylabel('Prezzo Medio Unitario (INR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('03_Prezzo_medio_Normalizzato.png', dpi=300)

#Volume Ordini (con promozione e senza promozione)
#Creiamo una colonna per raggruppare le due categorie
df['Stato_Promo'] = df['promotion_code'].apply(lambda x: 'Senza Promozione' if x == 'No promotion' else 'Con Promozione')
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Stato_Promo', hue='Stato_Promo', palette='pastel', legend=False)
plt.title('Impatto delle Promozioni sul Volume degli Ordini', fontsize=14, fontweight='bold')
plt.xlabel('Utilizzo Promozione')
plt.ylabel('Numero di Ordini')
plt.tight_layout()
plt.savefig('04_Volume_Promozioni.png', dpi=300)


#3: COMPORTAMENTO DEI CLIENTI
df['B2B_Label'] = df['B2B'].map({True: 'Azienda (B2B)', False: 'Consumatore (B2C)'}) #trasformiamo True/False in azienda/consumatore
plt.figure(figsize=(8, 8))
b2b_counts = df['B2B_Label'].value_counts() #Calcoliamo le frequenze
plt.pie(b2b_counts, labels=b2b_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Pastel2'))
plt.title('Percentuale di Ordini B2B sul Totale', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_Percentuale_B2B.png', dpi=300)

#Spesa Media per Ordine (Scontrino Medio) B2B vs B2C
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='B2B_Label', y='Amount_new', hue='B2B_Label', legend=False, palette='Pastel2')
plt.title('Scontrino Medio: B2B vs B2C', fontsize=14, fontweight='bold')
plt.xlabel('Tipo di Cliente')
plt.ylabel('Spesa Media per Ordine (INR)')
plt.tight_layout()
plt.savefig('06_Scontrino_Medio_B2B_B2C.png', dpi=300)


#4: LOGISTICA E GEOGRAFIA
g_logistica = sns.catplot(data=df, kind="count", x="Fulfilment", hue="ship-service-level", palette="Set1", height=5, aspect=1.2)
g_logistica.fig.suptitle('Efficienza Logistica: Gestore vs Velocità Spedizione', y=1.05, fontsize=15, fontweight='bold')
g_logistica.set_axis_labels("Gestore dell'Ordine (Fulfilment)", "Numero di Ordini")
plt.savefig('07_Logistica_Catplot.png', bbox_inches='tight', dpi=300)

plt.figure(figsize=(10, 6))
top_stati = df.groupby('ship-state')['Amount_new'].sum().nlargest(10).reset_index() #raggruppiamo per Stato e prendiamo i primi 10
sns.barplot(data=top_stati, y='ship-state', x='Amount_new', palette='mako', hue='ship-state', legend=False)
plt.title('Top 10 Stati per Fatturato Generato', fontsize=15, fontweight='bold')
plt.xlabel('Ricavo Totale (INR)')
plt.ylabel('Stato di Destinazione')
plt.tight_layout()
plt.savefig('08_Top10_Stati.png', dpi=300)