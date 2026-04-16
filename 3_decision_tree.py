import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('model_data.csv')

#Variabile target
colonne_status = [col for col in df.columns if 'Status_' in col]
df['ordine_canc'] = (df[colonne_status].sum(axis=1) == 0).astype(int)

#Selezione dei predittori
features = [col for col in df.columns if col.startswith(('Category_', 'promotion_code_', 'B2B',
                                                        'Expedited Shipment', 'Amazon Fulfilment', 'Easy Ship'))]
X = df[features]
y = df['ordine_canc']

#Train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#DECISION TREE
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Cross-validation
acc_cv = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Accuratezza media in Cross-Validation: {acc_cv.mean():.4f}")

#Valutazione delle performance
print(f"Accuratezza sul Test Set: {accuracy_score(y_test, y_pred):.4f}")
p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, average=None)

print("CLASSE 0 (Consegnati/In Transito):")
print(f"  Precision: {p[0]:.4f}")
print(f"  Recall:    {r[0]:.4f}")
print(f"  F1-Score:  {f1[0]:.4f}")
print(f"  Supporto:  {s[0]} ordini reali")

print("\nCLASSE 1 (Cancellati):")
print(f"  Precision: {p[1]:.4f}")
print(f"  Recall:    {r[1]:.4f}")
print(f"  F1-Score:  {f1[1]:.4f}")
print(f"  Supporto:  {s[1]} ordini reali")

#Matrice di confusione
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Previsto: Consegnato', 'Previsto: Cancellato'],
            yticklabels=['Reale: Consegnato', 'Reale: Cancellato'])

plt.title('Matrice di Confusione: Predizione Cancellazioni', fontsize=14, fontweight='bold')
plt.ylabel('Valore Reale')
plt.xlabel('Valore Predetto')
plt.tight_layout()
plt.savefig('Matrice_Confusione_Decision_Tree.png', dpi=300)