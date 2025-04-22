import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.over_sampling import SMOTE


# Cargamos los datos y reducimos su tamaño a 50 000 registros
df_completo = pd.read_csv('card_transdata.csv')
df = df_completo.iloc[:50000]


# Hacemos que se muestren todas las columnas
pd.set_option('display.max_columns', None)

# Nos familiarizamos con los datos
print("VALORES DE LAS COLUMNAS\n")
print(df.columns.values)
print("\n")
print("PRIMER REGISTRO\n")
print(df.head(1))
print("\n")
print("PRIMEROS REGISTROS\n")
print(df.head())
print("\n")
print("ÚLTIMOS REGISTROS\n")
print(df.tail())
print("\n")
print("INFORMACIÓN GENERAL\n")
print(df.info())
print("\n")
print("DESCRIPCIÓN DE LOS DATOS\n")
print(df.describe())
print("\n")
print("COMPROBACIÓN DE DUPLICADOS\n")
duplicados = df.duplicated().sum()
print("Número de duplicados: ", duplicados)

# Cuántas transacciones son fraudulentas y cuántas no
fraud_trans = (df['fraud'] == 1).sum()
no_fraud_trans = (df['fraud'] == 0).sum()

print("\nRECUENTO TRANSACCIONES FRAUDULENTAS/NO FRAUDULENTAS")
print("Fraudulentas: ",fraud_trans)
print("No fraudulentas: ",no_fraud_trans)

# Describe de ratio_to_median_purchase_price
print("\nDESCRIBE DE ratio_to_median_purchase_price AGRUPANDO POR FRAUDE")
print(df.groupby("fraud")['ratio_to_median_purchase_price'].describe())

print("\nANÁLISIS CON FUNCIONES PIVOTANTES\n")
home_distance = df[['distance_from_home', 'fraud']].groupby(['distance_from_home'],
                                                            as_index=False).mean().sort_values(by='fraud',ascending=False)
print(home_distance.head(20))
print("")
print(home_distance.tail(20))
print("")

ratio_median = df[['ratio_to_median_purchase_price', 'fraud']].groupby(['ratio_to_median_purchase_price'],
                                                                       as_index=False).mean().sort_values(by='fraud',
                                                                                                    ascending=False)
print(ratio_median.head(20))
print("")
print(ratio_median.tail(20))
print("")


retailer = df[['repeat_retailer', 'fraud']].groupby(['repeat_retailer'],
                                                      as_index=False).mean().sort_values(by='fraud',ascending=False)
print(retailer.head())
print("")

pin_number = df[['used_pin_number', 'fraud']].groupby(['used_pin_number'],
                                                      as_index=False).mean().sort_values(by='fraud',ascending=False)
print(pin_number.head())
print("")

online = df[['online_order', 'fraud']].groupby(['online_order'],
                                                      as_index=False).mean().sort_values(by='fraud',ascending=False)
print(online.head())
print("")
chip = df[['used_chip', 'fraud']].groupby(['used_chip'],
                                                      as_index=False).mean().sort_values(by='fraud',ascending=False)
print(chip.head())
print("")

# Usamos gráficas para visualizar los datos
print("\nVISUALIZACIÓN DE LOS DATOS\n")

plt.figure(figsize=(10,6))
corr_heatmap = df.corr(numeric_only=True)
sns.heatmap(corr_heatmap, cmap="YlGnBu", annot=True)
plt.tight_layout()
plt.show()

sns.kdeplot(df[df['fraud'] == 0]['ratio_to_median_purchase_price'], label="Fraude = 0")
sns.kdeplot(df[df['fraud'] == 1]['ratio_to_median_purchase_price'], label="Fraude = 1")
plt.xlabel("Ratio mediana compras anteriores")
plt.ylabel("Puntuación")
plt.legend()
plt.show()

online_graph = sns.FacetGrid(df, col='fraud')
online_graph.map(plt.hist, 'online_order')
plt.show()

pin_graph = sns.FacetGrid(df, col='fraud')
pin_graph.map(plt.hist, 'used_pin_number')
plt.show()

home_distance_graph = sns.FacetGrid(df, col='fraud')
home_distance_graph.map(plt.hist, 'distance_from_home')
plt.show()

chip_graph = sns.FacetGrid(df, col='fraud')
chip_graph.map(plt.hist, 'used_chip')
plt.show()

#Uso de boxplot para detectar outliers
plt.figure(figsize=(14,9))
df.boxplot()
plt.title('Boxplot')
plt.tight_layout()
plt.show()


# Creación de los modelos
# División de los ejes
x = df[['distance_from_home', 'distance_from_last_transaction',
                'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip', 'used_pin_number',
                'online_order']]

y = df['fraud']

# División train-test en 80%-20%
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size= 0.2,
                                                    random_state=20)

# División en train y validation
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=20)

# Convertimos y_test en un array para trabajar con él
y_test_array = y_test.to_numpy()

# LINEAR SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred_svc = linear_svc.predict(X_test)
linear_svc_results = pd.DataFrame({
    'Reales': y_test_array,
    'Predichos': y_pred_svc
})
print("\nLINEAR SVC")
print(linear_svc_results.head(20))
print(linear_svc_results.tail(20))

svc_confusion_mat = confusion_matrix(y_test, y_pred_svc)
print("\nMATRIZ DE CONFUSIÓN LINEAR SVC ")
print(svc_confusion_mat)

svc_report = classification_report(y_test, y_pred_svc)
print("\nCLASSIFICATION REPORT LINEAR SVC")
print(svc_report)
print("")

f1_svc = f1_score(y_test, y_pred_svc)
svc_metric = round(f1_svc * 100, 2)
print(f"Métrica final f1: {svc_metric}")


# SGD CLASSIFIER
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
sgd_results = pd.DataFrame({
    'Reales': y_test_array,
    'Predichos': y_pred_sgd
})

print("\nSGD CLASSIFIER")
print(sgd_results.head(20))
print(sgd_results.tail(20))

sgd_confusion_mat = confusion_matrix(y_test, y_pred_sgd)
print("\nMATRIZ DE CONFUSIÓN SGD CLASSIFIER")
print(sgd_confusion_mat)

sgd_report = classification_report(y_test, y_pred_sgd)
print("\nCLASSIFICATION REPORT SGD CLASSIFIER")
print(sgd_report)
print("")

f1_sgd = f1_score(y_test, y_pred_sgd)
sgd_metric = round(f1_sgd * 100, 2)
print(f"Métrica final f1: {sgd_metric}")

# RANDOM FOREST
random_forest =RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)

random_forest_results =  pd.DataFrame({
    'Reales': y_test_array,
    'Predichos': y_pred_random_forest
})
print("\nRANDOM FOREST")
print(random_forest_results.head(20))
print(random_forest_results.tail(20))

random_forest_confusion_mat = confusion_matrix(y_test, y_pred_random_forest)
print("\nMATRIZ DE CONFUSIÓN RANDOM FOREST")
print(random_forest_confusion_mat)
print("")

random_forest_report = classification_report(y_test, y_pred_random_forest)
print("CLASSIFICATION REPORT RANDOM FOREST")
print(random_forest_report)
print("")

f1_random = f1_score(y_test, y_pred_random_forest)
random_metric = round(f1_random * 100, 2)
print(f"Métrica final f1: {random_metric}")

# Feature importances
feature_importances_forest = pd.Series(random_forest.feature_importances_, index=X_train.columns)
feature_importances_forest = feature_importances_forest.sort_values(ascending=False)
fig, ax = plt.subplots()
feature_importances_forest.plot.bar()
ax.set_title("Gráfica feature importances Random Forest")
ax.set_ylabel("Puntuación relativa")
ax.set_xlabel("Características")
fig.tight_layout()
plt.show()

# LOGISTIC REGRESSION
print("\nLOGISTIC REGRESSION")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
logreg_results = pd.DataFrame({
    'Reales': y_test_array,
    'Predichos': y_pred_logreg
})

print(logreg_results.head(20))
print(logreg_results.tail(20))

logreg_confusion_mat = confusion_matrix(y_test, y_pred_logreg)
print("\nMATRIZ DE CONFUSIÓN LOGISTIC REGRESSION")
print(logreg_confusion_mat)
print("")

logreg_report = classification_report(y_test, y_pred_logreg)
print("CLASSIFICATION REPORT LOGISTIC REGRESSION")
print(logreg_report)
print("")

f1_logreg = f1_score(y_test, y_pred_logreg)
logreg_metric = round(f1_logreg * 100, 2)
print(f"Métrica final f1: {logreg_metric}\n")

# K-NEAREST NEIGHBORS
print("KNN\n")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_results = pd.DataFrame({
    'Reales': y_test_array,
    'Predichos': y_pred_knn
})
print(knn_results.head(20))
print(knn_results.tail(20))
print("")

knn_confusion_mat = confusion_matrix(y_test, y_pred_knn)
print("\nMATRIZ DE CONFUSIÓN KNN")
print(knn_confusion_mat)
print("")

knn_report = classification_report(y_test, y_pred_knn)
print("\nCLASSIFICATION REPORT KNN")
print(knn_report)
print("")

f1_knn = f1_score(y_test, y_pred_knn)
knn_metric = round(f1_knn * 100, 2)
print(f"Métrica final f1: {knn_metric}")

# DECISION TREE
print("\n DECISION TREE\n")
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
decision_tree_results = pd.DataFrame({
    'Reales': y_test_array,
    'Predichos': y_pred_decision_tree
})

print(decision_tree_results.head(20))
print(decision_tree_results.tail(20))

decision_tree_confusion_mat = confusion_matrix(y_test, y_pred_decision_tree)
print("\nMATRIZ DE CONFUSIÓN DECISION TREE")
print(decision_tree_confusion_mat)
print("")

decision_tree_report = classification_report(y_test, y_pred_decision_tree)
print("\nCLASSIFICATION REPORT DECISION TREE")
print(decision_tree_report)
print("")

f1_decision_tree = f1_score(y_test, y_pred_decision_tree)
decision_tree_metric = round(f1_decision_tree * 100, 2)
print(f"Métrica final f1: {decision_tree_metric}\n")

# Feature importance plot
feature_importances_decision = pd.Series(decision_tree.feature_importances_, index=X_train.columns)
feature_importances_decision = feature_importances_decision.sort_values(ascending=False)
fig, ax = plt.subplots()
feature_importances_decision.plot.bar()
ax.set_title("Gráfica feature importances Decision Tree")
ax.set_ylabel("Puntuación relativa")
ax.set_xlabel("Características")
fig.tight_layout()
plt.show()

# Métricas comparativas de todos los modelos y gráfica
lista_nombres_modelos = ['Random Forest','Decision Tree','KNN', 'Logistic Regression', 'Linear SVC', 'SGD Classifier']
lista_metricas_modelos = [random_metric, decision_tree_metric, knn_metric, logreg_metric, svc_metric, sgd_metric]

plt.rcParams['figure.figsize'] = 16,8
sns.set_style('whitegrid')
ax = sns.barplot(x=lista_nombres_modelos, y=lista_metricas_modelos, palette="husl")
plt.xlabel('Modelos', fontsize = 20)
plt.ylabel('Porcentaje de precisión', fontsize = 20)
plt.title('Comparativa de los modelos')
plt.show()

#Probamos con una Neural Network
start_time = time.time()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

r = model.fit(X_resampled, y_resampled, epochs=30, batch_size=32, validation_data=(X_val, y_val))

loss_hist = r.history['val_loss']
acc_hist = r.history['val_accuracy']
print("val_loss ->", len(loss_hist), "min:", min(loss_hist), "max:", max(loss_hist))
print("acc_hist ->", len(acc_hist), "min:", min(acc_hist), "max:", max(acc_hist))

evaluation = model.evaluate(X_test, y_test)
print(f"\nAccuracy del test: {evaluation[1] * 100:.2f}%")

end_time = time.time()
total_time = end_time - start_time
print(f"Tiempo transcurrido en la ejecución de la NN: {total_time:.2f}s")

# Matriz de confusión de la NN
y_pred_nn = model.predict(X_test)
y_pred_nn = np.argmax(y_pred_nn, axis=1)

# Classification report NN
print("\nCLASSIFIACTION REPORT NEURAL NETWORK")
print(classification_report(y_test, y_pred_nn, zero_division=0))

cm = confusion_matrix(y_test, y_pred_nn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(10,10))
disp =  disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')
plt.show()