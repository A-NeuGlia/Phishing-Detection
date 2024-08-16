#Importation des librairies nécessaires à la création d'un modèle.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy.stats import yeojohnson
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import optuna
import joblib
import warnings
#On vient aussi filtrer les avertissements émis par la librairie 'seaborn_oldcore' pour améliorer la visibilité de nos sorties. 
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn._oldcore')

#Lecture des données.
df = pd.read_csv('Phishing_Legitimate_full.csv')

#Exploration de nos données .
print(f"{df.dtypes}\n") 
#Taille de nos données.
print(f"Dimensions: {df.shape[0]} x {df.shape[1]}\n")
#Classification du types de données.
datatype_counts = df.dtypes.value_counts()
for dtype, count in datatype_counts.items():
    print(f"{dtype}: {count} columns")

#Abandon de la colonnes 'id'
df = df.drop("id", axis=1)

#Vérification de la présence de valeurs manquantes.
null = df.isnull().sum()
for i in range(len(df.columns)):
    print(f"{df.columns[i]}: {null[i]} ({(null[i]/len(df))*100}%)")
total_cellules = np.prod(df.shape)
total_absent = null.sum()
print(f"\nTotal missing values: {total_absent} ({(total_absent/total_cellules) * 100}%)\n")


def is_continuous(series):
    return series.nunique() > 10

colonne_continue = [col for col in df.columns if is_continuous(df[col])]

sns.pairplot(df[colonne_continue], height= 2.5)
plt.show()


#Affichage des corrélations.
corr = df.corr()
cols = corr.nlargest(50, 'CLASS_LABEL')['CLASS_LABEL'].index
cm = np.corrcoef(df[cols].values.T)
sns.set_theme(font_scale=0.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 4}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Vérification de l'étendue de nos données.
colonne_ordinale = [col for col in df.columns if col not in colonne_continue]
sns.set_theme(font_scale=1)
for col in colonne_ordinale:
    plt.hist(df[col], bins=10)  
    plt.xlabel(col)
    plt.ylabel('Frequence')
    plt.title(f'{col}')
    plt.show()
    def normal(mean, std, color="black"):
        x = np.linspace(mean-4*std, mean+4*std, 200)
        p = stats.norm.pdf(x, mean, std)
        z = plt.plot(x, p, color, linewidth=2)

for nom_col in colonne_continue:
    fig1, ax1 = plt.subplots()
    sns.histplot(x=df[nom_col], stat="density", ax=ax1)
    normal(df[nom_col].mean(), df[nom_col].std())
    
    fig2, ax2 = plt.subplots()
    stats.probplot(df[nom_col], plot=ax2)
    
    plt.show()


#Correction de nos valeurs aberrantes.
df = df[df['NumDots'] < 20]
df = df[df['NumDash'] < 40]
plt.scatter(x=df['NumDots'], y=df['NumDash'])
plt.xlabel('NumDots')
plt.ylabel('NumDash')
plt.show()

#Extraction de notre colonne"Class_label".
col = df.columns.to_list()
#Supression de celle-ci.( du data-set)
col.remove('CLASS_LABEL')

#Définition d'une nouvelle base de données.
X = df[col]
y = df["CLASS_LABEL"]

#On sépare notre échantillon.
X_entrainement, X_test, Y_entrainement, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#On définit les paramètres de notre forêt.
def objectif(essaie):
    n_estim = essaie.suggest_int('n_estimations', 10, 300)
    prodondeur_maximum = essaie.suggest_int('prodondeur_maximum', 2, 32, log=True)
    echantillons_min_div = essaie.suggest_int('echantillons_min_div', 2, 20)
    echantillons_min_noeud = essaie.suggest_int('echantillons_min_noeud', 1, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estim,
        max_depth=prodondeur_maximum,
        min_samples_split=echantillons_min_div,
        min_samples_leaf=echantillons_min_noeud,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_entrainement, Y_entrainement)

    y_pred = clf.predict(X_test)

    exactitude = accuracy_score(y_test, y_pred)

    return exactitude
#On crée une étude pour maximiser la valeur de l'exactitude en lançant notre fonction 100 fois.
etude = optuna.create_study(direction='maximize')
etude.optimize(objectif, n_trials=100)
#On extrait alors le meilleur essai nous permettant d'automatiser la recherche des meilleurs paramètres.
meilleur_essaie = etude.best_trial
resultat = meilleur_essaie.params
model = RandomForestClassifier(n_estimators=resultat['n_estimations'], max_depth=resultat['prodondeur_maximum'], min_samples_split=resultat['echantillons_min_div'], min_samples_leaf=resultat['echantillons_min_noeud'], random_state=42, n_jobs=-1)
model.fit(X_entrainement, Y_entrainement)
y_prediction = model.predict(X_test)
print(classification_report(y_test, y_prediction))

    