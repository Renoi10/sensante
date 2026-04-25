import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ============================================================
# ETAPE 2.1 : Charger le vrai dataset
# ============================================================
df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# ============================================================
# ETAPE 2.2 : Preparer les features et la cible
# ============================================================
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee',
                'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures : {X.shape}")
print(f"Cible : {y.shape}")

# ============================================================
# ETAPE 3.1 : Separer entrainement et test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEntrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# ============================================================
# ETAPE 4.1 : Entrainer le modele
# ============================================================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

# ============================================================
# ETAPE 5.1 : Predire sur les donnees de test
# ============================================================
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(f"\n{comparison}")

# ============================================================
# ETAPE 5.2 : Calculer l'accuracy
# ============================================================
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# ============================================================
# ETAPE 5.3 : Matrice de confusion
# ============================================================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# ============================================================
# ETAPE 5.4 : Visualiser la matrice de confusion
# ============================================================
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

# ============================================================
# ETAPE 6.1 : Serialiser le modele
# ============================================================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
size = os.path.getsize("models/model.pkl")
print(f"\nModele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# ============================================================
# ETAPE 6.2 : Sauvegarder les encodeurs
# ============================================================
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Encodeurs et metadata sauvegardes.")

# ============================================================
# ETAPE 7.1 : Recharger le modele depuis le fichier
# ============================================================
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"\nModele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# ============================================================
# ETAPE 7.2 : Predire pour un nouveau patient
# ============================================================
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': 1,
    'fatigue': 1,
    'maux_tete': 1,
    'frissons': 1,
    'nausee': 0,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    nouveau_patient['toux'],
    nouveau_patient['fatigue'],
    nouveau_patient['maux_tete'],
    nouveau_patient['frissons'],
    nouveau_patient['nausee'],
    region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")
print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:12s} : {proba:.1%} {bar}")