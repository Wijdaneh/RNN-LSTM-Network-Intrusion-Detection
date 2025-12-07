# train_unsw_pro_final_fixed.py - Version corrigée pour UNSW-NB15
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Scikit-learn / metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

# Utilities
import json
import joblib  # pour sauvegarder le scaler

print("="*80)
print("🚀 PROJET RNN/LSTM - DÉTECTION D'INTRUSION UNSW-NB15")
print("="*80)
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ Pandas version: {pd.__version__}")

# -----------------------------------------------------------------------------
# Paths et vérifications
# -----------------------------------------------------------------------------
training_file = "CSV Files/UNSW_NB15_training-set.csv"
testing_file = "CSV Files/UNSW_NB15_testing-set.csv"

if not os.path.exists(training_file):
    raise SystemExit(f"❌ Fichier d'entraînement non trouvé: {training_file}")

has_test = os.path.exists(testing_file)
if not has_test:
    print(f"⚠️  Fichier de test non trouvé: {testing_file} — on utilisera seulement le train pour split.")

# Créer dossiers de sortie
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# -----------------------------------------------------------------------------
# Chargement sécurisé (encodages)
# -----------------------------------------------------------------------------
def load_unsw_data(filepath, sample_size=None):
    encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']
    for enc in encodings:
        try:
            if sample_size:
                df = pd.read_csv(filepath, encoding=enc, nrows=sample_size, low_memory=False)
            else:
                df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"✅ Chargé {os.path.basename(filepath)} avec encodage: {enc}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Erreur lecture {enc}: {e}")
            continue
    # fallback
    df = pd.read_csv(filepath, encoding='latin-1', errors='ignore', low_memory=False)
    print(f"⚠️ Chargé en fallback latin-1 (errors='ignore')")
    return df

df_train = load_unsw_data(training_file, sample_size=70000)
print(f"✅ Données d'entraînement: {len(df_train):,} échantillons")

df_test = None
if has_test:
    df_test = load_unsw_data(testing_file, sample_size=30000)
    print(f"✅ Données de test: {len(df_test):,} échantillons")

# -----------------------------------------------------------------------------
# Known columns (ton mapping) — on l'affiche mais on ne l'impose pas
# -----------------------------------------------------------------------------
# (je garde ton dictionnaire mais ne l'utilise pas strictement)
unsw_known_columns = {
    # ... (tu peux garder le mapping complet si tu veux)
}

# -----------------------------------------------------------------------------
# Trouver colonne label
# -----------------------------------------------------------------------------
label_col = None
possible_label_cols = []
for col in df_train.columns:
    if 'label' in col.lower():
        possible_label_cols.append(col)
        if pd.api.types.is_numeric_dtype(df_train[col]):
            uniq = np.unique(df_train[col].dropna())
            if len(uniq) <= 2:
                label_col = col
                print(f"✅ Label numérique trouvé: '{col}'")
                break

if not label_col and 'label' in df_train.columns:
    label_col = 'label'
    print("📝 Utilisation de la colonne 'label'")

if not label_col and 'attack_cat' in df_train.columns:
    label_col = 'attack_cat'
    print("⚠️  'label' non trouvé, utilisation de 'attack_cat'")

if not label_col:
    raise SystemExit("❌ Aucune colonne label trouvée. Abandon.")

print(f"📊 Label utilisé: {label_col} ({df_train[label_col].dtype}), valeurs uniques: {df_train[label_col].nunique()}")

# -----------------------------------------------------------------------------
# Conversion en binaire
# -----------------------------------------------------------------------------
def prepare_binary_labels(df, label_column):
    if pd.api.types.is_numeric_dtype(df[label_column]):
        y = df[label_column].values
        unique_vals = np.unique(y[~pd.isna(y)])
        if set(unique_vals).issubset({0,1}):
            return y.astype(int)
        else:
            return (y > 0).astype(int)
    # text labels:
    label_values = df[label_column].astype(str).str.lower().str.strip()
    normal_categories = {'normal','benign','0','false','no attack','none','clean','legitimate','regular'}
    attack_categories = {'analysis','backdoor','dos','exploits','fuzzers','generic','reconnaissance','shellcode','worms','attack','malicious','anomaly','1','true','yes','fuzz','recon','worm'}
    y = np.zeros(len(label_values), dtype=int)
    for i, v in enumerate(label_values):
        v = str(v)
        if any(a in v for a in attack_categories):
            y[i] = 1
        elif any(n in v for n in normal_categories):
            y[i] = 0
        else:
            # par défaut considérer comme attaque si inconnu (optionnel)
            y[i] = 1
    return y

y_train = prepare_binary_labels(df_train, label_col)
print(f"➡️ Distribution labels train: normal={(y_train==0).sum():,}, attack={(y_train==1).sum():,}")

# -----------------------------------------------------------------------------
# Sélection features numériques
# -----------------------------------------------------------------------------
exclude_columns = {label_col, 'srcip','dstip','stime','ltime','attack_cat','Label','label'}
recommended_numeric_features = [
    'dur','sbytes','dbytes','sttl','dttl','sloss','dloss','sload','dload','spkts','dpkts','swin','dwin',
    'stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','sjit','djit','sintpkt','dintpkt',
    'tcprtt','synack','ackdat','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
    'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm'
]

selected_features = [f for f in recommended_numeric_features if f in df_train.columns and f not in exclude_columns and pd.api.types.is_numeric_dtype(df_train[f])]
# si pas assez, compléter
if len(selected_features) < 10:
    for c in df_train.columns:
        if c not in selected_features and c not in exclude_columns and pd.api.types.is_numeric_dtype(df_train[c]):
            selected_features.append(c)
            if len(selected_features) >= 15:
                break

print(f"✅ {len(selected_features)} features numériques sélectionnées.")
X_train_raw = df_train[selected_features].replace([np.inf, -np.inf], np.nan).copy()
# imputer median
for col in X_train_raw.columns:
    if X_train_raw[col].isna().any():
        X_train_raw[col].fillna(X_train_raw[col].median(), inplace=True)
# drop constant cols
constant_cols = [c for c in X_train_raw.columns if X_train_raw[c].nunique() == 1]
if constant_cols:
    X_train_raw.drop(columns=constant_cols, inplace=True)
    selected_features = [f for f in selected_features if f not in constant_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

# sauvegarder scaler + features
joblib.dump(scaler, 'models/unsw_scaler.pkl')
with open('models/unsw_features.json','w') as fp:
    json.dump(selected_features, fp)

# -----------------------------------------------------------------------------
# Préparer test si présent sinon split
# -----------------------------------------------------------------------------
if df_test is not None:
    y_test = prepare_binary_labels(df_test, label_col)
    X_test_raw = df_test[selected_features].replace([np.inf, -np.inf], np.nan).copy()
    for col in X_test_raw.columns:
        if X_test_raw[col].isna().any():
            X_test_raw[col].fillna(X_train_raw[col].median(), inplace=True)
    X_test_scaled = scaler.transform(X_test_raw)
else:
    X_test_scaled = None
    y_test = None

# -----------------------------------------------------------------------------
# Création des séquences (temporal)
# -----------------------------------------------------------------------------
def create_temporal_sequences(X, y, seq_length=15, stride=3):
    n_samples, n_features = X.shape
    if n_samples < seq_length:
        raise ValueError("Trop peu d'échantillons pour créer une séquence.")
    n_sequences = (n_samples - seq_length) // stride + 1
    if n_sequences < 1000:
        stride = 1
        n_sequences = n_samples - seq_length + 1
    X_seq = np.zeros((n_sequences, seq_length, n_features), dtype=np.float32)
    y_seq = np.zeros(n_sequences, dtype=int)
    for i in range(n_sequences):
        start = i * stride
        end = start + seq_length
        X_seq[i] = X[start:end]
        y_seq[i] = y[end-1]
    return X_seq, y_seq

seq_length = 15
stride = 3
X_train_seq, y_train_seq = create_temporal_sequences(X_train_scaled, y_train, seq_length, stride)
if X_test_scaled is not None:
    X_test_seq, y_test_seq = create_temporal_sequences(X_test_scaled, y_test, seq_length, stride)
else:
    X_test_seq, y_test_seq = None, None

# -----------------------------------------------------------------------------
# Split final / class weights
# -----------------------------------------------------------------------------
if X_test_seq is not None:
    X_train_final = X_train_seq
    y_train_final = y_train_seq
    X_val, X_test, y_val, y_test = train_test_split(X_test_seq, y_test_seq, test_size=0.5, random_state=42, stratify=y_test_seq)
else:
    X_temp, X_test, y_temp, y_test = train_test_split(X_train_seq, y_train_seq, test_size=0.15, random_state=42, stratify=y_train_seq)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)

class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_final), y=y_train_final)
# map to classes (np.unique ordered)
classes_unique = np.unique(y_train_final)
class_weights_dict = {int(classes_unique[i]): class_weights_arr[i] for i in range(len(classes_unique))}
print(f"Class weights: {class_weights_dict}")

print(f"\n📊 Répartition: train={X_train_final.shape}, val={X_val.shape}, test={X_test.shape}")

# -----------------------------------------------------------------------------
# Construction des modèles (metrics cohérentes)
# -----------------------------------------------------------------------------
input_shape = (X_train_final.shape[1], X_train_final.shape[2])
print(f"Input shape: {input_shape}")

def build_lstm(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True, dropout=0.3),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(64, dropout=0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ], name="LSTM_Model")
    # Ajout d'AUC (ROC) et AUC (PR)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),              # ROC AUC
            keras.metrics.AUC(name='auc_pr', curve='PR') # PR AUC
        ]
    )
    return model

def build_rnn(input_shape):
    model = keras.Sequential([
        keras.layers.SimpleRNN(128, input_shape=input_shape, return_sequences=True, dropout=0.3),
        keras.layers.BatchNormalization(),
        keras.layers.SimpleRNN(64, dropout=0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ], name="SimpleRNN_Model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

def build_gru(input_shape):
    model = keras.Sequential([
        keras.layers.GRU(128, input_shape=input_shape, return_sequences=True, dropout=0.3),
        keras.layers.BatchNormalization(),
        keras.layers.GRU(64, dropout=0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ], name="GRU_Model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

model_lstm = build_lstm(input_shape)
model_rnn = build_rnn(input_shape)
model_gru = build_gru(input_shape)

print(f"Paramètres LSTM: {model_lstm.count_params():,}")

# -----------------------------------------------------------------------------
# Entraînement principal (LSTM)
# -----------------------------------------------------------------------------
start_time = time.time()
print(f"⏰ Début entraînement: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

callbacks_lstm = [
    keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=8, restore_best_weights=True, mode='max', verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath='models/best_lstm_unsw.h5', monitor='val_auc_pr', save_best_only=True, mode='max', verbose=1)
]

history = model_lstm.fit(
    X_train_final, y_train_final,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=callbacks_lstm,
    verbose=1,
    shuffle=True
)

training_time = time.time() - start_time
print(f"⏱ Temps entraînement LSTM: {training_time:.1f}s")

# Si checkpoint sauvegardé, recharger le meilleur
best_lstm_path = 'models/best_lstm_unsw.h5'
if os.path.exists(best_lstm_path):
    model_lstm = keras.models.load_model(best_lstm_path)

# -----------------------------------------------------------------------------
# Boucle de comparaison: entraînement rapide pour RNN & GRU (si tu veux comparer)
# -----------------------------------------------------------------------------
models = {"SimpleRNN": model_rnn, "LSTM": model_lstm, "GRU": model_gru}
results_comparison = {}

for name, model in models.items():
    print(f"\n=== Entraînement/éval rapide: {name} ===")
    # callbacks minimaux
    ckpt_path = f"models/best_{name.lower()}.h5"
    cb = [
        keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=4, restore_best_weights=True, mode='max', verbose=0),
        keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_auc_pr', save_best_only=True, mode='max', verbose=0)
    ]
    # entraînement court (si modèle déjà entraîné, fit continue / réentraîne)
    _start = time.time()
    model.fit(X_train_final, y_train_final, validation_data=(X_val,y_val), epochs=10, batch_size=64,
              class_weight=class_weights_dict, callbacks=cb, verbose=1, shuffle=True)
    _t = time.time() - _start
    if os.path.exists(ckpt_path):
        model = keras.models.load_model(ckpt_path)

    # évaluation sur test
    test_results = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    results_comparison[name] = {
        'accuracy': test_results[1],
        'precision': prec,
        'recall': rec,
        'auc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'training_time': _t,
        'params': model.count_params()
    }
    print(f"{name} -> F1: {f1:.4f}, AUC-PR: {auc_pr:.4f}")

# sauvegarder tableau de comparaison
comp_df = pd.DataFrame(results_comparison).T
comp_df.to_csv('results/model_comparison.csv')
print(comp_df)

# -----------------------------------------------------------------------------
# Calcul du meilleur seuil sur la validation (pour le modèle LSTM final)
# -----------------------------------------------------------------------------
probs_val = model_lstm.predict(X_val, verbose=0).flatten()
thresholds = np.linspace(0.01, 0.99, 99)
best_th = 0.5
best_f1 = -1
for t in thresholds:
    preds = (probs_val >= t).astype(int)
    f = f1_score(y_val, preds)
    if f > best_f1:
        best_f1 = f
        best_th = t
print(f"🔎 Meilleur threshold (val): {best_th:.3f} => F1={best_f1:.4f}")

# Évaluer sur test avec ce threshold (si test existe)
probs_test = model_lstm.predict(X_test, verbose=0).flatten()
preds_test = (probs_test >= best_th).astype(int)
f1_test = f1_score(y_test, preds_test)
prec_test = precision_score(y_test, preds_test, zero_division=0)
rec_test = recall_score(y_test, preds_test, zero_division=0)
aucpr_test = average_precision_score(y_test, probs_test)
aucroc_test = roc_auc_score(y_test, probs_test)

# sauvegarder seuil
with open('models/unsw_threshold.json','w') as fp:
    json.dump({'threshold': float(best_th)}, fp)

print("\n=== Résultats finaux LSTM (avec threshold optimal) ===")
print(f"F1 (test): {f1_test:.4f}, Precision: {prec_test:.4f}, Recall: {rec_test:.4f}, AUC-PR: {aucpr_test:.4f}, AUC-ROC: {aucroc_test:.4f}")

# Sauvegarder modèle final
model_lstm.save('models/lstm_unsw_final.h5')

print("\n✅ FIN DU SCRIPT. Fichiers créés dans /models et /results.")
