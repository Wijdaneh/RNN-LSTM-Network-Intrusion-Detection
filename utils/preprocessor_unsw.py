"""
Module de prétraitement spécifique pour UNSW-NB15
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class UNSWPreprocessor:
    """Prétraitement spécifique pour le dataset UNSW-NB15"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.attack_cat_encoder = LabelEncoder()
        self.protocol_encoder = LabelEncoder()
        self.service_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        
    def load_unsw_data(self, file_paths):
        """Charge et combine les fichiers UNSW-NB15"""
        print(f"📥 Chargement des fichiers UNSW-NB15...")
        
        dataframes = []
        for file_path in file_paths:
            try:
                print(f"  Lecture de {file_path}...")
                df = pd.read_csv(file_path, low_memory=False)
                dataframes.append(df)
                print(f"    ✓ {len(df)} lignes chargées")
            except Exception as e:
                print(f"    ✗ Erreur: {e}")
        
        # Combiner tous les dataframes
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
            print(f"✅ Total: {len(combined_df):,} échantillons combinés")
            return combined_df
        else:
            raise ValueError("Aucune donnée chargée!")
    
    def map_attack_categories(self, attack_cat):
        """Simplifie les catégories d'attaques"""
        if pd.isna(attack_cat):
            return 'Normal'
        
        attack_cat = str(attack_cat).strip()
        
        # Mapping simplifié
        mapping = {
            'Normal': 'Normal',
            'Analysis': 'Analysis',
            'Backdoor': 'Backdoor',
            'DoS': 'DoS',
            'Exploits': 'Exploits',
            'Fuzzers': 'Fuzzers',
            'Generic': 'Generic',
            'Reconnaissance': 'Reconnaissance',
            'Shellcode': 'Shellcode',
            'Worms': 'Worms'
        }
        
        return mapping.get(attack_cat, 'Other')
    
    def preprocess_unsw(self, df):
        """Pipeline de prétraitement pour UNSW-NB15"""
        print("🔧 Début du prétraitement UNSW-NB15...")
        
        # Copie du dataframe
        df_processed = df.copy()
        
        # 1. Nettoyage des noms de colonnes
        df_processed.columns = df_processed.columns.str.strip()
        print(f"   Étape 1: Colonnes nettoyées - {len(df_processed.columns)} colonnes")
        
        # 2. Gestion des valeurs manquantes
        print("   Étape 2: Gestion des valeurs manquantes...")
        
        # Remplacer ' ' par NaN
        df_processed = df_processed.replace(r'^\s*$', np.nan, regex=True)
        
        # Colonnes catégorielles
        categorical_cols = ['proto', 'service', 'state']
        for col in categorical_cols:
            if col in df_processed.columns:
                # Remplacer NaN par 'unknown'
                df_processed[col] = df_processed[col].fillna('unknown')
        
        # Colonnes numériques
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
        
        # 3. Encodage des catégories d'attaques
        if 'attack_cat' in df_processed.columns:
            print("   Étape 3: Encodage des catégories d'attaques...")
            df_processed['attack_cat'] = df_processed['attack_cat'].apply(self.map_attack_categories)
        
        # 4. Encodage des variables catégorielles
        print("   Étape 4: Encodage des variables catégorielles...")
        
        # Protocole
        if 'proto' in df_processed.columns:
            self.protocol_encoder.fit(df_processed['proto'])
            df_processed['proto_encoded'] = self.protocol_encoder.transform(df_processed['proto'])
        
        # Service
        if 'service' in df_processed.columns:
            self.service_encoder.fit(df_processed['service'])
            df_processed['service_encoded'] = self.service_encoder.transform(df_processed['service'])
        
        # State
        if 'state' in df_processed.columns:
            self.state_encoder.fit(df_processed['state'])
            df_processed['state_encoded'] = self.state_encoder.transform(df_processed['state'])
        
        # 5. Préparation des labels
        print("   Étape 5: Préparation des labels...")
        
        # Créer une colonne label unifiée
        if 'label' in df_processed.columns:
            df_processed['binary_label'] = df_processed['label']  # 0=Normal, 1=Attack
            
        if 'attack_cat' in df_processed.columns:
            # Pour classification multiclasse
            self.attack_cat_encoder.fit(df_processed['attack_cat'])
            df_processed['multiclass_label'] = self.attack_cat_encoder.transform(df_processed['attack_cat'])
        
        # 6. Sélection et normalisation des features
        print("   Étape 6: Sélection et normalisation des features...")
        
        # Features à exclure (identifiants, etc.)
        exclude_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'label', 'attack_cat']
        
        # Features numériques pour la normalisation
        feature_cols = [col for col in df_processed.columns 
                       if col not in exclude_cols and df_processed[col].dtype in [np.int64, np.float64]]
        
        # Normalisation
        if feature_cols:
            df_processed[feature_cols] = self.scaler.fit_transform(df_processed[feature_cols])
        
        print(f"✅ Prétraitement terminé: {df_processed.shape}")
        print(f"   Features: {len(feature_cols)}")
        
        if 'binary_label' in df_processed.columns:
            normal_count = (df_processed['binary_label'] == 0).sum()
            attack_count = (df_processed['binary_label'] == 1).sum()
            print(f"   Normal: {normal_count:,} | Attaques: {attack_count:,}")
        
        return df_processed, feature_cols
    
    def create_sequences_unsw(self, df, feature_cols, label_col='binary_label', 
                             seq_length=20, step=5):
        """Crée des séquences pour RNN/LSTM à partir des données UNSW"""
        print(f"⏱️  Création de séquences (longueur={seq_length})...")
        
        # Extraire les features et labels
        X = df[feature_cols].values
        y = df[label_col].values if label_col in df.columns else None
        
        # Créer des séquences
        sequences = []
        labels = []
        
        for i in range(0, len(X) - seq_length, step):
            seq = X[i:i + seq_length]
            
            # Vérifier qu'il n'y a pas de NaN
            if not np.isnan(seq).any():
                sequences.append(seq)
                if y is not None:
                    labels.append(y[i + seq_length - 1])
        
        X_seq = np.array(sequences)
        
        if y is not None:
            y_seq = np.array(labels)
            print(f"✅ Séquences créées: {X_seq.shape}")
            return X_seq, y_seq
        else:
            print(f"✅ Séquences créées (sans labels): {X_seq.shape}")
            return X_seq
    
    def get_attack_statistics(self, df):
        """Retourne les statistiques des attaques"""
        stats = {}
        
        if 'attack_cat' in df.columns:
            attack_counts = df['attack_cat'].value_counts()
            stats['attack_distribution'] = attack_counts.to_dict()
            stats['total_attacks'] = len(df[df['attack_cat'] != 'Normal'])
            stats['total_normal'] = len(df[df['attack_cat'] == 'Normal'])
        
        if 'label' in df.columns:
            stats['binary_distribution'] = {
                'Normal': int((df['label'] == 0).sum()),
                'Attack': int((df['label'] == 1).sum())
            }
        
        return stats