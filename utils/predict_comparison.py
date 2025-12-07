# predict_comparison.py - Prédiction avec comparaison des modèles
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import joblib

print("="*80)
print("🔮 PRÉDICTEUR COMPARATIF RNN/LSTM/GRU")
print("="*80)

def load_all_models():
    """Charge tous les modèles entraînés"""
    models = {}
    
    model_files = {
        "RNN": "models/best_rnn.h5",
        "LSTM": "models/best_lstm.h5", 
        "GRU": "models/best_gru.h5"
    }
    
    print("📥 Chargement des modèles...")
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
            print(f"   ✅ {name} chargé")
        else:
            print(f"   ⚠️  {name} non trouvé: {path}")
    
    return models

def predict_with_all_models(models, sample_data):
    """Prédit avec tous les modèles"""
    results = {}
    
    for name, model in models.items():
        prediction = model.predict(sample_data, verbose=0)[0][0]
        pred_label = 1 if prediction > 0.5 else 0
        
        results[name] = {
            'probability': float(prediction),
            'prediction': pred_label,
            'interpretation': '🚨 ATTAQUE' if pred_label == 1 else '✅ NORMAL',
            'confidence': float(prediction if pred_label == 1 else 1 - prediction)
        }
    
    return results

def main():
    """Fonction principale"""
    # Charger les modèles
    models = load_all_models()
    
    if not models:
        print("❌ Aucun modèle chargé")
        return
    
    # Charger le scaler et les features
    scaler = joblib.load("models/scaler.pkl")
    with open("models/features.json", "r") as f:
        features = json.load(f)
    
    print(f"\n📋 Informations:")
    print(f"   • Modèles disponibles: {list(models.keys())}")
    print(f"   • Features: {len(features)}")
    
    # Mode interactif
    print("\n🎮 MODE INTERACTIF")
    print("="*40)
    
    while True:
        print("\nOptions:")
        print("  1. Tester avec données aléatoires")
        print("  2. Comparer les modèles sur un échantillon")
        print("  3. Quitter")
        
        choice = input("\nVotre choix (1-3): ").strip()
        
        if choice == '1':
            # Données aléatoires
            print("\n🔬 Génération de données aléatoires...")
            
            # Créer des données d'exemple
            sample_data = {}
            for feature in features:
                sample_data[feature] = np.random.normal(0, 1)
            
            # Convertir en format approprié
            sample_df = pd.DataFrame([sample_data])
            sample_scaled = scaler.transform(sample_df)
            
            # Créer une séquence (répétition)
            seq_length = 10
            sequence = np.tile(sample_scaled, (seq_length, 1))
            sequence = sequence.reshape(1, seq_length, -1)
            
            # Prédire avec tous les modèles
            results = predict_with_all_models(models, sequence)
            
            print(f"\n📊 RÉSULTATS:")
            for model_name, result in results.items():
                print(f"\n  {model_name}:")
                print(f"    • Score: {result['probability']:.4f}")
                print(f"    • Prédiction: {result['interpretation']}")
                print(f"    • Confiance: {result['confidence']:.2%}")
        
        elif choice == '2':
            # Comparaison détaillée
            print("\n📊 COMPARAISON DES MODÈLES")
            
            # Charger les résultats de comparaison
            if os.path.exists("results/comparison_results.json"):
                with open("results/comparison_results.json", "r") as f:
                    comparison = json.load(f)
                
                print(f"\n📈 PERFORMANCES SUR LE JEU DE TEST:")
                print(f"{'Modèle':10} {'Accuracy':10} {'Precision':10} {'Recall':10} {'F1-Score':10}")
                print("-" * 60)
                
                for model_name, metrics in comparison.items():
                    print(f"{model_name:10} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     "
                          f"{metrics['recall']:.4f}     {metrics['f1_score']:.4f}")
                
                # Trouver le meilleur
                best_model = max(comparison.items(), key=lambda x: x[1]['f1_score'])[0]
                print(f"\n🏆 MEILLEUR MODÈLE: {best_model}")
                print(f"   • F1-Score: {comparison[best_model]['f1_score']:.4f}")
            
        elif choice == '3':
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide")

if __name__ == "__main__":
    main()