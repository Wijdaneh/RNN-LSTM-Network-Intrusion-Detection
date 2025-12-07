"""
Module de visualisation pour la détection d'intrusion avec UNSW-NB15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class UNSW_Visualizer:
    """Classe pour la visualisation des données UNSW-NB15"""
    
    def __init__(self, color_palette='Set3'):
        self.color_palette = color_palette
        self.unsw_colors = {
            'Normal': '#2Ecc71',
            'Generic': '#E74C3C',
            'Exploits': '#9B59B6',
            'Fuzzers': '#3498DB',
            'DoS': '#E67E22',
            'Reconnaissance': '#1ABC9C',
            'Analysis': '#F1C40F',
            'Backdoor': '#7F8C8D',
            'Shellcode': '#D35400',
            'Worms': '#C0392B'
        }
        self.set_style()
    
    def set_style(self):
        """Définit le style des graphiques"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette(self.color_palette)
        
        # Style Plotly pour UNSW
        self.plotly_template = 'plotly_white'
        
        # Couleurs UNSW pour Plotly
        self.plotly_colors = {
            'Normal': 'green',
            'Attack': 'red',
            'Generic': 'red',
            'Exploits': 'purple',
            'Fuzzers': 'blue',
            'DoS': 'orange',
            'Reconnaissance': 'teal',
            'Analysis': 'yellow',
            'Backdoor': 'gray',
            'Shellcode': 'brown',
            'Worms': 'darkred'
        }
    
    def plot_unsw_attack_distribution(self, labels=None, attack_cats=None):
        """Visualise la distribution des attaques UNSW-NB15"""
        
        # Données par défaut si non fournies
        if attack_cats is None:
            attack_data = {
                'Attack Type': ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 
                               'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'],
                'Count': [2000000, 215481, 44525, 24246, 16353, 
                         13987, 2677, 2329, 1511, 174],
                'Percentage': [78.0, 8.5, 1.8, 1.0, 0.6, 
                             0.6, 0.1, 0.1, 0.1, 0.01]
            }
            df = pd.DataFrame(attack_data)
        else:
            # Utiliser les données fournies
            df = pd.DataFrame({'Attack Type': attack_cats})
            df['Count'] = df['Attack Type'].value_counts().values
            df['Percentage'] = df['Count'] / df['Count'].sum() * 100
        
        # Graphique à barres
        fig = px.bar(df, x='Attack Type', y='Count',
                     color='Attack Type',
                     title='Distribution des attaques UNSW-NB15',
                     color_discrete_map=self.unsw_colors,
                     hover_data=['Percentage'])
        
        fig.update_layout(
            xaxis_title="Type d'attaque",
            yaxis_title="Nombre d'échantillons",
            showlegend=False,
            xaxis_tickangle=45,
            template=self.plotly_template
        )
        
        # Ajouter les pourcentages sur les barres
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['Attack Type'],
                y=row['Count'],
                text=f"{row['Percentage']:.1f}%",
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )
        
        return fig
    
    def plot_binary_distribution(self, labels, title="Distribution binaire (Normal/Attaque)"):
        """Visualise la distribution binaire"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Mapper 0/1 à Normal/Attack
        label_names = ['Normal' if label == 0 else 'Attack' for label in unique_labels]
        
        fig = px.pie(values=counts, names=label_names,
                     title=title,
                     color=label_names,
                     color_discrete_map={'Normal': 'green', 'Attack': 'red'})
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template=self.plotly_template)
        
        return fig
    
    def plot_feature_importance(self, feature_names, importances, top_n=15):
        """Visualise l'importance des features"""
        # Créer un DataFrame pour les features les plus importantes
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(top_n)
        
        fig = px.bar(importance_df, 
                     x='Importance', y='Feature',
                     orientation='h',
                     title=f'Top {top_n} des features les plus importantes',
                     color='Importance',
                     color_continuous_scale='Viridis')
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Importance",
            yaxis_title="Feature",
            template=self.plotly_template,
            height=400
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data, title="Matrice de corrélation"):
        """Affiche la matrice de corrélation"""
        # Calculer la corrélation
        corr_matrix = data.corr()
        
        # Réduire la taille si trop grande
        if corr_matrix.shape[0] > 30:
            # Prendre les 30 features les plus corrélées avec la target
            if 'label' in corr_matrix.columns:
                top_features = corr_matrix['label'].abs().sort_values(ascending=False).head(30).index
                corr_matrix = corr_matrix.loc[top_features, top_features]
        
        fig = px.imshow(corr_matrix,
                       title=title,
                       color_continuous_scale='RdBu',
                       zmin=-1, zmax=1,
                       aspect='auto')
        
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            template=self.plotly_template,
            height=600
        )
        
        return fig
    
    def plot_model_comparison(self, models_data):
        """Compare les performances des modèles"""
        # Exemple de données
        if models_data is None:
            models_data = {
                'Model': ['LSTM', 'GRU', 'RNN'],
                'Accuracy': [0.982, 0.976, 0.961],
                'Precision': [0.985, 0.978, 0.965],
                'Recall': [0.981, 0.975, 0.958],
                'F1-Score': [0.983, 0.976, 0.961],
                'AUC-ROC': [0.995, 0.992, 0.985]
            }
        
        df = pd.DataFrame(models_data)
        
        # Créer un graphique radar
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        for i, model in enumerate(df['Model']):
            values = df.loc[df['Model'] == model, metrics].values[0]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.9, 1.0]
                )
            ),
            title='Comparaison des modèles (Radar Chart)',
            template=self.plotly_template,
            showlegend=True
        )
        
        return fig
    
    def plot_training_history(self, history_dict, model_name="Model"):
        """Affiche l'historique d'entraînement"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Precision', 'Recall'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(y=history_dict.get('loss', []), mode='lines', name='Train Loss'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history_dict.get('val_loss', []), mode='lines', name='Val Loss'),
            row=1, col=1
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(y=history_dict.get('accuracy', []), mode='lines', name='Train Acc'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history_dict.get('val_accuracy', []), mode='lines', name='Val Acc'),
            row=1, col=2
        )
        
        # Precision
        if 'precision' in history_dict:
            fig.add_trace(
                go.Scatter(y=history_dict.get('precision', []), mode='lines', name='Train Precision'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=history_dict.get('val_precision', []), mode='lines', name='Val Precision'),
                row=2, col=1
            )
        
        # Recall
        if 'recall' in history_dict:
            fig.add_trace(
                go.Scatter(y=history_dict.get('recall', []), mode='lines', name='Train Recall'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(y=history_dict.get('val_recall', []), mode='lines', name='Val Recall'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Historique d\'entraînement - {model_name}',
            height=600,
            template=self.plotly_template,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=2, col=1)
        fig.update_yaxes(title_text="Recall", row=2, col=2)
        
        return fig
    
    def plot_confusion_matrix(self, cm, model_name="Model", classes=['Normal', 'Attack']):
        """Affiche la matrice de confusion"""
        fig = px.imshow(cm,
                       text_auto=True,
                       color_continuous_scale='Blues',
                       title=f'Matrice de confusion - {model_name}',
                       labels=dict(x="Prédiction", y="Vérité", color="Count"),
                       x=classes,
                       y=classes)
        
        fig.update_layout(
            xaxis_title="Prédiction",
            yaxis_title="Vérité",
            template=self.plotly_template,
            coloraxis_showscale=False
        )
        
        return fig
    
    def plot_roc_curve(self, fpr, tpr, auc_score, model_name="Model"):
        """Trace la courbe ROC"""
        fig = go.Figure()
        
        # Courbe ROC
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Ligne de référence
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Courbe ROC - {model_name}',
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template=self.plotly_template,
            showlegend=True,
            width=600,
            height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions):
        """Visualise la confiance des prédictions"""
        fig = px.histogram(predictions, x='confidence', 
                          color='prediction',
                          nbins=30,
                          title='Distribution des confiances de prédiction',
                          labels={'confidence': 'Confiance', 'count': 'Nombre'},
                          color_discrete_map={'Attack': 'red', 'Normal': 'green'})
        
        fig.update_layout(
            barmode='overlay',
            template=self.plotly_template,
            showlegend=True
        )
        fig.update_traces(opacity=0.7)
        
        return fig
    
    def plot_attack_timeline(self, predictions, time_column='timestamp'):
        """Crée une timeline des attaques"""
        if time_column not in predictions.columns:
            predictions[time_column] = pd.date_range(
                start='2024-01-01', 
                periods=len(predictions), 
                freq='S'
            )
        
        # Séparer attaques et normal
        attacks = predictions[predictions['is_attack']]
        normal = predictions[~predictions['is_attack']]
        
        fig = go.Figure()
        
        # Points normaux
        if len(normal) > 0:
            fig.add_trace(go.Scatter(
                x=normal[time_column],
                y=normal['confidence'],
                mode='markers',
                name='Normal',
                marker=dict(color='green', size=5, opacity=0.5)
            ))
        
        # Points d'attaque
        if len(attacks) > 0:
            fig.add_trace(go.Scatter(
                x=attacks[time_column],
                y=attacks['confidence'],
                mode='markers',
                name='Attaque',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig.update_layout(
            title='Timeline des prédictions',
            xaxis_title="Temps",
            yaxis_title="Confiance",
            template=self.plotly_template,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def plot_feature_distributions(self, data, feature_names, n_features=9):
        """Affiche les distributions des features"""
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=feature_names[:n_features]
        )
        
        for i in range(min(n_features, len(feature_names))):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            feature = feature_names[i]
            if feature in data.columns:
                fig.add_trace(
                    go.Histogram(x=data[feature].dropna(), name=feature),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text='Distributions des features',
            height=300 * n_rows,
            template=self.plotly_template,
            showlegend=False
        )
        
        return fig
    
    def create_dashboard(self, metrics_dict):
        """Crée un dashboard complet avec plusieurs visualisations"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Accuracy', 'Precision', 'Recall',
                          'F1-Score', 'AUC-ROC', 'Confusion Matrix',
                          'ROC Curve', 'Feature Importance', 'Training History'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics_dict.get('accuracy', 0) * 100,
                title={'text': "Accuracy"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 0}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics_dict.get('precision', 0) * 100,
                title={'text': "Precision"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 1}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics_dict.get('recall', 0) * 100,
                title={'text': "Recall"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 2}
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics_dict.get('f1_score', 0),
                title={'text': "F1-Score"},
                domain={'row': 2, 'column': 0}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=metrics_dict.get('auc_roc', 0),
                title={'text': "AUC-ROC"},
                domain={'row': 2, 'column': 1}
            ),
            row=2, col=2
        )
        
        # Confusion Matrix
        cm = metrics_dict.get('confusion_matrix', np.eye(2))
        fig.add_trace(
            go.Heatmap(
                z=cm,
                colorscale='Blues',
                showscale=True,
                text=cm.astype(str),
                texttemplate="%{text}"
            ),
            row=2, col=3
        )
        
        # ROC Curve (placeholder)
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'),
            row=3, col=1
        )
        
        # Feature Importance (placeholder)
        features = [f'Feature {i}' for i in range(10)]
        importances = np.random.rand(10)
        fig.add_trace(
            go.Bar(x=importances, y=features, orientation='h', name='Importance'),
            row=3, col=2
        )
        
        # Training History (placeholder)
        epochs = list(range(20))
        loss = [0.8 * (0.9 ** i) for i in epochs]
        fig.add_trace(
            go.Scatter(x=epochs, y=loss, mode='lines', name='Loss'),
            row=3, col=3
        )
        
        fig.update_layout(
            height=900,
            title_text="Dashboard de Performance - UNSW-NB15 IDS",
            template=self.plotly_template,
            showlegend=False
        )
        
        return fig
    
    def plot_model_performance_comparison(self, models_performance):
        """Compare les performances de plusieurs modèles"""
        df = pd.DataFrame(models_performance)
        
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        for metric in metrics:
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df['Model'],
                    y=df[metric],
                    text=df[metric].round(3),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title='Comparaison des performances des modèles',
            xaxis_title="Modèle",
            yaxis_title="Score",
            barmode='group',
            template=self.plotly_template,
            yaxis_range=[0.9, 1.0]
        )
        
        return fig