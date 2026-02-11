"""
Deep Learning Model for Drug Side Effect Prediction
Multi-label classification using a neural network with attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionBlock(nn.Module):
    """Self-attention mechanism for feature importance."""
    
    def __init__(self, embed_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = np.sqrt(embed_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        return attended, attention_weights


class DrugSideEffectModel(nn.Module):
    """
    Neural network for predicting drug side effects.
    
    Architecture:
    - Input: Drug molecular features + category encoding
    - Hidden layers with batch normalization and dropout
    - Attention mechanism for interpretability
    - Output: Multi-label binary predictions + severity scores
    """
    
    def __init__(self, input_dim, num_side_effects, hidden_dims=[512, 256, 128]):
        super(DrugSideEffectModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_side_effects = num_side_effects
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),  # Reduced dropout slightly for better learning
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = AttentionBlock(hidden_dims[-1])
        
        # Side effect prediction head (binary)
        self.side_effect_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1], num_side_effects),
            nn.Sigmoid()
        )
        
        # Severity prediction head
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1], num_side_effects),
            nn.Sigmoid()
        )
        
        # Feature importance layer
        self.feature_importance = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        # Feature importance weighting
        importance = torch.sigmoid(self.feature_importance(x))
        x_weighted = x * importance
        
        # Extract features
        features = self.feature_extractor(x_weighted)
        
        # Apply attention (reshape for attention)
        features_reshaped = features.unsqueeze(1)
        attended_features, attention_weights = self.attention(features_reshaped)
        attended_features = attended_features.squeeze(1)
        
        # Combine original and attended features
        combined = features + attended_features
        
        # Predict side effects and severity
        side_effect_probs = self.side_effect_head(combined)
        severity_scores = self.severity_head(combined)
        
        return {
            "side_effect_probs": side_effect_probs,
            "severity_scores": severity_scores,
            "attention_weights": attention_weights,
            "feature_importance": importance
        }


class DrugSideEffectPredictor:
    """High-level predictor wrapper for the model."""
    
    def __init__(self, model_path=None, input_dim=None, num_side_effects=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path:
            self.load_model(model_path)
        elif input_dim and num_side_effects:
            self.model = DrugSideEffectModel(input_dim, num_side_effects).to(self.device)
        else:
            raise ValueError("Either model_path or (input_dim, num_side_effects) required")
        
        self.model.eval()
    
    def load_model(self, model_path):
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        input_dim = checkpoint["input_dim"]
        num_side_effects = checkpoint["num_side_effects"]
        
        self.model = DrugSideEffectModel(input_dim, num_side_effects).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.side_effect_names = checkpoint.get("side_effect_names", [])
        self.feature_names = checkpoint.get("feature_names", [])
        
        print(f"Model loaded from {model_path}")
        print(f"  Input dim: {input_dim}, Side effects: {num_side_effects}")
    
    def predict(self, features, threshold=0.4):
        """
        Predict side effects for given drug features.
        
        Args:
            features: numpy array of drug features
            threshold: probability threshold for positive prediction
            
        Returns:
            dict with predictions, probabilities, and severities
        """
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            output = self.model(features)
            
            probs = output["side_effect_probs"].cpu().numpy()[0]
            severities = output["severity_scores"].cpu().numpy()[0]
            importance = output["feature_importance"].cpu().numpy()[0]
            
            # Get predicted side effects
            predicted_effects = []
            for i, (prob, sev) in enumerate(zip(probs, severities)):
                if prob >= threshold:
                    effect_name = self.side_effect_names[i] if i < len(self.side_effect_names) else f"Effect_{i}"
                    predicted_effects.append({
                        "name": effect_name,
                        "probability": float(prob),
                        "severity": float(sev),
                        "severity_label": self._severity_label(sev)
                    })
            
            # Sort by probability
            predicted_effects.sort(key=lambda x: x["probability"], reverse=True)
            
            # Feature importance
            feature_importance = {}
            for i, imp in enumerate(importance):
                fname = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
                feature_importance[fname] = float(imp)
            
            return {
                "predicted_side_effects": predicted_effects,
                "all_probabilities": {
                    self.side_effect_names[i]: float(p) 
                    for i, p in enumerate(probs)
                },
                "feature_importance": feature_importance,
                "total_predicted": len(predicted_effects)
            }
    
    @staticmethod
    def _severity_label(severity):
        """Convert severity score to human-readable label."""
        if severity >= 0.7:
            return "Severe"
        elif severity >= 0.4:
            return "Moderate"
        else:
            return "Mild"
