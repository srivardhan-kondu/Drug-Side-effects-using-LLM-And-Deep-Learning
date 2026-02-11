import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
import joblib
from sklearn.model_selection import train_test_split

# Fix path to allow imports from root
sys.path.append(os.getcwd())

from models.deep_learning_model import DrugSideEffectModel

# Configuration
DATA_PATH = "data/drug_data.json"
MODEL_PATH = "saved_models/drug_side_effect_model.pth"
METADATA_PATH = "saved_models/model_metadata.pkl"
AUGMENT_FACTOR = 50 # Generate 50 variations per drug -> ~5,700 samples (enough for demo)

def load_and_augment_data():
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'r') as f:
        db = json.load(f)
    
    drugs = db["drugs"]
    categories = db["categories"] # For encoding
    all_effects = db["all_side_effects"] # For labels
    
    features_list = []
    labels_list = []
    
    # Feature Helper
    def get_features(drug_data):
        cat = drug_data.get("category", "Other")
        # Ensure category matches one of the known categories, or default to 0s
        cat_enc = [1 if c == cat else 0 for c in categories]
        if sum(cat_enc) == 0:
             cat_enc = [0] * len(categories)

        # Normalize features roughly to [0, 1] range based on typical max values
        num_feats = [
            drug_data.get("molecular_weight", 300) / 800.0,
            (drug_data.get("log_p", 2.0) + 2.0) / 10.0,
            drug_data.get("h_bond_donors", 2) / 10.0,
            drug_data.get("h_bond_acceptors", 5) / 15.0,
            drug_data.get("polar_surface_area", 80.0) / 200.0,
            drug_data.get("rotatable_bonds", 5) / 20.0,
        ]
        return np.array(num_feats + cat_enc, dtype=np.float32)

    # Label Helper
    def get_labels(drug_data):
        lbls = [0.0] * len(all_effects)
        for eff, score in drug_data.get("side_effects", {}).items():
            if eff in all_effects:
                idx = all_effects.index(eff)
                lbls[idx] = score # Use severity score
        return np.array(lbls, dtype=np.float32)

    print("Augmenting data...")
    for name, data in drugs.items():
        base_feat = get_features(data)
        base_label = get_labels(data)
        
        # Add original
        features_list.append(base_feat)
        labels_list.append(base_label)
        
        # Augment
        for _ in range(AUGMENT_FACTOR):
            # Noise in features
            noise = np.random.normal(0, 0.02, base_feat.shape)
            aug_feat = np.clip(base_feat + noise, 0, 1)
            
            # Noise in labels (slightly) - regression target jitter
            # We want the model to be robust to exact scores
            label_noise = np.random.normal(0, 0.02, base_label.shape)
            aug_label = np.clip(base_label + label_noise, 0, 1)
            # Keep zeros as zeros (sparse)
            aug_label[base_label == 0] = 0 
            
            features_list.append(aug_feat)
            labels_list.append(aug_label)
            
    return np.array(features_list), np.array(labels_list), categories, all_effects

def train():
    X, y, categories, side_effects = load_and_augment_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    print(f"Model Input Dim: {input_dim}, Output Dim: {output_dim}")
    
    model = DrugSideEffectModel(input_dim, output_dim, hidden_dims=[512, 256, 128]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Predicting severity scores
    
    print("Training...")
    BATCH_SIZE = 64
    EPOCHS = 20
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            # Use severity_scores head
            loss = criterion(outputs["severity_scores"], batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Val
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out["severity_scores"], y_val_t)
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss {total_loss/len(train_loader):.4f}, Val Loss {val_loss.item():.4f}")
            
    # Save
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    metadata = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "categories": categories,
        "side_effects": side_effects
    }
    joblib.dump(metadata, METADATA_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
