# Financial News + Market Hybrid Model

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

def main():
    # 1. Load Data
    df = pd.read_csv("data.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # 2. Handle Text
    df["Headlines"] = df["Headlines"].str.strip()
    df.loc[df["Headlines"] == "No major news.", "Headlines"] = "no_news"

    # Flag feature
    df["is_no_news"] = (df["Headlines"] == "no_news").astype(int)

    # 3. Create Target 
    threshold = df["Target"].quantile(0.5)
    df["Direction"] = (df["Target"] > threshold).astype(int)

    # 4. Train-Test Split (Time-based)
    train = df[df["Date"] <= "2012-05-22"]
    test  = df[df["Date"] >  "2012-05-22"]

    y_train = train["Direction"]
    y_test  = test["Direction"]

    # 5. FinBERT Embeddings 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model_bert = AutoModel.from_pretrained("ProsusAI/finbert")
    model_bert.to(device)
    model_bert.eval()

    def get_embeddings(texts, batch_size=8):
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_bert(**inputs)
            
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

    print("Generating embeddings...")
    X_text_train = get_embeddings(train["Headlines"].tolist())
    X_text_test  = get_embeddings(test["Headlines"].tolist())

    # 6. Numeric Features
    lag_cols = [f"Return_Lag_{i}" for i in range(1, 101)]

    X_num_train = train[lag_cols + ["is_no_news"]].fillna(0)
    X_num_test  = test[lag_cols + ["is_no_news"]].fillna(0)

    # 7. Combine Features
    X_train = np.hstack([X_text_train, X_num_train.values])
    X_test  = np.hstack([X_text_test,  X_num_test.values])

    # 8. Train Model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 9. Predictions
    y_prob = model.predict_proba(X_test)[:,1]

    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    # 10. Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 11. Threshold vs F1 Plot
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.05, 0.95, 50)
    f1_scores = []

    for t in thresholds:
        preds = (y_prob > t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1 Score")
    plt.show()

    # 12. ROC Curve
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(fpr, tpr, label="Model")
    plt.plot([0,1], [0,1], linestyle='--', label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()