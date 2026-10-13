# train_and_save.py
import sys
from pathlib import Path
from sklearn.ensemble import IsolationForest
import pandas as pd
import pickle

def main():
    # 1. Validate command line arguments
    if len(sys.argv) < 2:
        print("❌ Error: Missing input file.")
        print("Usage: python train_and_save.py <path_to_csv_file>")
        sys.argv.append("bathroom_history.csv") # Default fallback if nothing provided
        print(f"Using default fallback: {sys.argv[1]}\n")

    # 2. Extract file paths dynamically
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"❌ Error: The file '{csv_path}' does not exist.")
        sys.exit(1)

    # Generate the .pkl path dynamically (e.g., "data.csv" -> "data.pkl")
    pkl_path = csv_path.with_suffix('.pkl')

    print(f"⚙️ Loading data from '{csv_path}'...")

    # 3. Load and clean data
    df = pd.read_csv(csv_path).fillna(0)
    df.columns = df.columns.str.strip().str.lower()

    # 4. Train model
    feature = 'activity_metric' 
    if feature not in df.columns:
        print(f"❌ Error: Column '{feature}' not found in the CSV file.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    X_train = df[[feature]].values
    
    print("🧠 Training Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train)

    # 5. Save model to file using the dynamic .pkl path
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    print(f"🎉 Model trained successfully and saved as '{pkl_path}'!")

if __name__ == "__main__":
    main()
