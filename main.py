import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score


class DependencyAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = RobustScaler()

    def preprocess_data(self, df):
        # Create a copy
        df = df.copy()

        # First, clean the 'Dependent Projects' column
        df = df[df['Dependent Projects'] != "no changes, no records"]
        df['Dependent Projects'] = pd.to_numeric(df['Dependent Projects'], errors='coerce')

        # Convert other numeric columns
        numeric_columns = ['Total Dependencies', 'Recent Dependents',
                           'avg_dependency_depth', 'max_dependency_chain']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any NaN values
        df = df.dropna()

        print(f"\nNumber of samples after cleaning: {len(df)}")
        print("\nData statistics after cleaning:")
        print(df.describe())

        return df

    def prepare_features(self, df):
        # Select features
        features = ['Total Dependencies', 'avg_dependency_depth', 'max_dependency_chain']
        X = df[features].values
        y = df['Dependent Projects'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, features

    def train_and_evaluate(self, df):
        # Preprocess data
        clean_df = self.preprocess_data(df)

        if len(clean_df) < 10:
            raise ValueError("Not enough samples after cleaning. Need at least 10 samples.")

        # Prepare features and target
        X, y, feature_names = self.prepare_features(clean_df)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Get feature importance
        importance = self.model.feature_importances_

        print("\nFeature Importance:")
        for name, imp in zip(feature_names, importance):
            print(f"{name}: {imp:.3f}")

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'test_actual': y_test,
            'test_pred': y_pred,
            'feature_importance': dict(zip(feature_names, importance))
        }

    def predict_impact(self, total_deps, avg_depth, max_chain):
        input_data = np.array([[total_deps, avg_depth, max_chain]])
        input_scaled = self.scaler.transform(input_data)
        return self.model.predict(input_scaled)[0]


# Main execution
if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('top50dependencies.csv')

        print("Initial data shape:", df.shape)
        print("\nInitial data sample:")
        print(df.head())

        # Initialize and run analysis
        analyzer = DependencyAnalyzer()
        metrics = analyzer.train_and_evaluate(df)

        # Print results
        print("\nModel Performance Metrics:")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")

        # Example prediction
        example = analyzer.predict_impact(
            total_deps=2000,
            avg_depth=30,
            max_chain=100
        )
        print(f"\nExample Prediction:")
        print(f"For a project with:")
        print(f"- Total Dependencies: 2000")
        print(f"- Average Dependency Depth: 30")
        print(f"- Maximum Chain Length: 100")
        print(f"Predicted number of dependent projects: {example:.0f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nData info:")
        print(df.info())