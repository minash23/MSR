import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score


class DependencyAnalyzerNB:
    def __init__(self):
        self.model = GaussianNB()
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

        # Create categorical labels based on number of dependent projects
        df['impact_category'] = pd.qcut(df['Dependent Projects'],
                                        q=5,
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        print(f"\nNumber of samples after cleaning: {len(df)}")
        print("\nData statistics after cleaning:")
        print(df.describe())
        print("\nImpact category distribution:")
        print(df['impact_category'].value_counts())

        return df

    def prepare_features(self, df):
        # Select features
        features = ['Total Dependencies', 'avg_dependency_depth', 'max_dependency_chain']
        X = df[features].values
        y = df['impact_category'].values

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
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate class probabilities
        class_probs = self.model.predict_proba(X_test)

        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))

        # Store feature means for each class
        feature_means = {}
        for class_label in np.unique(y):
            class_mask = (y_train == class_label)
            feature_means[class_label] = {
                feature: np.mean(X_train[class_mask, i])
                for i, feature in enumerate(feature_names)
            }

        print("\nFeature means by impact category:")
        for category, means in feature_means.items():
            print(f"\n{category}:")
            for feature, mean in means.items():
                print(f"  {feature}: {mean:.2f}")

        return {
            'accuracy': accuracy,
            'feature_means': feature_means,
            'test_actual': y_test,
            'test_pred': y_pred,
            'class_probabilities': class_probs
        }

    def predict_impact(self, total_deps, avg_depth, max_chain):
        input_data = np.array([[total_deps, avg_depth, max_chain]])
        input_scaled = self.scaler.transform(input_data)

        # Get both prediction and probabilities
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]

        # Get probability labels in the same order as probabilities
        probability_labels = self.model.classes_

        return {
            'predicted_category': prediction,
            'probabilities': dict(zip(probability_labels, probabilities))
        }


# Main execution
if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv('top50dependencies.csv')

        print("Initial data shape:", df.shape)
        print("\nInitial data sample:")
        print(df.head())

        # Initialize and run analysis
        analyzer = DependencyAnalyzerNB()
        metrics = analyzer.train_and_evaluate(df)

        # Print results
        print("\nModel Performance Summary:")
        print(f"Accuracy Score: {metrics['accuracy']:.3f}")

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
        print(f"Predicted impact category: {example['predicted_category']}")
        print("\nProbability breakdown:")
        for category, prob in example['probabilities'].items():
            print(f"- {category}: {prob:.1%}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nData info:")
        print(df.info())
