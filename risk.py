import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class DependencyRiskAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = RobustScaler()

    def calculate_impact_score(self, row):
        """
        Enhanced impact score calculation with logarithmic scaling
        """
        # Log-scale the metrics to handle outliers better
        dep_score = np.log1p(row['Dependent Projects']) / np.log1p(10000)
        depth_score = np.log1p(row['avg_dependency_depth']) / np.log1p(100)
        chain_score = np.log1p(row['max_dependency_chain']) / np.log1p(200)
        total_deps_score = np.log1p(row['Total Dependencies']) / np.log1p(5000)

        # Adjusted weights based on domain knowledge
        weights = {
            'dependents': 0.35,  # Slightly reduced as it was too dominant
            'depth': 0.25,  # Increased as it's a key complexity indicator
            'chain': 0.25,  # Increased for similar reasons
            'total_deps': 0.15  # Slightly reduced
        }

        # Calculate weighted score with sigmoid normalization
        score = (dep_score * weights['dependents'] +
                 depth_score * weights['depth'] +
                 chain_score * weights['chain'] +
                 total_deps_score * weights['total_deps'])

        # Sigmoid transformation for better distribution
        score = 1 / (1 + np.exp(-10 * (score - 0.5)))

        return score * 100

    def get_risk_category(self, score):
        """
        Refined risk categories with more balanced thresholds
        """
        if score >= 85:
            return 'Critical'
        elif score >= 70:
            return 'High'
        elif score >= 50:
            return 'Medium'
        elif score >= 30:
            return 'Low'
        else:
            return 'Minimal'

    def get_risk_details(self, score, risk_factors):
        """
        Provides detailed risk analysis and recommendations
        """
        details = {
            'score': score,
            'category': self.get_risk_category(score),
            'factors': risk_factors,
            'primary_concerns': [],
            'recommendations': []
        }

        # Analyze risk factors
        if risk_factors['dependent_projects_impact'] > 50:
            details['primary_concerns'].append("High number of dependent projects")
            details['recommendations'].append("Consider splitting functionality into smaller, more focused packages")

        if risk_factors['dependency_depth_impact'] > 50:
            details['primary_concerns'].append("Deep dependency chains")
            details['recommendations'].append("Review and potentially flatten dependency structure")

        if risk_factors['chain_length_impact'] > 50:
            details['primary_concerns'].append("Long dependency chains")
            details['recommendations'].append("Identify and reduce circular or unnecessary dependencies")

        if risk_factors['total_deps_impact'] > 50:
            details['primary_concerns'].append("High total dependencies")
            details['recommendations'].append("Audit dependencies for unused or redundant inclusions")

        return details

    def analyze_library(self, df):
        # Preprocess and analyze as before...
        clean_df = self.preprocess_data(df)
        X, y, feature_names = self.prepare_features(clean_df)

        # Calculate class weights for balanced learning
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))

        # Update model with computed weights
        self.model.set_params(class_weight=class_weight_dict)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Generate detailed report
        print("\nDetailed Risk Analysis Report")
        print("=" * 50)

        # Overall statistics
        risk_stats = clean_df['risk_category'].value_counts()
        print("\nRisk Distribution:")
        for category, count in risk_stats.items():
            print(f"{category}: {count} libraries ({count / len(clean_df) * 100:.1f}%)")

        # Model performance
        print("\nModel Performance by Category:")
        print(classification_report(y_test, y_pred))

        # High-risk libraries
        high_risk = clean_df[clean_df['impact_score'] >= 70][['Library', 'impact_score', 'risk_category']]
        print("\nHigh-Risk Libraries:")
        print(high_risk)

        return {
            'model_accuracy': (y_pred == y_test).mean(),
            'risk_distribution': risk_stats.to_dict(),
            'high_risk_count': len(high_risk)
        }

    def assess_library_risk(self, dependent_projects, total_deps, avg_depth, max_chain):
        """
        Enhanced risk assessment with detailed analysis
        """
        input_data = {
            'Dependent Projects': dependent_projects,
            'Total Dependencies': total_deps,
            'avg_dependency_depth': avg_depth,
            'max_dependency_chain': max_chain
        }

        risk_factors = {
            'dependent_projects_impact': (np.log1p(dependent_projects) / np.log1p(10000)) * 100,
            'dependency_depth_impact': (np.log1p(avg_depth) / np.log1p(100)) * 100,
            'chain_length_impact': (np.log1p(max_chain) / np.log1p(200)) * 100,
            'total_deps_impact': (np.log1p(total_deps) / np.log1p(5000)) * 100
        }

        impact_score = self.calculate_impact_score(pd.Series(input_data))
        risk_details = self.get_risk_details(impact_score, risk_factors)

        return risk_details


# Main execution
if __name__ == "__main__":
    try:
        print("Loading and analyzing dependency data...")

        # Load data
        df = pd.read_csv('top50dependencies.csv')

        # Initialize analyzer
        analyzer = DependencyRiskAnalyzer()

        # Analyze example libraries
        print("\n=== Example Library Risk Assessments ===")

        # High-impact example
        high_impact = analyzer.assess_library_risk(
            dependent_projects=20000,  # Many dependent projects
            total_deps=5000,  # Many dependencies
            avg_depth=50,  # Deep dependency chains
            max_chain=150  # Long maximum chain
        )

        print("\nHigh-Impact Library Assessment:")
        print(f"Risk Score: {high_impact['score']:.1f}")
        print(f"Risk Category: {high_impact['category']}")
        print("\nPrimary Concerns:")
        for concern in high_impact['primary_concerns']:
            print(f"- {concern}")
        print("\nRecommendations:")
        for rec in high_impact['recommendations']:
            print(f"- {rec}")

        # Medium-impact example
        medium_impact = analyzer.assess_library_risk(
            dependent_projects=5000,
            total_deps=1000,
            avg_depth=20,
            max_chain=50
        )

        print("\nMedium-Impact Library Assessment:")
        print(f"Risk Score: {medium_impact['score']:.1f}")
        print(f"Risk Category: {medium_impact['category']}")
        print("\nPrimary Concerns:")
        for concern in medium_impact['primary_concerns']:
            print(f"- {concern}")
        print("\nRecommendations:")
        for rec in medium_impact['recommendations']:
            print(f"- {rec}")

        # Low-impact example
        low_impact = analyzer.assess_library_risk(
            dependent_projects=100,
            total_deps=50,
            avg_depth=5,
            max_chain=10
        )

        print("\nLow-Impact Library Assessment:")
        print(f"Risk Score: {low_impact['score']:.1f}")
        print(f"Risk Category: {low_impact['category']}")
        print("\nPrimary Concerns:")
        for concern in low_impact['primary_concerns']:
            print(f"- {concern}")
        print("\nRecommendations:")
        for rec in low_impact['recommendations']:
            print(f"- {rec}")

        # Example of assessing a specific library from your dataset
        print("\n=== Analyzing Specific Libraries from Dataset ===")

        # Get metrics for Spring Cloud Commons (known high-impact library)
        spring_assessment = analyzer.assess_library_risk(
            dependent_projects=76309,
            total_deps=1078,
            avg_depth=105.02,
            max_chain=190
        )

        print("\nSpring Cloud Commons Assessment:")
        print(f"Risk Score: {spring_assessment['score']:.1f}")
        print(f"Risk Category: {spring_assessment['category']}")
        print("\nPrimary Concerns:")
        for concern in spring_assessment['primary_concerns']:
            print(f"- {concern}")
        print("\nRecommendations:")
        for rec in spring_assessment['recommendations']:
            print(f"- {rec}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nData info:")
        print(df.info())
