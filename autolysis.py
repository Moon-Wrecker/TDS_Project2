# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "numpy",
#   "python-dotenv",
#   "scikit-learn",
#   "openai",
#   "scipy"
# ]
# ///

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import base64
import time
from typing import Dict, List, Tuple, Any

# Configure OpenAI settings
load_dotenv()
api_key = os.getenv("AIPROXY_TOKEN")
if not api_key:
    raise ValueError("AIPROXY_TOKEN is missing. Check your .env file.")

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

def chunked_api_call(messages: List[Dict], max_retries: int = 3, initial_wait: int = 4):
    """Make API calls with retries and chunked data."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                timeout=30  # Add timeout to prevent hanging
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Final attempt failed: {str(e)}")
                return "Analysis could not be completed due to API limitations."
            wait_time = initial_wait * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

class DataAnalyzer:
    """Advanced data analysis with statistical testing and ML insights."""
    
    def __init__(self, csv_filename: str):
        self.csv_filename = csv_filename
        self.df = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.correlation_matrix = None
        self.summary_stats = None
        self.missing_values = None
        self.statistical_tests = {}
        self.clusters = None
        
        # Create output directory based on dataset name
        self.dataset_name = os.path.splitext(os.path.basename(csv_filename))[0]
        os.makedirs(self.dataset_name, exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare data with enhanced error handling."""
        try:
            self.df = pd.read_csv(self.csv_filename, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_filename, encoding='ISO-8859-1')
        
        # Identify column types
        self.numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Handle missing values and outliers
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self._handle_missing_values()
        
    def _handle_missing_values(self):
        """Handle missing values intelligently."""
        for col in self.numeric_cols:
            if self.df[col].isnull().sum() > 0:
                # Use median for skewed distributions, mean for normal
                if abs(stats.skew(self.df[col].dropna())) > 1:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    
    def perform_analysis(self):
        """Comprehensive data analysis with statistical tests."""
        # Basic statistics
        self.summary_stats = self.df.describe(include='all')
        self.missing_values = self.df.isnull().sum()
        
        # Enhanced correlation analysis
        if len(self.numeric_cols) > 1:
            self.correlation_matrix = self.df[self.numeric_cols].corr()
            
            # Perform statistical tests
            self._perform_statistical_tests()
            
            # Perform clustering if enough numeric columns
            if len(self.numeric_cols) >= 2:
                self._perform_clustering()
                
    def _perform_statistical_tests(self):
        """Perform various statistical tests on the data."""
        for col in self.numeric_cols:
            # Normality test
            stat, p_value = stats.normaltest(self.df[col].dropna())
            self.statistical_tests[f"{col}_normality"] = {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05
            }
            
            # Outlier detection using Z-score
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = np.sum(z_scores > 3)
            self.statistical_tests[f"{col}_outliers"] = outliers
            
    def _perform_clustering(self):
        """Perform clustering on numeric data."""
        # Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[self.numeric_cols])
        
        # Determine optimal number of clusters (max 5)
        inertias = []
        max_clusters = min(5, len(self.df) // 2)
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
            
        # Use elbow method to find optimal clusters
        optimal_clusters = 2  # default
        for i in range(1, len(inertias) - 1):
            if (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]) < 2:
                optimal_clusters = i + 1
                break
                
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(scaled_data)
        
    def create_visualizations(self):
        """Generate enhanced visualizations with insights."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        
        self._create_distribution_plots()
        self._create_correlation_heatmap()
        self._create_missing_values_plot()
        if self.clusters is not None:
            self._create_cluster_plot()
        
        return ['distributions.png', 'correlation_heatmap.png', 'missing_values.png', 'clusters.png']
        
    def _create_distribution_plots(self):
        """Create enhanced distribution plots."""
        if len(self.numeric_cols) > 0:
            fig = plt.figure(figsize=(15, 10))
            for idx, col in enumerate(self.numeric_cols[:5]):
                plt.subplot(2, 3, idx+1)
                sns.histplot(self.df[col], kde=True)
                plt.title(f'{col} Distribution')
                if col in self.statistical_tests:
                    is_normal = self.statistical_tests[f"{col}_normality"]["is_normal"]
                    plt.title(f'{col} Distribution\n{"Normal" if is_normal else "Non-normal"}')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.dataset_name, 'distributions.png'))
            plt.close()
            
    def _create_correlation_heatmap(self):
        """Create enhanced correlation heatmap."""
        if self.correlation_matrix is not None:
            plt.figure(figsize=(12, 8))
            mask = np.triu(np.ones_like(self.correlation_matrix))
            sns.heatmap(self.correlation_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       fmt='.2f',
                       linewidths=0.5)
            plt.title('Correlation Matrix Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dataset_name, 'correlation_heatmap.png'))
            plt.close()
            
    def _create_missing_values_plot(self):
        """Create missing values visualization."""
        plt.figure(figsize=(12, 6))
        missing_percentages = (self.missing_values / len(self.df)) * 100
        sns.barplot(x=missing_percentages.index, 
                   y=missing_percentages.values)
        plt.title('Missing Values by Column (%)')
        plt.xticks(rotation=45)
        plt.ylabel('Percentage Missing')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_name, 'missing_values.png'))
        plt.close()
        
    def _create_cluster_plot(self):
        """Create cluster visualization."""
        if self.clusters is not None and len(self.numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                self.df[self.numeric_cols[0]],
                self.df[self.numeric_cols[1]],
                c=self.clusters,
                cmap='viridis'
            )
            plt.colorbar(scatter)
            plt.title('Cluster Analysis')
            plt.xlabel(self.numeric_cols[0])
            plt.ylabel(self.numeric_cols[1])
            plt.tight_layout()
            plt.savefig(os.path.join(self.dataset_name, 'clusters.png'))
            plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report with dynamic insights."""
        try:
            self.load_and_prepare_data()
            self.perform_analysis()
            image_files = self.create_visualizations()
            
            # Prepare analysis context with chunked data
            context = {
                "dataset_info": {
                    "name": self.dataset_name,
                    "rows": len(self.df),
                    "columns": len(self.df.columns),
                    "numeric_columns": list(self.numeric_cols),
                    "categorical_columns": list(self.categorical_cols)
                },
                "statistical_tests": self.statistical_tests,
                "missing_values": self.missing_values.to_dict()
            }
            
            # Generate insights with optimized prompts
            messages = [
                {"role": "system", "content": """You are a data scientist. Provide clear, specific insights.
                Focus on actionable findings and their implications. Use clear Markdown formatting."""},
                {"role": "user", "content": f"Analyze this data context and provide key insights:\n{json.dumps(context)}"}
            ]
            analysis = chunked_api_call(messages)
            
            # Write the report
            readme_path = os.path.join(self.dataset_name, 'README.md')
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# Analysis Report: {self.dataset_name}\n\n")
                f.write("## Key Findings\n\n")
                f.write(analysis + "\n\n")
                
                if self.statistical_tests:
                    f.write("## Statistical Analysis\n\n")
                    for test, result in self.statistical_tests.items():
                        if isinstance(result, dict):
                            f.write(f"### {test}\n")
                            f.write(f"- Result: {'Normal' if result['is_normal'] else 'Non-normal'}\n")
                            f.write(f"- P-value: {result['p_value']:.4f}\n\n")
                
                f.write("## Visualizations\n\n")
                for image in image_files:
                    if os.path.exists(os.path.join(self.dataset_name, image)):
                        f.write(f"### {image.split('.')[0].title()}\n")
                        f.write(f"![{image}](./{image})\n\n")
                
            print(f"Analysis complete! Results saved to {readme_path}")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

def main(csv_filename: str):
    """Main function to run the analysis."""
    analyzer = DataAnalyzer(csv_filename)
    analyzer.generate_report()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <csv_filename>")
    else:
        main(sys.argv[1])