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
#   "tenacity",
#   "scikit-learn"
# ]
# ///

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import json
from dotenv import load_dotenv
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import base64

# Configure OpenAI settings
load_dotenv()
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

if not openai.api_key:
    raise ValueError("AIPROXY_TOKEN is missing. Check your .env file.")

class DataAnalyzer:
    """Main class for analyzing datasets and generating insights."""
    
    def __init__(self, csv_filename):
        """Initialize with the CSV filename and prepare for analysis."""
        self.csv_filename = csv_filename
        self.df = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.correlation_matrix = None
        self.summary_stats = None
        self.missing_values = None
        self.outliers = None
        
    def load_and_prepare_data(self):
        """Load the CSV file and prepare data for analysis."""
        try:
            self.df = pd.read_csv(self.csv_filename, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_filename, encoding='ISO-8859-1')
            
        # Identify column types
        self.numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Basic cleaning
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
    def perform_analysis(self):
        """Perform comprehensive data analysis."""
        # Basic statistics
        self.summary_stats = self.df.describe(include='all')
        self.missing_values = self.df.isnull().sum()
        
        # Correlation analysis for numeric columns
        if len(self.numeric_cols) > 0:
            self.correlation_matrix = self.df[self.numeric_cols].corr()
            
        # Detect outliers using IQR method
        self.outliers = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(self.df[(self.df[col] < (Q1 - 1.5 * IQR)) | 
                                      (self.df[col] > (Q3 + 1.5 * IQR))])
            self.outliers[col] = outlier_count
            
    def create_visualizations(self):
        """Generate insightful visualizations."""
        # Set style for all plots
        plt.style.use('seaborn')
        
        # 1. Distribution plots for numeric columns
        self._create_distribution_plots()
        
        # 2. Correlation heatmap
        self._create_correlation_heatmap()
        
        # 3. Missing values visualization
        self._create_missing_values_plot()
        
        return ['distributions.png', 'correlation_heatmap.png', 'missing_values.png']
        
    def _create_distribution_plots(self):
        """Create distribution plots for numeric columns."""
        if len(self.numeric_cols) > 0:
            fig, axes = plt.subplots(figsize=(15, 10))
            for col in self.numeric_cols[:5]:  # Limit to first 5 columns
                sns.kdeplot(data=self.df[col], label=col)
            
            plt.title('Distribution of Numeric Variables', pad=20)
            plt.xlabel('Values')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig('distributions.png')
            plt.close()
            
    def _create_correlation_heatmap(self):
        """Create correlation heatmap for numeric columns."""
        if self.correlation_matrix is not None:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       fmt='.2f',
                       linewidths=0.5)
            plt.title('Correlation Matrix Heatmap', pad=20)
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png')
            plt.close()
            
    def _create_missing_values_plot(self):
        """Create missing values visualization."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=self.missing_values.index, 
                   y=self.missing_values.values)
        plt.title('Missing Values by Column', pad=20)
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('missing_values.png')
        plt.close()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_initial_insights(self):
        """Generate initial insights about the dataset using GPT-4."""
        context = {
            "dataset_name": os.path.basename(self.csv_filename),
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "numeric_columns": list(self.numeric_cols),
            "categorical_columns": list(self.categorical_cols),
            "missing_values_summary": self.missing_values.to_dict(),
            "outliers_summary": self.outliers
        }
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are a data scientist analyzing a dataset. 
                 Focus on key insights, patterns, and potential areas for deeper analysis.
                 Structure your response in Markdown format with clear sections."""},
                {"role": "user", "content": f"Analyze this dataset overview and provide initial insights:\n{json.dumps(context, indent=2)}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_correlations(self):
        """Analyze correlations and generate insights."""
        if self.correlation_matrix is None:
            return "No numeric columns available for correlation analysis."
            
        # Find strongest correlations
        correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                col1 = self.correlation_matrix.columns[i]
                col2 = self.correlation_matrix.columns[j]
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) > 0.5:  # Only strong correlations
                    correlations.append((col1, col2, corr))
                    
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data scientist analyzing correlations."},
                {"role": "user", "content": f"Analyze these correlations and suggest potential implications:\n{json.dumps(correlations, indent=2)}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_visualizations(self):
        """Analyze the generated visualizations using GPT-4 Vision."""
        insights = []
        
        for image_path in ['distributions.png', 'correlation_heatmap.png', 'missing_values.png']:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert."},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Analyze this {image_path} visualization and provide key insights:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low"
                            }}
                        ]}
                    ],
                    temperature=0.7
                )
                insights.append(response.choices[0].message.content)
                
        return "\n\n".join(insights)

    def generate_report(self):
        """Generate the final analysis report."""
        try:
            # Load and analyze data
            self.load_and_prepare_data()
            self.perform_analysis()
            
            # Generate visualizations
            image_files = self.create_visualizations()
            
            # Generate insights using multiple approaches
            initial_insights = self.generate_initial_insights()
            correlation_insights = self.analyze_correlations()
            visual_insights = self.analyze_visualizations()
            
            # Compile the report
            with open('README.md', 'w', encoding='utf-8') as f:
                f.write("# Automated Data Analysis Report\n\n")
                
                f.write("## Dataset Overview\n\n")
                f.write(initial_insights + "\n\n")
                
                f.write("## Correlation Analysis\n\n")
                f.write(correlation_insights + "\n\n")
                
                f.write("## Visual Analysis\n\n")
                for image in image_files:
                    f.write(f"### {image.split('.')[0].title()}\n")
                    f.write(f"![{image}](./{image})\n\n")
                
                f.write("## Visual Insights\n\n")
                f.write(visual_insights + "\n\n")
                
            print("Analysis complete! Results saved to README.md")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

def main(csv_filename):
    """Main function to run the analysis."""
    analyzer = DataAnalyzer(csv_filename)
    analyzer.generate_report()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <csv_filename>")
    else:
        main(sys.argv[1])