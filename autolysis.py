# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "openai",
#     "python-dotenv",
#     "tenacity"
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

if not openai.api_key:
    raise ValueError("AIPROXY_TOKEN is missing. Please set it in your environment variables.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_data(df):
    """Perform basic statistical analysis on the dataset"""
    analysis = {
        'basic_stats': df.describe(include='all'),
        'missing_values': df.isnull().sum(),
        'dtypes': df.dtypes,
        'shape': df.shape,
        'columns': list(df.columns)
    }
    
    # Calculate correlations for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 1:
        analysis['correlation'] = df[numerical_cols].corr()
    
    return analysis

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_visualizations(df, analysis):
    """Generate visualizations based on the data"""
    plt.style.use('seaborn')
    visualizations = []
    
    # 1. Distribution plots for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, 6))
        df[numerical_cols].hist(bins=30)
        plt.tight_layout()
        plt.savefig('distributions.png')
        plt.close()
        visualizations.append('distributions.png')

    # 2. Correlation heatmap
    if 'correlation' in analysis:
        plt.figure(figsize=(10, 8))
        sns.heatmap(analysis['correlation'], annot=True, cmap='coolwarm', center=0)
        plt.tight_layout()
        plt.savefig('correlation.png')
        plt.close()
        visualizations.append('correlation.png')

    # 3. Missing values plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=analysis['missing_values'].index, y=analysis['missing_values'].values)
    plt.xticks(rotation=45)
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.savefig('missing_values.png')
    plt.close()
    visualizations.append('missing_values.png')

    return visualizations

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_narrative(analysis, visualizations):
    """Generate a narrative using the LLM"""
    prompt = f"""
    Analyze this dataset and create a detailed story. Include these sections:
    1. Data Overview
    2. Key Findings
    3. Insights
    4. Recommendations

    Dataset Information:
    - Shape: {analysis['shape']}
    - Columns: {analysis['columns']}
    - Basic Statistics: {analysis['basic_stats'].to_dict()}
    - Missing Values: {analysis['missing_values'].to_dict()}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data analyst creating insightful reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def main(csv_filename):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filename)
        
        # Perform analysis
        analysis = analyze_data(df)
        
        # Create visualizations
        visualizations = create_visualizations(df, analysis)
        
        # Generate narrative
        narrative = generate_narrative(analysis, visualizations)
        
        # Create README.md
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write(narrative + "\n\n")
            f.write("## Visualizations\n\n")
            for viz in visualizations:
                f.write(f"![{viz.replace('.png', '')}](./{viz})\n\n")
            
        print(f"Analysis complete. Results saved in README.md and visualization files.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_filename>")
    else:
        main(sys.argv[1])