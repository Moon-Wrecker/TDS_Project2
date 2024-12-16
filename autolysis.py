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
#   "jaraco.classes",
#   "uvicorn",
#   "fastapi",
#   "openai",
#   ""
# ]
# ///
from dotenv import load_dotenv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken  # Added for token counting and truncation
from fastapi import FastAPI
import openai

# Load AI Proxy token from environment variable
load_dotenv()
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

if not openai.api_key:
    raise ValueError("AIPROXY_TOKEN is missing. Check your .env file.")

# FastAPI initialization
app = FastAPI()

def analyze_data(csv_filename):
    try:
        df = pd.read_csv(csv_filename, encoding='utf-8')
    except UnicodeDecodeError:
        print("Default UTF-8 encoding failed, trying 'ISO-8859-1'.")
        df = pd.read_csv(csv_filename, encoding='ISO-8859-1')

    summary_stats = df.describe(include='all')
    missing_values = df.isnull().sum()
    correlation_matrix = df.corr(numeric_only=True)

    return df, summary_stats, missing_values, correlation_matrix

def create_visualizations(df, correlation_matrix):
    df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 6))
    df[numerical_cols].boxplot()
    plt.xticks(rotation=45)
    plt.title('Distribution of Numerical Variables')
    plt.tight_layout()
    plt.savefig('boxplots.png')
    plt.close()

    return ['histograms.png', 'correlation_heatmap.png', 'boxplots.png']

def get_high_correlations(correlation_matrix, threshold=0.7):
    high_corrs = []
    for col in correlation_matrix.columns:
        high_corrs.extend(
            [(col, idx, correlation_matrix.at[col, idx]) for idx in correlation_matrix.index
             if col != idx and abs(correlation_matrix.at[col, idx]) > threshold]
        )
    return sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True)[:5]

def summarize_data(summary_stats, missing_values, high_corrs):
    summary = {
        "Summary Statistics": summary_stats.to_dict(),
        "Missing Values": missing_values.to_dict(),
        "Top Correlations": high_corrs
    }
    return summary

def truncate_text(text, max_tokens):
    encoding = tiktoken.get_encoding("gpt-4")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text

def generate_prompt(summary):
    prompt = (
        "Analyze the following data and create a detailed narrative:\n\n"
        f"Summary Statistics: {summary['Summary Statistics']}\n\n"
        f"Missing Values: {summary['Missing Values']}\n\n"
        f"Top Correlations: {summary['Top Correlations']}\n"
    )
    return truncate_text(prompt, max_tokens=3000)

def generate_narrative(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        narrative = response["choices"][0]["message"]["content"].strip()
        return narrative
    except Exception as e:
        print(f"Error generating narrative: {str(e)}")
        return "Unable to generate a narrative due to an API error."

def iterative_prompt(summary, feedback):
    prompt = generate_prompt(summary)
    prompt += f"\nFeedback for improvement: {feedback}"
    return truncate_text(prompt, max_tokens=3000)

def main(csv_filename):
    try:
        if not os.path.exists(csv_filename):
            print(f"Error: File '{csv_filename}' not found!")
            return

        print(f"Processing file: {csv_filename}")

        df, summary_stats, missing_values, correlation_matrix = analyze_data(csv_filename)
        print("Data analysis complete")

        images = create_visualizations(df, correlation_matrix)
        print("Visualizations created")

        high_corrs = get_high_correlations(correlation_matrix)
        summary = summarize_data(summary_stats, missing_values, high_corrs)

        prompt = generate_prompt(summary)
        narrative = generate_narrative(prompt)

        with open('README.md', 'w', encoding='utf-8') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write("## Summary of Analysis\n\n")
            f.write(f"### Data Columns: {', '.join(df.columns)}\n\n")
            f.write(f"### Summary Statistics:\n```{summary_stats}\n``\n\n")
            f.write(f"### Missing Values:\n```{missing_values}\n``\n\n")
            f.write(f"### Correlation Matrix:\n```{correlation_matrix}\n``\n\n")
            f.write("## Data Insights\n\n")
            f.write(f"{narrative}\n\n")
            f.write("## Visualizations\n\n")
            f.write("### Histograms\n")
            f.write("![Histograms](./histograms.png)\n\n")
            f.write("### Correlation Heatmap\n")
            f.write("![Correlation Heatmap](./correlation_heatmap.png)\n\n")
            f.write("### Box Plots\n")
            f.write("![Box Plots](./boxplots.png)\n\n")

        print("README.md created successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_filename>")
    else:
        csv_filename = sys.argv[1]
        main(csv_filename)