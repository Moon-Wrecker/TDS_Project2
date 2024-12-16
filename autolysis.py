from dotenv import load_dotenv
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import uvicorn
from fastapi import FastAPI
import openai
from dotenv import load_dotenv

# Load AI Proxy token from environment variable
load_dotenv()
openai.api_key = os.getenv("AIPROXY_TOKEN")

# Set the API base URL for the proxy
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

if not openai.api_key:
    raise ValueError("AIPROXY_TOKEN is missing. Check your .env file.")

# FastAPI initialization
app = FastAPI()

def analyze_data(csv_filename):
    # Read dataset with a specified encoding
    try:
        df = pd.read_csv(csv_filename, encoding='utf-8')  # Default attempt with utf-8
    except UnicodeDecodeError:
        print("Default UTF-8 encoding failed, trying 'ISO-8859-1'.")
        df = pd.read_csv(csv_filename, encoding='ISO-8859-1')  # Fallback to ISO-8859-1

    # Generate basic statistics
    summary_stats = df.describe(include='all')
    missing_values = df.isnull().sum()

    # Correlation matrix
    correlation_matrix = df.corr(numeric_only=True)

    return df, summary_stats, missing_values, correlation_matrix


# Function to generate visualizations
def create_visualizations(df, correlation_matrix):
    # 1. Histogram of each column
    df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close()  # Close the figure to free memory
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()  # Close the figure to free memory

    # 3. Box plots for numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 6))
    df[numerical_cols].boxplot()
    plt.xticks(rotation=45)
    plt.title('Distribution of Numerical Variables')
    plt.tight_layout()
    plt.savefig('boxplots.png')
    plt.close()  # Close the figure to free memory

    return ['histograms.png', 'correlation_heatmap.png', 'boxplots.png']

def generate_narrative(df, summary_stats, missing_values, correlation_matrix):
    # Prepare context for the LLM
    data_context = {
        "columns": list(df.columns),
        "summary_stats": summary_stats.to_dict(),
        "missing_values": missing_values.to_dict(),
        "correlation_matrix": correlation_matrix.to_dict()
    }
    
    # Debugging: Print the context being sent
    print("Data context being sent to OpenAI API:")
    #print(data_context)

    try:
        # Call the LLM for a narrative
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": f"Analyze the following data and create a detailed story explaining the insights:\n\n"
                                             f"Data columns: {data_context['columns']}\n"
                                             f"Summary Stats: {data_context['summary_stats']}\n"
                                             f"Missing Values: {data_context['missing_values']}\n"
                                             f"Correlation Matrix: {data_context['correlation_matrix']}"}
            ],
            max_tokens=500,
            temperature=0.7  # Optional: Adjust creativity level
        )

        # Extract and return the narrative
        narrative = response["choices"][0]["message"]["content"].strip()
        return narrative

    except Exception as e:
        # Log error and return a fallback message
        print(f"Error generating narrative: {str(e)}")
        return "Unable to generate a narrative due to an API error."


def main(csv_filename):
    try:
        # Verify file exists
        if not os.path.exists(csv_filename):
            print(f"Error: File '{csv_filename}' not found!")
            return

        print(f"Processing file: {csv_filename}")  # Debug print
        
        # Analyze data
        df, summary_stats, missing_values, correlation_matrix = analyze_data(csv_filename)
        print("Data analysis complete")  # Debug print

        # Create visualizations
        images = create_visualizations(df, correlation_matrix)
        print("Visualizations created")  # Debug print

        # Generate narrative
        try:
            narrative = generate_narrative(df, summary_stats, missing_values, correlation_matrix)
            print("Narrative generated")  # Debug print
        except Exception as e:
            print(f"Error generating narrative: {str(e)}")
            narrative = "Error generating narrative"

        # Write results to Markdown file
        print("Attempting to create README.md...")  # Debug print
        with open('README.md', 'w', encoding='utf-8') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write("## Summary of Analysis\n\n")
            f.write(f"### Data Columns: {', '.join(df.columns)}\n\n")
            f.write(f"### Summary Statistics:\n```\n{summary_stats}\n```\n\n")
            f.write(f"### Missing Values:\n```\n{missing_values}\n```\n\n")
            f.write(f"### Correlation Matrix:\n```\n{correlation_matrix}\n```\n\n")
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
# Run the script with a given dataset

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py happiness.csv")
    else:
        csv_filename = sys.argv[1]
        main(csv_filename)

