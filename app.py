import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load data (replace with your dataset path)
data_path = 'your_dataset.csv'
data = pd.read_csv(data_path)

@app.route('/')
def home():
    return "Welcome to the Data Analysis API!"

@app.route('/summary', methods=['GET'])
def summary():
    return data.describe().to_json()

@app.route('/missing', methods=['GET'])
def missing_values():
    missing = data.isnull().sum().to_dict()
    return jsonify(missing)

@app.route('/correlation', methods=['GET'])
def correlation():
    correlation_matrix = data.corr()
    return correlation_matrix.to_json()

@app.route('/plot/<plot_type>', methods=['GET'])
def plot(plot_type):
    column1 = request.args.get('column1')
    column2 = request.args.get('column2', None)

    if plot_type == 'histogram' and column1:
        plt.hist(data[column1].dropna(), bins=20, color='blue', alpha=0.7)
        plt.title(f'Histogram of {column1}')
        plt.savefig('histogram.png')
        return "Histogram saved as 'histogram.png'"

    elif plot_type == 'scatter' and column1 and column2:
        plt.scatter(data[column1], data[column2], alpha=0.7)
        plt.title(f'Scatter plot of {column1} vs {column2}')
        plt.savefig('scatter.png')
        return "Scatter plot saved as 'scatter.png'"

    elif plot_type == 'heatmap':
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig('heatmap.png')
        return "Heatmap saved as 'heatmap.png'"

    else:
        return "Invalid plot type or parameters!", 400

if __name__ == '__main__':
    app.run(debug=True)
