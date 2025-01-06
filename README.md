# Iris Project 🌸

Welcome to the Iris Project! ✨ This project is a part of the **Statistical Inference Methods** course (Politecnico di Milano, year 2022/2023), where we dive into the iconic *Iris dataset* to uncover insights using various statistical techniques. Along the way, you'll see data preprocessing, stunning visualizations, linear regression, logistic regression for classification, and how to evaluate our models. Let's get into the fun of analyzing data!

![Iris Flower](iris-data-pic.jpg)

## Project Structure 📁

Here’s how everything is organized in the project:

- `data`: Contains the Iris dataset (iris.data.txt) and any other essential data files used in the analysis.

- `anaconda_projects`: This folder holds the following Jupyter notebooks:

    - `01_data_preprocessing.ipynb`: Clean, transform, and prepare the data for analysis.
    - `02_visualization.ipynb`: Eye-catching visualizations to explore the data and reveal hidden patterns.
    - `03_linear_model.ipynb`: Build and assess a linear regression model for prediction.
    - `04_logistic_regression.ipynb`: Create a logistic regression model to classify species.
    - `05_prediction_analysis.ipynb`: Evaluate the performance of the models on test data.

- `results`:

    - `plots`: Contains graphical representations and plots generated during the analysis, including:

        - 02-class vs ...
        - 03-qqplot-m5.png
        - 03-residuals-m2.png
        - 03-standardized residuals-m5.png
        - 04-residuals-log.png
        - 05-ROC.png
        - 05-confusionmatrix.png

        Additional plots like correlation matrix and pair plots.
    - `tables`: Contains output files with results from the analysis, such as model summaries and statistical tests (e.g., Shapiro-Wilk test), saved in text format. Examples: m1.txt, m2.txt, m3.txt, ..., m8.txt.
    
- `README.md`: Provides an overview of the project, including its purpose, methodology, and how to run the analysis.


## What You’ll Need 🔧

To run this project and explore the Jupyter notebooks with an R kernel, you’ll need to have a few tools installed on your system:

### Software Dependencies

1. **R**: The R programming language must be installed. Get it from the [official R website](https://cran.r-project.org/mirrors.html).
2. **Jupyter Notebook**: To run the notebooks, you'll need Jupyter Notebook with the R Kernel.

You can install Jupyter using pip:

```
pip install notebook
```

Then, install the R kernel for Jupyter by following the steps here:  
https://irkernel.github.io/installation/

### Cloning the Repository

Ready to dive in? Start by cloning the repository to your local machine with this command:

```
git clone https://github.com/elisabettatea/iris-project.git
```


Once you've cloned the repository, navigate to the project folder:

```
cd iris-project
```

### Running Jupyter Notebook

To run the notebooks, just start the Jupyter Notebook server by running:

```
jupyter notebook
```

This will open the Jupyter Notebook interface in your browser, and you're ready to explore the data and models!

## How to Use the Notebooks 📝

Each notebook is packed with short explanations of the code and the results. To make sure things run smoothly, follow this order:

1. **`01_data_preprocessing.ipynb`**: Clean, transform, and prepare the dataset.
2. **`02_visualization.ipynb`**: Visualize the dataset and explore relationships between features.
3. **`03_linear_model.ipynb`**: Build a linear regression model to predict based on the data.
4. **`04_logistic_regression.ipynb`**: Perform binary classification using logistic regression (guess who the Iris-versicolor is!).
5. **`05_prediction_analysis.ipynb`**: Evaluate how well our model performs on test data.

## What You’ll Learn 💡

In this project, we’ll take a deep dive into the Iris dataset using the following techniques:

- **Data Preprocessing**: Clean up the dataset, handle missing values, and ensure the data is in the right shape for modeling.
- **Data Visualization**: Create beautiful visualizations (think scatter plots, pair plots) to uncover hidden patterns.
- **Linear Regression**: Build a linear model to understand the relationship between the features and predict outcomes.
- **Logistic Regression**: Apply logistic regression for binary classification (Is it Iris-versicolor or not?).
- **Model Evaluation**: Evaluate how well our models are doing using various statistical tests and performance metrics.

## Results 📊

Here’s a sneak peek at the results:

- **Linear Regression**: We explore how sepal length, width, and petal dimensions relate to predicting Iris species.
- **Logistic Regression**: We fine-tune a logistic regression model to classify the species, focusing on Iris-versicolor vs. others.
- **Model Performance**: We evaluate the predictions using statistical tests, ensuring that our models perform well and make reliable predictions.

## Questions? 

If you have any questions, feel free to reach out! You can contact me at:  

✉️ **elisabetta.tea@gmail.com**

Happy coding and analyzing! ✨
