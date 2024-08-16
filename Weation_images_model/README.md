# Weather Data Analysis

This repository provides a framework for working with the weather dataset from Kaggle. The dataset can be used to build machine learning models for predicting weather conditions and analyzing historical weather data.

## Dataset

The dataset can be accessed from Kaggle at the following link: [Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset). You can either download the dataset manually or use the Kaggle API to import it directly into your Google Colab environment.

## Prerequisites

- **Kaggle Account**: To access the dataset, you'll need a Kaggle account. If you don't have one, you can create it [here](https://www.kaggle.com/account/login).
- **Google Colab**: This guide assumes you're using Google Colab to run your code.

## Using the Dataset

### Option 1: Download the Dataset Manually

1. Visit the [Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) page.
2. Click on the "Download" button to download the dataset to your local machine.
3. Upload the downloaded dataset to your Google Colab environment using the following code:

    ```python
    from google.colab import files
    uploaded = files.upload()
    ```

4. Once uploaded, you can load the dataset into your notebook:

    ```python
    import pandas as pd

    df = pd.read_csv('weatherHistory.csv')
    ```

### Option 2: Import the Dataset Directly from Kaggle

1. **Create a Kaggle API Token**:
    - Log in to your Kaggle account.
    - Go to your account settings by clicking on your profile picture in the top right corner and selecting "Account".
    - Scroll down to the "API" section and click on "Create New API Token". This will download a `kaggle.json` file containing your API credentials.

2. **Upload the Kaggle API Token to Google Colab**:
    - In your Google Colab notebook, run the following code to upload the `kaggle.json` file:

    ```python
    from google.colab import files
    files.upload()
    ```

3. **Set Up Kaggle API**:
    - After uploading the `kaggle.json` file, run the following commands to set up the Kaggle API:

    ```python
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```

4. **Download the Dataset Using Kaggle API**:
    - Use the following Kaggle API command to download the dataset:

    ```python
    !kaggle datasets download -d jehanbhathena/weather-dataset
    ```

5. **Unzip the Dataset**:
    - Once downloaded, unzip the dataset:

    ```python
    !unzip weather-dataset.zip
    ```

6. **Load the Dataset**:
    - Finally, load the dataset into your notebook:

    ```python
    import pandas as pd

    df = pd.read_csv('weatherHistory.csv')
    ```

## Usage

Once the dataset is loaded into your environment, you can proceed with data analysis, preprocessing, and model building. This repository includes scripts and notebooks to help you get started with various weather-related projects.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request.
