# Price Suggestion Online Selling
This project is to build a regression model to predict the sale price of a listing based on information a user provides for this listing. There are both categorical (brand, category of the item, free shipping or not) and textual data (title, item description), therefore natrual language processing techniques should be used. By using the extracted features, a regression model containing embedding and GRU layers is built via Keras and it achieved a rmsle score of 0.52.

## Introduction to the Dataset
The data is available here https://www.kaggle.com/c/mercari-price-suggestion-challenge/data. The files consist of a list of product listings. These files are tab-delimited.

* `train_id` or `test_id` - the id of the listing
* `name` - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]
* `item_condition_id` - the condition of the items provided by the seller
* `category_name` - category of the listing
* `brand_name` - the brand of the item
* `price` - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict.
* `shipping` - 1 if shipping fee is paid by seller and 0 by buyer
* `item_description` - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]

## Functions of Each File
* `print_df_info.py` prints the information of the dataframe.
* `ps_load_data.py` loads the data into dataframe.
* `Price_suggestion_EDA.py` the EDA of the dataset.
* `main.py` is the main function. It performs data importing, cleaning, generating features (including NLP processing), building regression models and measuring the performance.


## Possible Further Improvements
* Try with more dense layers or more rnn outputs.
* Try to increase the embedding factors.

## References
Thanks to the authors below who provide excellent kernels:

https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling

https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
