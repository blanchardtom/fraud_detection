from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from code.tokenizer.tokenizer import KerasTokenizer
from data.preprocessing.preprocess import preprocessing
import argparse
from dotenv import load_dotenv
from code.utils.configs import load_config, load_data_config

def main(path_data_config, path_model_config) : 

    load_dotenv()

    model_config = load_config(path_model_config)
    data_config = load_data_config(path_data_config)
    # Load the dataset
    X_train_df, y_train_df, categorical_columns, numerical_columns = preprocessing(data_config["path_data"])


    # Define transformers
    cat_pipeline = make_pipeline(KerasTokenizer(num_words=model_config["num_words"]))
    num_pipeline = make_pipeline(StandardScaler())

    # Create the preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('cat_pipeline', cat_pipeline, categorical_columns),
        ('num_pipeline', num_pipeline, numerical_columns)
    ])

    # Define model
    model_rf = RandomForestClassifier(random_state=model_config["random_state"],
                                      n_estimators=model_config["n_estimators_rf"])

    # Create a custom scoring function using average_precision_score
    scoring = make_scorer(average_precision_score)

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_rf)
    ])

    # Split the data into training and validation sets
    _, X_val, _, y_val = train_test_split(
        X_train_df, y_train_df, test_size=data_config["test_size"], 
        random_state=model_config["random_state"])

    rf_pipeline.fit(X_train_df, y_train_df)

    # Evaluate the pipeline using average_precision_score
    y_pred = rf_pipeline.predict(X_val)
    print("Average precision score: ")
    print(average_precision_score(y_val, y_pred)*100)

if __name__ == "__main__" : 

    # Parser
    parser = argparse.ArgumentParser(description="Parser for training the model")

    parser.add_argument("-d","--path_data_config", type=str, default="configs\data_config.yaml", help="Path to the data config")
    parser.add_argument("-m","--path_model_config", type=str, default="configs\model_config.yaml", help="Path to the model config")

    args = parser.parse_args()

    main(args.path_data_config, args.path_model_config)