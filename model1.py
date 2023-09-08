from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import average_precision_score
from data.preprocessing.preprocess import preprocessing_features, preprocessing_labels, column_selection
from code.utils.configs import load_config, load_data_config
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main(path_data_config, path_model_config) : 

    load_dotenv()

    # Loading configs
    model_config = load_config(path_model_config)
    data_config = load_data_config(path_data_config)

    # Load the dataset
    X_train_df, _  = preprocessing_features(os.path.join(data_config["path_data"], "X_train.csv"))
    y_train_df = preprocessing_labels(os.path.join(data_config["path_data"], "Y_train.csv"))
    X_train_df = column_selection(X_train_df)

    class_weight = {0.: model_config["class_weight_0"], 1.: model_config["class_weight_1"]}

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_df, y_train_df, test_size=data_config["test_size"], 
        random_state=model_config["random_state"])

    # Loading models
    classifier_rf = RandomForestClassifier(random_state=model_config["random_state"], 
                                        n_estimators=model_config["n_estimators_rf"], class_weight=class_weight)
    classifier_ada = AdaBoostClassifier(random_state=model_config["random_state"],
                                        n_estimators=model_config["n_estimators_ada"],
                                        learning_rate=model_config["learning_rate_ada"])
    
    # Fitting models
    classifier_rf.fit(X_train_df, y_train_df)
    rf_predicted_probs = classifier_rf.predict_proba(X_train_df)
    classifier_ada.fit(rf_predicted_probs, y_train_df)

    # Evaluating models
    rf_predicted_probs = classifier_rf.predict_proba(X_val)
    y_pred_proba = classifier_ada.predict_proba(rf_predicted_probs)[:,1]
    y_pred_proba = pd.DataFrame(y_pred_proba).astype(float)
    print("PR-AUC on val set: {:.3f}".format(average_precision_score(y_val, y_pred_proba)*100))

if __name__ == "__main__" : 

    # Parser
    parser = argparse.ArgumentParser(description="Parser for training the model")

    parser.add_argument("-d","--path_data_config", type=str, default="configs\data_config.yaml", help="Path to the data config")
    parser.add_argument("-m","--path_model_config", type=str, default="configs\model_config.yaml", help="Path to the model config")

    args = parser.parse_args()

    main(args.path_data_config, args.path_model_config)