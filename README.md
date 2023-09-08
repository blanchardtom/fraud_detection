This project aims to predict scammers on shopping websites for BNP Paribas. 

It contains 2 models : 
- model1.py using random tokenisation of the features with RandomForestClassifier and AdaboostClassifier sequentially
- model2.py using trained tokenisation with KerasTokenizer and RandomForestClassifier in a pipeline

Both use sklearn models to ensure prediction. 

A first glance into the data structure is possible running data\vizualisation.py

To train the model "model1.py" : 
```python model1.py -d configs\data_config.yaml -m configs\model_config.yaml```

To train the model "model2.py" : 
```python model2.py -d configs\data_config.yaml -m configs\model_config.yaml```