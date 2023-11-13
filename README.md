# End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC
## Directory Structure
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/172dfcb0-d9ba-4e5b-851f-35c5b1f84074)

## Creating Python Environment
```
python -m venv dvcdemoenv
```

## Activasting Environment
```
.\dvcdemoenv\Scripts\activate
```

## Installing Libraries
```
pip install pandas pyaml scikit-learn scipy matplotlib
```

## Installing DVC and DVCLIVE
```
pip install dvc
pip install dvclive>=2.0
```

## Pipeline
Pipeline Steps:
1. Data Split: Splitting the data into training and testing dataset. Here we use winequality dataset. Here we use randomforest model. We read the parameters like split ratio, random state, data source path from params.yaml file. </br>
We can also do data versoning using gs bucket, s3 bucket or local machine. </br>
We define the parameters in params.yaml file so that we dont hard code any paramaters within our code. If we want to change any parameter we can change that here and we can simply run the pipeline. </br>

2. Data Processing: We give the data path for the data processing function as shown. Here we can pass training and testing data individually. This process will not have the problem of data leakage as we are not doing the whole data processing together before train test split. This process will generate the final train and test data files which will be kept inside process directory. </br>

3. Train: All the parameters are defined in params.yaml file. We are using random forest model for training and testing the model with number of estimators as 10. Finally we dump the pickle model file in the model_dir folder. </br>

4. Evaluate: Here we generating the metrics json file which will have all the ROC and ASUC scores and appropiate plots. Here we read paramaters file, model, data, data category: whether we are executing the performance of the model in the training data or the testing data. Using live we can log everything. Here we are recording average precision score, roc auc scores and precision recall curves. Finally we dunp a json file which will contain the values of precision, recall and threshold. We also plot the confusion matrix. All the evaluation output will be defined in EVAL path for train and test data seperately. Then we dump the feature importance image in EVAL path folder </br>

## Creating Data Pipeline
### Creating Pipeline stages
```
dvc stage add -n data_split -p data_source.local_path,base.random_state,split.split_ratio -d src/data_split.py -o data/split python src/data_split.py


dvc stage add -n data_processing -p split.dir,split.train_file,split.test_file,process.dir,process.train_file,process.test_file -d src/data_processing.py -d data/split -o data/processed python src/data_processing.py data/processed


dvc stage add -n train -p process.dir,process.train_file,process.test_file,base.random_state,base.target_col,train.n_est,model_dir -d src/train.py -d data/processed -o model/model.pkl python src/train.py data/features model/model.pkl


dvc stage add -n evaluate -d src/evaluate.py -d model/model.pkl -d data/processed -M eval/live/metrics.json -O eval/live/plots -O eval/prc -o eval/importance.png python src/evaluate.py model/model.pkl data/processed
```

Before creating pipeline stages we need to initialize the DVC in the directory. </br>
```
dvc init
```
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/17ccbf06-5cf4-4be7-8d73-5288d1d214f5)

```
dvc remote add -d local /../../dvc_remote
```
Here config file contains the remote path where we will save the versions. Now we can see that the config file getting updated.
After running all the commands for creating dvc stages a new file dvc.yaml will be created with different stages will all the dependencies, parameters and output directory. The DVC pipeline will only get executed only if there is a change in parameters file. </br>

## dvc.yaml file created with different stages in the pipeline
```
stages:
  data_split:
    cmd: python src/data_split.py
    deps:
    - src/data_split.py
    params:
    - base.random_state
    - data_source.local_path
    - split.split_ratio
    outs:
    - data/split
  data_processing:
    cmd: python src/data_processing.py data/processed
    deps:
    - data/split
    - src/data_processing.py
    params:
    - process.dir
    - process.test_file
    - process.train_file
    - split.dir
    - split.test_file
    - split.train_file
    outs:
    - data/processed
  train:
    cmd: python src/train.py data/features model/model.pkl
    deps:
    - data/processed
    - src/train.py
    params:
    - base.random_state
    - base.target_col
    - model_dir
    - process.dir
    - process.test_file
    - process.train_file
    - train.n_est
    outs:
    - model/model.pkl
  evaluate:
    cmd: python src/evaluate.py model/model.pkl data/processed
    deps:
    - data/processed
    - model/model.pkl
    - src/evaluate.py
    outs:
    - eval/importance.png
    - eval/live/plots:
        cache: false
    - eval/prc:
        cache: false
    metrics:
    - eval/live/metrics.json:
        cache: false

plots:
- ROC:
    template: simple
    x: fpr
    y:
      eval/live/plots/sklearn/roc/train.json: tpr
      eval/live/plots/sklearn/roc/test.json: tpr
- Confusion-Matrix:
    template: confusion
    x: actual
    y:
      eval/live/plots/sklearn/cm/train.json: predicted
      eval/live/plots/sklearn/cm/test.json: predicted
- Precision-Recall:
    template: simple
    x: recall
    y:
      eval/prc/train.json: precision
      eval/prc/test.json: precision
- eval/importance.png

```

## DVC Pipiline Command
```
dvc repro
```
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/8c5bcb1b-2966-4a53-8cc2-fd2be7917bc5)

## DVC DAG
```
dvc dag
```
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/01b4b85d-9df9-49a3-8561-b4584f4a4cf2)

## Visualization and Metrics Plots
```
dvc metrics show
```
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/e21accb4-a885-4145-b95b-69c21fc9b111)

For plots we need to add following code in dvc.yaml
```
plots:
- ROC:
    template: simple
    x: fpr
    y:
      eval/live/plots/sklearn/roc/train.json: tpr
      eval/live/plots/sklearn/roc/test.json: tpr
- Confusion-Matrix:
    template: confusion
    x: actual
    y:
      eval/live/plots/sklearn/cm/train.json: predicted
      eval/live/plots/sklearn/cm/test.json: predicted
- Precision-Recall:
    template: simple
    x: recall
    y:
      eval/prc/train.json: precision
      eval/prc/test.json: precision
- eval/importance.png
```

### Command to generate plots
```
dvc plots show
```

![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/5f65d9e2-f7b9-4300-b458-20c2bf0ec0b1)
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/566e07eb-f16d-4ccb-9814-8abef16ecb02)  </br>

With index file path </br>
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/58b84bdf-a6c5-408f-adc1-a55370258e02)

## Experiment Tracking
Here we need to certain parameters to check how the model behaves to track different version of model.</br>
```
dvc push
```
This will push the data in the remote backend. This will create new files in dvc_remote folder containing tracked data in binary object format or blob storage format.

![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/6e6842cb-9b6d-4101-8517-7abc27064cca)

## For tracking the files
```
git add --all
git commit -m "first version"
git tag v1.0
git push
git push --tag
```

Change the estimators in the params.yaml file to check the model versoning. Changing the parameters in a stage will run the particular stage and all other stages will not run. </br>

If we want to see the differences in metrics and params with respect to different versions with different parameters. This should be run before dvc push command to see the difference </br> 
```
dvc metrics diff
dvc params diff
```
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/d73d0c6f-924d-4a27-833c-26ddf6c6fa7e)


## Inorder to go to the first version:
```
git checkout v1.0
dvc pull
dvc metrics show
```

## To remove the pipeline stages:
```
dvc remove data_split
dvc remove data_processing
dvc remove train
dvc remove evaluate
```

## Final Folder Directory
![image](https://github.com/srsapireddy/End-to-End-Machine-Learning-Pipeline-Creation-Using-DVC/assets/32967087/d0d3b8d1-6dcd-4938-b7ae-5e5182f898b2)
































