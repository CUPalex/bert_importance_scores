# :paw_prints: Identifying components of BERT that are essential for retrieving linguistic information

## :paw_prints: About
This is a course project in which I explore the method based on importance scores to localize linguistic information in BERT model.

The results are presented in the ```coli_final_project.pdf``` file.

## :paw_prints: Files content

All in all, the repository has the following meaningful files:

```
linguistic_features.py # contains a class with which one can extract linguistic features from a dataset
model.py # contains a class with model class with linear head on top
pipeline.py # contains all the logic on model fine-tuning and importance scores calculation
task.py # contains methods for creating datasets and dataloaders
run.py # entrypoint file
coli_final_project.pdf # description of the project and results
analysis.ipynb # here I plot graphs which I then insert into the .pdf file 
```

## :paw_prints: Usage

To train the models and compute the importance scores, just run ```run.py``` file with python:

```python run.py```

:heavy_exclamation_mark: One important notice: do not forget to change wandb project name for one that is suitable for you!

You might need to install ```transformers, pytorch, numpy, scipy``` and ```stanza``` libraries. The full list of requirements used in my environment can be found in ```requirements.txt```.