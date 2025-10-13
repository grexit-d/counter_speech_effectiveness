# Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study

This repository contains data and code of the paper "Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study".

We annotated two datasets, containing hate-speech/counter-speech pairs, CONAN (Chung et al., 2019), and Twitter Dataset (Albanyan & Blanco, 2022). We release two samples of 50 examples each, taken from both datasets. 

The two samples were used to calculate the Inter Annotator Agreement (IAA) between three annotators. 

- [Data](https://github.com/grexit-d/counter-speech_effectiveness/tree/main/Data) contains all the data used for the IAA, organized by first and last annotation rounds. The full annotated datasets will be publicly available upon paper's acceptance.
- [Guidelines](https://github.com/grexit-d/counter-speech_effectiveness/tree/main/Guidelines) contains the annotation guidelines. To view them, download the PDF file and open it on your computer.
- [notebooks](notebooks) includes the code used in the experiments:
  - `bert_baselines.ipynb`: Baseline models (Bert_CS and Bert_CS_HS)
  - `learnable_dependency_matrix.py`: Multi-task and Dependency Matrix models
    
To run the `learnable_dependency_matrix.py` script, use the following command in the terminal:

 `python learnable_dependency_matrix.py --train-file data/train.csv --eval-file data/eval.csv`
  
Replace train.csv and eval.csv with the paths to the desired training and evaluation datasets (i.e., CONAN, Twitter).

- [full_results.xlsx](full_results.xlsx) contains the full results of all the experiments in the paper. 
