# Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study

This repository contains data and code of the paper "Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study", accepted at WI-IAT 2025.

We annotated two datasets, containing hate-speech/counter-speech pairs, CONAN (Chung et al., 2019), and Twitter Dataset (Albanyan & Blanco, 2022).

We release the two datasets annotated with the our six effectiveness dimensions. 

Additionally, We release the two samples that were used to calculate the Inter Annotator Agreement (IAA) between three annotators. 

- [Data](https://github.com/grexit-d/counter-speech_effectiveness/tree/main/Data) contains all the data used for the IAA, organized by first and last annotation rounds. The full annotated datasets will be publicly available upon paper's acceptance.
   - The full annotated datasets are in the folder [Annotated datasets](https://github.com/grexit-d/counter-speech_effectiveness/tree/main/Data/Annotated\datasets).
   - The samples used for the IAA are in the folder [IAA Data](https://github.com/grexit-d/counter-speech_effectiveness/tree/main/Data/IAA\Data).
- [Guidelines](https://github.com/grexit-d/counter-speech_effectiveness/tree/main/Guidelines) contains the annotation guidelines. To view them, download the PDF file and open it on your computer.
- [notebooks](notebooks) includes the code used in the experiments:
  - `bert_baselines.ipynb`: Baseline models (Bert_CS and Bert_CS_HS)
  - `learnable_dependency_matrix.py`: Multi-task and Dependency Matrix models
    
To run the `learnable_dependency_matrix.py` script, use the following command in the terminal:

 `python learnable_dependency_matrix.py --train-file data/train.csv --eval-file data/eval.csv`
  
Replace train.csv and eval.csv with the paths to the desired training and evaluation datasets (i.e., CONAN, Twitter).

- [full_results.xlsx](full_results.xlsx) contains the full results of all the experiments in the paper.


## Datasets




## Cite us

If you find this repo useful or you use our dataseta, please cite us!

@article{damo2025effectiveness,
  title={Effectiveness of Counter-Speech against Abusive Content: A Multidimensional Annotation and Classification Study},
  author={Damo, Greta and Cabrio, Elena and Villata, Serena},
  journal={arXiv preprint arXiv:2506.11919},
  year={2025}
}
