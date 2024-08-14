# Finetuning models

For every common task such as regression, binary classification or multi-classification, the downstreaming module can be used. It is one folder for every finetuning task. Most of the finetuning parameters must be specified in a yaml file. A hyperparameter class exists to define them, they are initialized in the yaml file as a key with keys and values, e.g., n_epochs.

It is important to look for a consistent name giving for the save_path in the config file. The model and tokenizer are either paths of pretrained models or names of models which are available on huggingface, e.g., "google/electra-base-discriminator". The end of the save_path should include the name of the task and the model type, e.g., .../financial_phrasebank/electra_01

Huggingface datasets are often in odd order. For instance if we split the financial phrasebank dataset without shuffling, the classes in training and test data have very different balances. Or, the stanford imdb data first includes 0 labels and afterwards 1 labels. This is why the import and shuffling of the datasets must be done specifically in the training script. 

Furthermore, in the script you find a model_loader which must be defined. This implementation generates flexibilty w.r.t the pretraining model, however, also make sure this is specified consistent. I.e., if we pretrain a model with Electra classes, we need to use Electra classes when importing pretrained models for finetuning. 