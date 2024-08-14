# How to start

Clone this repo, create a virtual environment, activate it and install all packages in the requirements.txt file. Afterwards navigate to the finlm folder and use *pip install -e .* to install the package locally in developer mode. If you want to create lasting changes, please use branching.   

# Financial language modeling

This folder includes a package called "finlm" with modules which can be used for pretraining and finetuning encoder based models. Furthermore, four directories include scripts how to 

* train a tokenizer - 00_tokenizer
* chunk raw texts into equal sizes - 01_text_chunking
* pretrain models either by masked language modeling, a discriminator or by the Electra approach
* finetune models for tasks such as regression and classification

In each of these folder, you will find a seperate read_me which provides some basic information. Also take a look at the test folder in the finlm package which demonstrates how to conduct the major steps for reduced data and model sizes. 

Furthermore, the finlm models module includes two new encoder models whose head is meant to receive documents which consist of an arbitray number of sequences whose embeddings are mapped by aggregated either by a fully connected neural network or a simple attention mechanism. Botch models can be used for regressino and classification tasks. 

## Future steps

* Train a small and base model for every pretraining approach. I already pretrained a mlm and electra model, however, failed to set the maximum position to 256. While this should be no problem, I want to start fresh. Small means, we use a hidden size of 256, base models have a hidden size of 512. 
* Collect different finance specific finetuning tasks
* Pretrain different models in order to find out what mostly impacts the performance of domain specific language models
