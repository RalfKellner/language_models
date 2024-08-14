# Pretraining

Each folder represent a different approach for pretraining:

* Masked language modeling only
* Discriminator only
* Electra - Masked modeling and discriminator

The current plan is to pretrain two models of each, one small with a hidden size of 256 and one large (or base) with a hidden size of 512. Depending on the hidden size batch sizes are somewhere between 32 and 128. Of course, we may want to experiment with hyperparamters and at a later stage, I also want to experiment of the impact of different text types (for instance, trained on Form 10K fillings, only). Also something we should not forget is to experiment with a masking approach that dominantely masks domain specific words. 

Everything can be specified in the yaml-config files. If we set, e.g., the number of sequences for a specific text type to 1000 and the epochs to 5. The first 5000 unique sequences of the text source will be used. Only if the limit times the number of epochs exceeds the number of available sequences, the dataset will start to include sequences from the start of the table again. 

If a model is pretrained, it gets saved at the /data/language_models/pretrained_models folder. The name and path, respectively, is created automatically. It is the way of pretraining and a successive number, e.g., electra_01.
