# Patent-Summarizer
Patent abstract summarization using recurrent neural networks.

## Abstract

This work presents an abstractive summarization model for extracting the main ideas from patent abstracts and summarizing them into a title. The model uses an encoder-decoder Recurrent Neural Network (RNN) with attention mechanism. The evaluation is performed on a patent dataset specifically accumulated for this work. The dataset is kept generic and contains patents from multiple domains.


## Dataset

We used patents whose publication dates were from year 2000 and onwards. This also al- lowed us to avoid old information in the text. We also made sure that the abstract information was only in English while creating the dataset. The final dataset consisted of 25,743 patents. We split this dataset into training, validation and testing set, containing 16,000, 4,000 and 5,743 patents respectively. To ensure that our model generalized well to different categories of patents, we explicitly collected patents from multiple diverse fields.  
 
 For word embeddings, we used pre-trained GloVe embeddings(http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Running the model

For training a model, we need to specify a name for our model. E.g., with name <i>model-name</i>, run:

<style background-color="gray"> python main.py model-name </style>

To load a previously saved model, set the load flag in main.py to True. 

## Evaluate the model

For evaluating a model <i>model-name</i> on given patent dataset <i>data-set</i>, run:

python evaluation.py <i>model-name</i> <i>data-set</i>

The format of dataset files should be similar to the files in /data folder.
