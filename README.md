# Stance-Classification

The stance classification program predicts stance in texts related to airline-reviews. The project uses deep neural network architecture for prediction, and employs attention mechanism (through softmax linear layer). The project has been written in Python and keras.

# Dataset
The dataset consists of stance values, belonging to either of the 3 values: 0 (negative), 1 (neutral) and 2(positive). The 'text' attribute  consists of raw text. The dataset if cleaned (data_clean.py) prior to use for prediction. The cleaning process removes hashtags, links(http) and stopwords. Slangs and abbreviations (if any) are substituted for, with the help of a dictionary (emnlp_dict.txt) that specifies the interpretations for various slangs used online.

# Dependencies
Python 2.7
Numpy
Tensorflow and its dependencies
Keras

# Execution
First, the dataset needs preprocessing prior to its usage. The data_clean.py file is used for this task, and accepts argument values for specifying the input data file and the output processed data file. Run the file as the following terminal instruction:

    python data_clean.py airline_stance_data.csv cleaned_data.csv
      
Now, this 'cleaned_data' can be used as the dataset for the model, which accepts the data file name as argument. Therefore, the stance classification program can be run as:

    python model.py cleaned_data.py
    
The accuracy and loss graph are generated after training. Also, along with this, a few text examples from the test set generated and their predicted stance are printed.
