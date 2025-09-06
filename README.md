# deep-learning-coursework
Coursework completed in university for Deep Learning module

## Project 1
The goal of this initial project comprised of implementing a Multi-Layer Perceptron Network from scratch, utilising **Pandas**, **Numpy**, **Matplotlib** and specific **Scikit-Learn** libraries.

### Task 1 - Initial Model
- Implement activation functions (ReLu, Sigmoid, tanh).
- Initialise weights using a uniform distribution U(-1, 1) with bias 0.
- Use mini-batch GD and backpropagation to train the network. 
- Hyperparameters (batch size, learning rate, epochs) were predefined in the assignment specification.

### Task 2 - Evaluation and Metrics
- Visualised the loss and accuracy across training epochs, and printed the final values.
- Assessed whether the model demonstrated correct learning behaviour (i.e. decreasing loss, stabilising accuracy).

### Task 3 - Optimisation
- Experimented with varying **learning rates**, **batch sizes**, and **network architectures** (e.g. number of hidden units/layers).
- Experiment with optimisation methods.
- Compare to previous model and explained improvements (or regressions).

## Project 2

The goal of Project 2 was to detect computer-generated (fake) reviews with different ANN architectures.
The dataset consisted of 2,500 labelled reviews - 2,000 for training and 500 for testing - and I was tasked
with creating these architectures using **Keras** and **Tensorflow** with word embeddings - **word2vec** and **GloVe** - for text classification

Included Jupyter Notebook for this project, PDF document not yet included.

### Task 1 - Model Implementations
(All specified to be placed within one Notebook)

- **Multilayer Perceptron (MLP):**
  - Each input as a single averaged embedding vector.
  - Each input as a sequence of embedding vectors (Keras embedding layer).
- **Convolutional Neural Network (CNN):**
    - Each input taken as a sequence of embedding vectors (again using Keras).
    - Utilise 1D Convolution and Global Max Pooling.
- **Recurrent Neural Network and LSTM:**
  - Each input is taken the same as prior.
  - Tested on RNN and LSTM models.
- Visualised loss and accuracy for each model.

### Task 2 - Analysis and Results

- Documented experimentation with hyperparameters - batch size, learning rate, optimisers, dropout.
- Presented findings through data visualisation and evaluation metrics - accuracy and loss, confusion matrix, ROC curve.
- Performed K-Fold Cross Validation.

### Task 3 - Final Outcome

- Identified best performing model as RNN.
- Reflected on each model's limitations and challenges faced with each one.
- Looked at improvements to be made and future exploration - for example, alternative embeddings or transformer-based models.
