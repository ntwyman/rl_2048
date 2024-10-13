# tensorflow_play

Playground for exploring machine learning with tensorflow

# contents

### mnist_softmax.py

Following along from https://www.tensorflow.org/get_started/mnist/beginners I changeed some of the node/variable names to be more instructive, e.g. 'model' instead of 'y' for the model outputs and 'actual' instead of 'y_' for the training labels.

### three_layer.py

Based on the code from mnist_software.py I tried to implement the 3 layer neural net described in http://neuralnetworksanddeeplearning.com/chap1.html.

At the same learning rate of 10000 batches it did much worse with zero initialized wwights and biases. With randomized initial values it did about the same. I then modified the code to run more batches and print out the accuracy score after each batch as I was wondering how quickly the models converge.


### mnist_convol.py

Following along with https://www.tensorflow.org/get_started/mnist/pros. Again I changed the node names to be (to me) more illustrative. For an excellent introduction to convolution neural networks see https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
