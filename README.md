## Hello TensorFlow - Simple Neuron Learning and Visualization
## Credits: Aaron Schumacher - O'Reilly Media [post](https://www.oreilly.com/learning/hello-tensorflow)

This small implementation of the post from Aaron Schumacher guides you through the whole implementation in Python code. If you much rather read comments in code as well as see every piece of it implemented checkout `main.py` that has all explained for you. 

The value of this repository is allowing you to visualize your neuron in your browser with TensorBoard and 
# Getting Started
1. Install [TensorFlow](https://www.tensorflow.org/install/)
2. Clone this repository `git clone https://github.com/sebasalvarado/hello-tensorflow.git` `cd hello-tensorflow`

# Visualize Neuron
In your terminal run:
`python main.py`

Open one tab in your terminal and run 
`tensorboard --logdir=log_simple_graph`

Open your browser in `localhost:6006/#scalars`
# Visualize Weight Learning

On another terminal window run:
`python main.py`

Open your browser in `localhost:6006/#output`  to visualize the weight learning graph and how fast our function learned its parameters.

