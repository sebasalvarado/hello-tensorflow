import tensorflow as tf


# Find implicit default graph.
graph = tf.get_default_graph()
print(graph)

# Find implicit operations in the graph
print(graph.get_operations())

# Start with a simple constant in Tf
input_value = tf.constant(1.0)
# After we defined this value, it lives as an operation in the default graph
operations = graph.get_operations()
print(operations, operations[0])
# Inspecting our constant 32-bit float
print(input_value)

## NOTE: Until here we have only done "definition" now to evaluate all that we have done we need to create a session.
session = tf.Session()
print(session.run(input_value))

## FIRST NEURON: Let's build a neuron with one parameter ---------------------------------------------------------------
weight = tf.Variable(8.0)
# Check what operations are added to the graph for just adding one variable
for op in graph.get_operations(): print(op.name)
output_value = weight * input_value

# How about now? What operations do we have
for op in graph.get_operations(): print(op.name)

# Run the multiplication initializing all variables
init = tf.global_variables_initializer()
session.run(init)
runned_output = session.run(output_value)
print(runned_output)


### VISUALIZATION: Picture our computation graph in TensorBoard =------------------------------------------------------
x = tf.constant(1.0, name="input")
w = tf.Variable(0.8, name="weight")
y = tf.multiply(x, w, name="output")

# To allow TensorBoard to read our session we need to write it into a directory using SummaryWriter
summary_writer = tf.summary.FileWriter("log_simple_graph", session.graph)

### LEARNING: We want to allow our simple neuron to learn the weight parameter to minimize some function
# Objective: We want our neuron to learn the function f(1) = 0, right now our function takes 1 and return 0.8
# In the end the weight should approach 0 to make 1 * 0 = 0
y_ = tf.constant(0.0)
loss = (y - y_) ** 2
optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)
grads_and_vars = optim.compute_gradients(loss)

session.run(tf.global_variables_initializer())
print(session.run(grads_and_vars[1][0]))

#Apply the gradient which finishes the backpropagation on one train step. Our weight is moving in right direction
session.run(optim.apply_gradients(grads_and_vars))
print(session.run(w))


# Training on 1000 steps to minimize the loss
STEPS = 1000
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
for i in range(STEPS):
    session.run(train_step)

print(session.run(y))



## EVALUATION: Visualize the training steps on TensorBoard
# Instrumenting the computation graph by adding operations that summarize its state
summary_y = tf.summary.scalar("output", y)
summary_writer = tf.summary.FileWriter("log_simple_stats")
session.run(tf.global_variables_initializer())
for i in range(STEPS):
    summary_str = session.run(summary_y)
    summary_writer.add_summary(summary_str, i)
    session.run(train_step)
