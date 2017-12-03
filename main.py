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


