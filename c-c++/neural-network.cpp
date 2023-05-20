#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

/*
  The steps to this neural network:
  Each neuron gets its corresponding bias added to it.
  Each neuron in the next layer becomes the sum of all the neurons in the previous layer multiplied to their corresponding weights.
  The activation function is applied to each neuron of the new layer.
  Repeat for the next layer.
*/

// Derivative of tanh (hyperbolic tangent) function.
double tanh_derivative (double x) {
  // Tanh is already defined in the <cmath> header so I only need a function for its derivative.
  return 1 - pow(tanh(x), 2);
}

// ReLU function.
double relu (double x) {
  return x < 0 ? 0 : x;
}

// Derivative of ReLU function.
double relu_derivative (double x) {
  return x < 0 ? 0 : 1;
}

// Leaky ReLU function.
double leaky_relu (double x) {
  return x < 0 ? (0.1 * x) : x;
}

// Derivative of leaky ReLU function.
double leaky_relu_derivative (double x) {
  return x < 0 ? 0.1 : 1;
}

// Sigmoid function
double sigmoid (double x) {
  return 1/(1 + exp(-x));
}

// Derivative of sigmoid function.
double sigmoid_derivative (double x) {
  double sigmoid_x = sigmoid(x);
  return sigmoid_x * (1 - sigmoid_x);
}

// Returns a random number between -1 and 1.
double random_number () {
  // I don't really know how this works.
  static random_device generator;
  static uniform_real_distribution<double> distribution(-1.0, 1.0);
  return distribution(generator);
  //  return 1;
}

// struct for a data point.
struct Data_Point {
  vector<double> inputs;
  vector<double> outputs;
};

// Layer class.
class Layer {
private:
  // weights[output neuron][input neuron]
  vector<vector<double>> weights;
  vector<vector<double>> weight_cost_gradients;
  // biases[output neuron]
  vector<double> biases;
  vector<double> bias_cost_gradients;

  // The inputs to the layer.
  vector<double> inputs;
  // The weighted sums of the inputs bewfore being passed through the activation function.
  vector<double> preactivations;
  // Output values of the layer.
  vector<double> activations;
  
  int num_neurons;
  int num_inputs;
public:
  // Layer constructor.
  Layer (int num_inputs, int num_neurons) :
    // Setting num_neurons and num_inputs to their corresponding values.
    num_neurons(num_neurons), num_inputs(num_inputs),
    // Resizing the weights and biases.
    weights(num_neurons, vector<double> (num_inputs)), biases(num_neurons),
    // Resizing the weight and bias cost gradients.
    weight_cost_gradients(num_neurons, vector<double> (num_inputs)), bias_cost_gradients(num_neurons),
    // Resizing the preactivations, and the activations.
    inputs(num_inputs), preactivations(num_neurons), activations(num_neurons) {}

  int get_activations_size () {
    return activations.size();
  }
  
  // Allows changing the weights array from outside the Layer class.
  void set_weights (const int output_neuron, const int input_neuron, const double value) {
    weights[output_neuron][input_neuron] = value;
  }
  
  // Allows the reading of the weights array from outside the layer class.
  double get_weights (const int output_neuron, const int input_neuron) {
    return weights[output_neuron][input_neuron];
  }
  
  // Allows changing the biases array from outside the Layer class.
  void set_biases (const int output_neuron, const double value) {
    biases[output_neuron] = value;
  }
  
  // Allows the reading of the biases array from outside the layer class.
  double get_biases (const int output_neuron) {
    return biases[output_neuron];
  }
  
  // Allows changing the weight_cost_gradient array from outisde the Layer class.
  void set_weight_cost_gradients (const int output_neuron, const int input_neuron, const double value) {
    weight_cost_gradients[output_neuron][input_neuron] = value;
  }
  
  // Allows changing the bias_cost_gradients variable from outside the Layer class.
  void set_bias_cost_gradients (const int output_neuron, const double value) {
    bias_cost_gradients[output_neuron] = value;
  }
  
  // Allows the reading of the number of neurons in this layer.
  int get_num_neurons () {
    return num_neurons;
  }
  
  // Allows the reading of the number of neurons inputing into this layer.
  int get_num_input_neurons () {
    return num_inputs;
  }
  
  // Initialize the layers weights and biases randomly to a value between -1 and 1.
  void initialize_layer_randomly () {
    for (auto &output : weights) {
      // This loops over every neuron that this layer outputs to.
      for (auto &weight : output) {
	weight = random_number();
      }
      // To get the index of the loop, subtract the starting location from the current location.
      biases[&output - &weights[0]] = random_number();
    }
  }
  
  // Calculates the outputs for the layer from a given input.
  vector<double> calculate_layer_outputs (const vector<double> &layer_inputs) {
    inputs = layer_inputs;
    
    int output_index = 0;
    
    for (const auto &weights_for_this_output : weights) {
      int input_index = 0;
      // Loop over every output, output is a reference to every weight that leads to the current output neuron.
      // Set the weighted output as the bias, because this is the same as just adding the bias to 0.
      preactivations[output_index] = biases[output_index];
      for (const auto &weight : weights_for_this_output) {
	// Loop over every weight that leads to the current output neuron.
	preactivations[output_index] += layer_inputs[input_index] * weight;

	input_index++;
      }
      // Now that preactivations are calculated, apply the activation function.
      activations[output_index] = leaky_relu(preactivations[output_index]);
      
      output_index++;
    }
    
    return activations;
  }
  
  // Applies the gradient to each weight and bias, effectively moving the cost downhill.
  void apply_gradients (const double learn_rate) {
    // I multiply the gradients by the learn rate, so that it won't overshoot the minimum, and to tweak with the step size in downhill.
    for (auto &output : weights) {
      // This loops over every neuron that this layer outputs to.
      for (auto &weight : output) {
	// This for loop loops over every weight that is connected to the output neuron.
	// Subtracting the slope of the weight makes the cost move downhill.
	// &weight - &output[0] gets the index of the inner loop.
	weight -= weight_cost_gradients[&output - &weights[0]][&weight - &output[0]] * learn_rate;
      }
      // &output - &weights[0] gives the index of the outer loop.
      // Subtracting the slope of the bias makes the cost move downhill.
      biases[&output - &weights[0]] -= bias_cost_gradients[&output - &weights[0]] * learn_rate;
    }
  }

  // Calculates the cost of a single neuron given its output and the desired/expected output.
  double calculate_neuron_cost (double output, double expected_output) {
    double neuron_cost = output - expected_output;
    return neuron_cost * neuron_cost;
  }
  
  // Calculates the partial derivative of the cost with respect to the output activation.
  double calculate_neuron_cost_derivative (double output, double expected_output) {
    return 2 * (output - expected_output);
  }

  vector<double> calculate_output_layer_node_values (vector<double> expected_outputs, int num_outputs) {
    vector<double> node_values;
    node_values.resize(num_outputs);
    for (int output = 0; output < num_outputs; output++) {
      // Loop over each output neuron and calculate its partial derivative.
      double cost_derivative = calculate_neuron_cost_derivative(activations[output], expected_outputs[output]);
      double activation_function_derivative = leaky_relu_derivative(preactivations[output]);
      node_values[output] = cost_derivative * activation_function_derivative;
    }
    return node_values;
  }

  vector<double> calculate_hidden_layer_node_values (Layer old_layer, const vector<double> old_node_values, int num_old_nodes) {
    vector<double> new_node_values;
    new_node_values.resize(num_neurons);

    for (auto &new_node_value : new_node_values) {
      new_node_value = 0;
      for (int old_node = 0; old_node < num_old_nodes; old_node++) {
	// &new_node_value - &new_node_values[0] gives the index of the outer loop.
	new_node_value += old_layer.get_weights(old_node, (&new_node_value - &new_node_values[0])) * old_node_values[old_node];
      }
      // &new_node_value - &new_node_values[0] gives the index of the outer loop.
      new_node_value *= leaky_relu_derivative(preactivations[&new_node_value - &new_node_values[0]]);
    }
    
    return new_node_values;
  }

  void clear_gradients () {
    for (auto &output : weight_cost_gradients) {
      // Loop over every neuron that this layer outputs to.
      for (auto &weight_cost_gradient : output) {
	// Loop over weight cost gradient that connects to the output.
	// Set the weight gradient to 0.
	weight_cost_gradient = 0;
      }
      // &output - &weight_cost_gradients[0] gives the index of the outer loop.
      // Set the bias gradient to 0.
      bias_cost_gradients[&output - &weight_cost_gradients[0]] = 0;
    }
  }

  void update_gradients (vector<double> node_values) {
    for (auto &output : weight_cost_gradients) {
      // Loop over each weight cost gradient.
      for (auto &weight_cost_gradient : output) {
	// Evaluate the partial derivative cost/weight of the current weight.
	// &output - &weight_cost_gradients[0] gives the index of the outer loop.
	// &weight_cost_gradient - &output[0] gives the index of the inner loop.
	double derivative_cost_wrt_weight = inputs[&weight_cost_gradient - &output[0]] * node_values[&output - &weight_cost_gradients[0]];
	//	cout << &output - &weight_cost_gradients[0] << " " << node_values[&output - &weight_cost_gradients[0]] << "\n";
	weight_cost_gradient += derivative_cost_wrt_weight;
      }
      // The partial derivative of cost/bias is just 1, becuase the bias isn't multiplied by anything.
      // This makes the derivative of the cost with respect to the bias just the node value.
      // &output - &weight_cost_gradients[0] gives the index of the outer loop.
      bias_cost_gradients[&output - &weight_cost_gradients[0]] += node_values[&output - &weight_cost_gradients[0]];
    }
  }

  // Prints all the weight and bias gradients of the layer.
  void print_gradients () {
    int bias_gradient_index = 0;
    for (const auto &weights_for_this_output : weight_cost_gradients) {
      // This loops over every set of weights that connect to this output.
      cout << "Weight Gradients: ";
      for (const auto &weight_cost_gradient : weights_for_this_output) {
	// This for loop loops over every weight that is connected to the neuron.
	// Print out the weight.
	cout << weight_cost_gradient << " ";
      }
      cout << "Bias Gradient " << bias_gradient_index << ": " << bias_cost_gradients[bias_gradient_index] << " ";
      bias_gradient_index++;
    }
  }
  
  // Prints all the weights and biases of the layer.
  void print_layer () {
    for (const auto &weights_for_this_output : weights) {
      // This loops over every set of weights that connect to this output.
      cout << "Weights: ";
      for (const auto &weight : weights_for_this_output) {
	// This for loop loops over every weight that is connected to the neuron.
	// Print out the weight.
	cout << weight << " ";
      }
      // To get the index of the loop, subtract the starting location from the current location.
      cout << "Bias " << &weights_for_this_output - &weights[0] << ": " << biases[&weights_for_this_output - &weights[0]] << " ";
    }
  }
};

// Network class.
class Network {
private:
  vector<Layer> layers;
  int num_layers;
public:
  // Network constructor.
  Network (int layer_sizes[], int num_layers) :
    num_layers(num_layers) {
    for (int i = 1; i < num_layers; i++) {
      // I starts at 1 because the input layer doesn't have any weights or biases, but the size of the input layer is recorded.
      // Add a layer to the end of the list of layers.
      layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
    }
  }
  
  // Initializes each weight and bias of the network to be a random number between -1 and 1.
  void initialize_network_randomly () {
    for (auto &layer : layers) {
      // For each layer, randomize its weights and biases.
      layer.initialize_layer_randomly();
    }
  }
  
  // Calculates the outupts of the network from a given input.
  vector<double> calculate_network_outputs (vector<double> inputs) {
    for (auto &layer : layers) {
      // For each layer, calculate its output, and then treat that output as the new input for the next layer.
      inputs = layer.calculate_layer_outputs(inputs);
    }
    
    return inputs;
  }
  
  // Calculates the cost of a single data point.
  double calculate_data_point_cost (Data_Point data_point) {
    vector<double> outputs = calculate_network_outputs(data_point.inputs);
    double cost = 0;
    
    Layer &output_layer = layers[layers.size() - 1];
    
    for (const auto &output : outputs) {
      // &output - &outputs[0] gives the index of the loop.
      cost += output_layer.calculate_neuron_cost(output, data_point.outputs[&output - &outputs[0]]);
    }
    
    return cost;
  }
  
  // Calculates the average cost of an array of data points.
  double calculate_average_data_cost (Data_Point data[], const int num_data_points) {
    double total_cost = 0;
    for (int data_point = 0; data_point < num_data_points; data_point++) {
      total_cost += calculate_data_point_cost(data[data_point]);
    }
    
    return total_cost/num_data_points;
  }
  
  void update_all_gradients (Data_Point data_point) {
    // Run the inputs through the network to store values such as the weighted inputs and activations.
    calculate_network_outputs(data_point.inputs);
    
    // Make a copy of the last layer.
    Layer &output_layer = layers[layers.size() - 1];
    // Calculate the output layer node values.
    vector<double> node_values = output_layer.calculate_output_layer_node_values(data_point.outputs, data_point.outputs.size());
    // Update the gradients using the node values.
    output_layer.update_gradients(node_values);
    
    for (int hidden_layer_index = layers.size() - 2; hidden_layer_index >= 0; hidden_layer_index--) {
      // Loop backwards through the hidden layers.
      // Update the gradients of the hidden layer.
      Layer &hidden_layer = layers[hidden_layer_index];
      // Calculate the hidden layer node values.
      node_values = hidden_layer.calculate_hidden_layer_node_values(layers[hidden_layer_index + 1], node_values, node_values.size());
      // Update the gradients using the node values.
      hidden_layer.update_gradients(node_values);
    }
  }
  
  void clear_all_gradients () {
    for (auto &layer : layers) {
      // For each layer, clear its gradients.
      layer.clear_gradients();
    }
  }
  
  // Applies the gradient of all weights and biases for all layers.
  void apply_all_gradients (const double learn_rate) {
    for (auto &layer : layers) {
      layer.apply_gradients(learn_rate);
    }
  }

  void print_network_gradients () {    
    for (int layer = 0; layer < num_layers - 1; layer++) {
      // Looping over each layer - 1.
      // Because the input layer doesn't get printed, the layer with index 0 is actually the 1st layer, not the 0th layer.
      cout << "Layer " << layer + 1 << ": ";
      layers[layer].print_gradients();
      cout << "\n";
    }
  }

  void learn (Data_Point training_batch[], const int training_batch_size, const double learn_rate) {
    // Update the gradients using each data point of the training batch.
    for (int data_point = 0; data_point < training_batch_size; data_point++) {
      update_all_gradients(training_batch[data_point]);
    }

    //print_network_gradients();
    
    // Apply all the gradients.
    apply_all_gradients(learn_rate/training_batch_size);
    
    // Clear all the gradients so that they will be ready for the next training batch.
    clear_all_gradients();
  }
  
  // Runs a single iteration of gradient descent not using calculus.
  void alearn (Data_Point training_data[], const int num_data_points, const double learn_rate) {
    // To calculate the slope of a function at any point, I divide a small change in the input by a small change in the output.
    // The formula for this can be shown as this: (y2 - y1)/(x2 - x1).
    // This is dividing the difference between the output of the function, and the input of the function.
    // The smaller this difference between y2 and y1 and x2 and x1 is, the closer to the actuall slope I get.
    // I am using h as this difference.
    // For each weight and bias, the weight and bias acts as x, and the cost acts as y.
    // I will find the cost, then change the weight or bias by h then find the cost again and then use (y2 - y1)/(x2 - x1) to find the sloep of that weight or bias.
    // I then store the slope into the weight_cost_gradients[][] and bias_cost_gradients of the current layer, and then apply all the gradients.
    
    const double h = 0.000001;
    const double original_cost = calculate_average_data_cost(training_data, num_data_points);
    
    for (auto &layer : layers) {
      // Loop over each layer.
      for (int output_neuron = 0; output_neuron < layer.get_num_neurons(); output_neuron++) {
	// Loop over each neuron that the layer outputs to.
	for (int input_neuron = 0; input_neuron < layer.get_num_input_neurons(); input_neuron++) {
	  // Loop over each neuron in the previous layer.
	  // Change the weight connecting the input and output neuron by h.
	  layer.set_weights(output_neuron, input_neuron, (layer.get_weights(output_neuron, input_neuron) + h));
	  // Calculate the cost gradient by dividing the change in cost by the small change to the weight.
	  layer.set_weight_cost_gradients(output_neuron, input_neuron, ((calculate_average_data_cost(training_data, num_data_points) - original_cost)/h));
	  // Put the weight back to its original value.
	  layer.set_weights(output_neuron, input_neuron, (layer.get_weights(output_neuron, input_neuron)) - h);
	}
        
	// Change the bias connecting to the output neuron by h.
	layer.set_biases(output_neuron, (layer.get_biases(output_neuron) + h));
	// Calculate the cost gradient of the bias by dividing the change in the cost by the small change to the bias.
	layer.set_bias_cost_gradients(output_neuron, ((calculate_average_data_cost(training_data, num_data_points) - original_cost)/h));
	// Setting the bias back to its original value.
	layer.set_biases(output_neuron, (layer.get_biases(output_neuron) - h));
      }
    }
    //print_network_gradients();
    apply_all_gradients(learn_rate);
  }
  
  // Prints all the weights and biases of the network.
  void print_network () {
    // Input layer has no weights and biases.
    cout << "Input layer (layer 0) has no weights and biases.\n";
    
    for (int layer = 0; layer < num_layers - 1; layer++) {
      // Looping over each layer - 1.
      // Because the input layer doesn't get printed, the layer with index 0 is actually the 1st layer, not the 0th layer.
      cout << "Layer " << layer + 1 << ": ";
      layers[layer].print_layer();
      cout << "\n";
    }
  }
};

int main () {
  int layer_sizes[] = {2, 3, 1};
  int num_layers = sizeof(layer_sizes)/sizeof(layer_sizes[0]);
  Network my_network(layer_sizes, num_layers);
  
  my_network.initialize_network_randomly();
  my_network.print_network();
  
  vector<double> inputs;
  inputs.resize(2);
  vector<double> outputs;
  outputs.resize(1);
  
  Data_Point data[4];
  data[0].inputs = {0, 0};
  data[0].outputs = {0};
  data[1].inputs = {1, 1};
  data[1].outputs = {0};
  data[2].inputs = {0, 1};
  data[2].outputs = {1};
  data[3].inputs = {1, 0};
  data[3].outputs = {1};

  int num_data_points = sizeof(data)/sizeof(data[0]);
  cout << my_network.calculate_average_data_cost(data, num_data_points) << "\n";

  int num_iterations = 10000;
  //int num_iterations = 1;
  for (int i = 0; i < num_iterations; i++) {
    my_network.learn(data, num_data_points, 0.1);
    /*my_network.initialize_network_randomly();
    my_network.print_network();
    my_network.alearn(data, num_data_points, 0.1);*/
    if (i % 500 == 0) {
      cout << num_iterations - i << " " << my_network.calculate_average_data_cost(data, num_data_points) << "\n";
    }
  }

  cout << my_network.calculate_average_data_cost(data, num_data_points) << "\n";

  while (1) {
    cin >> inputs[0];
    cin >> inputs[1];
    cout << (my_network.calculate_network_outputs(inputs)[0] /*> 0.5 ? 1 : 0*/) << "\n";
  }
  
  return 0;
}
