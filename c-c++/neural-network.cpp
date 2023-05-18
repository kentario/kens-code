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
    weight_cost_gradients(num_neurons, vector<double> (num_inputs)), bias_cost_gradients(num_neurons) {}
  
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
  vector<double> calculate_layer_outputs (const vector<double> &inputs) {
    vector<double> weighted_outputs;
    weighted_outputs.resize(num_neurons);
    
    for (const auto &output : weights) {
      // Loop over every output, output is a reference to every weight that leads to the current output neuron.
      // &output - &weights[0] gets the index of the loop.
      // Set the weighted output as the bias, because this is the same as just adding the bias to 0.
      weighted_outputs[&output - &weights[0]] = biases[&output - &weights[0]];
      for (const auto &input : output) {
	// &input - &output[0] gets the index of the inner loop.
	// Loop over every weight that leads to the current output neuron.
	weighted_outputs[&output - &weights[0]] += inputs[&input - &output[0]] * weights[&output - &weights[0]][&input - &output[0]];
      }
      // Once the weighted outputs are calculated, apply the activation function.
      weighted_outputs[&output - &weights[0]] = leaky_relu(weighted_outputs[&output - &weights[0]]);
    }
    
    return weighted_outputs;
  }
  
  // Applies the gradient to each weight and bias, effectively moving the cost downhill.
  void apply_gradients (const double learn_rate) {
    // I multiply the gradients by the learn rate (some small number such as 0.1) so that it won't overshoot the minimum.
    // Also so that I can change the rate of learning.
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
  
  // Prints all the weights and biases of the layer.
  void print_layer () {
    for (const auto &output : weights) {
      // This loops over every neuron that this layer outputs to.
      cout << "Weights: ";
      for (const auto &weight : output) {
	// This for loop loops over every weight that is connected to the neuron.
	// Print out the weight.
	cout << weight << " ";
      }
      // To get the index of the loop, subtract the starting location from the current location.
      cout << "Bias " << &output - &weights[0] << ": " << biases[&output - &weights[0]] << " ";
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
  
  // Calculates the cost of a single output neuron given its output and the desired/expected output.
  double calculate_neuron_cost (double output, double expected_output) {
    double neuron_cost = output - expected_output;
    return neuron_cost * neuron_cost;
  }
  
  // Calculates the cost of a single data point.
  double calculate_data_point_cost (Data_Point data_point) {
    vector<double> outputs = calculate_network_outputs(data_point.inputs);
    double cost = 0;
    
    for (const auto &output : outputs) {
      // &output - &outputs[0] gives the index of the loop.
      cost += calculate_neuron_cost(output, data_point.outputs[&output - &outputs[0]]);
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
  
  // Applies the gradient of all weights and biases for all layers.
  void apply_all_gradients (const double learn_rate) {
    for (auto &layer : layers) {
      layer.apply_gradients(learn_rate);
    }
  }
  
  // Runs a single iteration of gradient descent not using calculus.
  void learn (Data_Point training_data[], const int num_data_points, const double learn_rate) {
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
  for (int i = 0; i < num_iterations; i++) {
    my_network.learn(data, num_data_points, 1);
    if (i % 500 == 0) {
      cout << num_iterations - i << " " << my_network.calculate_average_data_cost(data, num_data_points) << "\n";
    }
  }

  cout << my_network.calculate_average_data_cost(data, num_data_points) << "\n";

  while (1) {
    cin >> inputs[0];
    cin >> inputs[1];
    cout << (my_network.calculate_network_outputs(inputs)[0] > 0.5 ? 1 : 0) << "\n";
  }
  
  return 0;
}
