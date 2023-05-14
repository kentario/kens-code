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

// Returns the input double passed through sigmoid sigmoid functin.
double sigmoid (double x) {
  return 1/(1 + exp(-x));
}

// Returns the input double passed through the derivative of the sigmoid function.
double sigmoid_derivative (double x) {
  double sigmoid_x = sigmoid(x);
  return sigmoid_x * (1 - sigmoid_x);
}

double random_number () {
  // I don't really know how this works.
  static default_random_engine generator;
  static uniform_real_distribution<double> distribution(0, 1.0);
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
  vector<vector<double>> weights;
  // weights[output neuron][input neuron]
  vector<double> biases;
  // biases[output neuron]
  int num_neurons;
  int num_inputs;
public:
  // Layer constructor.
  Layer (int num_inputs, int num_neurons) :
    num_neurons(num_neurons), num_inputs(num_inputs),
    weights(num_neurons, vector<double>(num_inputs)), biases(num_neurons) {}

  // Initialize the layers weights and biases randomly to a value between 0 and 1.
  void initialize_layer_randomly () {
    // Loop over every neuron that the weights output to.
    for (int output = 0; output < num_neurons; output++) {
      // Loop over every neuron that gets fed into the weights.
      for (int input = 0; input < num_inputs; input++) {
	// Randomize each weight.
	weights[output][input] = random_number();
      }
      // There will be one bias for each neuron that the weights output to.
      biases[output] = random_number();
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
      weighted_outputs[&output - &weights[0]] = sigmoid(weighted_outputs[&output - &weights[0]]);
    }
    
    return weighted_outputs;
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
  Network (int layer_sizes[], int num_layers, bool automatically_initialize_network) :
    num_layers(num_layers) {
    for (int i = 1; i < num_layers; i++) {
      // I starts at 1 because the input layer doesn't have any weights or biases, but the size of the input layer is recorded.
      // Add a layer to the end of the list of layers.
      layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
    }
    if (automatically_initialize_network) {
      initialize_network_randomly();
    }
  }

  // Initializes each weight and bias of the network to be a random number between 0 and 1.
  void initialize_network_randomly () {
    for (auto &layer : layers) {
      // For each layer, randomize its weights and biases.
      layer.initialize_layer_randomly();
    }
  }

  // Calculates the outupts of the network from a given input.
  vector<double> calculate_network_outputs (vector<double> &inputs) {
    for (int layer = 0; layer < num_layers - 1; layer++) {
      // Loops over every layer - 1, calculates the outputs of that layer, and that output becomes the new input for the next layer.
      // I use num_layers - 1 because I don't have to calculate anything for the input layer.
      inputs = layers[layer].calculate_layer_outputs(inputs);
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
  double calculate_average_data_cost (Data_Point data[], int num_data_points) {
    double total_cost = 0;

    for (int data_point = 0; data_point < num_data_points; data_point++) {
      total_cost += calculate_data_point_cost(data[data_point]);
    }

    return total_cost/num_data_points;
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

int main() {
  int layer_sizes[] = {2, 3, 3, 1};
  int num_layers = sizeof(layer_sizes)/sizeof(layer_sizes[0]);
  Network my_network(layer_sizes, num_layers, true);

  my_network.print_network();
  vector<double> input = {1, 2};
  vector<double> output;
  output = my_network.calculate_network_outputs(input);
  cout << output[0] << "\n";
  
  return 0;
}
