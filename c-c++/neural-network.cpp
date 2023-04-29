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

double sigmoid (double x) {
  return 1/(1 + exp(-x));
}

double sigmoid_derivative (double x) {
  double sigmoid_x = sigmoid(x);
  return sigmoid_x * (1 - sigmoid_x);
}

double random_number () {
  // I don't really know how this works, but apparently it works well.
  static default_random_engine generator;
  static uniform_real_distribution<double> distribution(0, 1.0);
  return distribution(generator);
}

class layer {
private:
  vector<vector<double>> weights;
  vector<double> biases;
  int num_neurons;
  int num_inputs;
public:
  layer (int num_inputs, int num_neurons) :
    num_neurons(num_neurons), num_inputs(num_inputs),
    weights(num_neurons, vector<double>(num_inputs)), biases(num_neurons) {}
  
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
  
  vector<double> calculate_layer_outputs (const vector<double> inputs) {
    vector<double> weighted_outputs;
    weighted_outputs.resize(num_neurons);
    for (int output = 0; output < num_neurons; output++) {
      // Loop over every weighted output.
      // Set the weighted output as the bias because it is the same as just adding the bias to 0.
      weighted_outputs[output] = biases[output];
      for (int input = 0; input < num_inputs; input++) {
	// Loops over each input, and multiply it by its corresponding weight, and then add it to the weighted output.
	weighted_outputs[output] += inputs[input] * weights[output][input];
      }
      // Once the weighted outputs are calculated, apply the activation function.
      weighted_outputs[output] = sigmoid(weighted_outputs[output]);
    }
    return weighted_outputs;
  }
  
  void print_layer () {
    for (const auto &output : weights) {
      // This for loop loops over every neuron that this layer of weights outputs to.
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

class network {
private:
  vector<layer> layers;
  int num_layers;
public:
  network (int layer_sizes[], int num_layers) :
    num_layers(num_layers) {
    for (int i = 1; i < num_layers; i++) {
      // I starts at 1 because the input layer doesn't have any weights or biases, but the size of the input layer is recorded.
      // Add a layer to the end of the list of layers.
      layers.push_back(layer(layer_sizes[i - 1], layer_sizes[i]));
    }
  }
  
  void initialize_network_randomly () {
    for (auto &layer : layers) {
      // For each layer, randomize its weights and biases.
      layer.initialize_layer_randomly();
    }
  }
  
  void calculate_network_outputs (vector<double> &inputs) {
    for (int layer = 0; layer < num_layers - 1; layer++) {
      // num_layers - 1 because we don't have to calculate anything for the input layer.
      // Loops over every layer - 1, calculates the outputs of that layer, and that output becomes the new input for the next layer.
      inputs = layers[layer].calculate_layer_outputs(inputs);
    }
  }
  
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
  int layer_sizes[] = {2, 3, 3, 2};
  int num_layers = sizeof(layer_sizes)/sizeof(layer_sizes[0]);
  network my_network(layer_sizes, num_layers);
  
  my_network.initialize_network_randomly();
  my_network.print_network();
  
  return 0;
}
