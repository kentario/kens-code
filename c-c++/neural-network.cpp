#include <iostream>
#include <string.h>
#include <cmath>
#include <cstdlib>
#include <time.h>

using namespace std;

/*
The steps to this neural network:
Each neuron gets its corresponding bias added to it.
Each neuron in the next layer becomes the sum of all the neurons in the previous layer multiplied to their corresponding weights.
The activation function is applied to each neuron of the new layer.
Repeat for the next layer.

The biases are stored at the end of the weights, so if the weigts for a neruon were 5, 3, 2, 6, then 6 would be the bias.
*/

const int neurons_per_layer[] = {1, 2, 2, 3};
const int number_of_layers = sizeof(neurons_per_layer)/sizeof(neurons_per_layer[0]);

int b_max = 0;
int c_max = 0;
// Declare the variable that is used to access the network.
double *network;
//      layer, neuron, weight
#define NETWORK(a, b, c) network[a * b_max * c_max + b * c_max + c]

double sigmoid (double x) {
  return 1/(1 + exp(-x));
}

double sigmoid_derivative (double x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

double cost_function (double desired_output[], double real_output[]) {
  double cost = 0;
  for (int i = 0; i < neurons_per_layer[number_of_layers - 1]; i++) {
    cost += (real_output[i] - desired_output[i]) * (real_output[i] - desired_output[i]);
  }
  return cost;
}

void print_network () {
  cout << "\nInput Layer (Layer 0) has no weights.";
  // Loop through each layer in the network
  for (int current_layer = 1; current_layer < number_of_layers; current_layer++) {
    // Printing out the current layer number.
    cout << "\nLayer " << current_layer << ": ";
    // Loop through each neuron in the current layer.
    for (int current_neuron = 0; current_neuron < neurons_per_layer[current_layer]; current_neuron++) {
      // Print out the current neuron number.
      cout << " Neuron " << current_neuron << ": ";
      // Loop through each weight (and bias) of the current neuron.
      for (int current_weight = 0; current_weight < neurons_per_layer[current_layer- 1] + 1; current_weight++) {
	// Printing out the current weight (or bias).
	cout << NETWORK(current_layer, current_neuron, current_weight) << " ";
      }
    }
  }
  cout << "\n";
}

void initialize_network () {
  // Creating the weights from one layer to the next, the input layer has no weights.
  // Looping through each layer except the input layer.
  for (int current_layer = 1; current_layer < number_of_layers; current_layer++) {
    for (int current_neuron = 0; current_neuron < neurons_per_layer[current_layer]; current_neuron++) {
      // Looping through every input neuron, plus one.
      for (int input_neuron = 0; input_neuron < neurons_per_layer[current_layer- 1] + 1; input_neuron++) {
	// Assigning a random number between 0 and 1 as the weight and bias.
	NETWORK(current_layer, current_neuron, input_neuron) = (double)random()/(double)RAND_MAX;
      }
    }
  }
}

void forward_propagate (double input[], double *output_location) {
  double input_buffer[b_max];
  double output_buffer[b_max];
  bzero(input_buffer, sizeof(input_buffer));
  bzero(output_buffer, sizeof(output_buffer));
  for (int i = 0; i < neurons_per_layer[0]; i++) {
    input_buffer[i] = input[i];
  }
  
  // Propogate input through each network.
  for (int current_layer = 1; current_layer < number_of_layers; current_layer++) {
    // Loop through each neuron in the current layer.
    for (int current_neuron = 0; current_neuron < neurons_per_layer[current_layer]; current_neuron++) {
      // Add the bias.
      output_buffer[current_neuron] = NETWORK(current_layer, current_neuron, neurons_per_layer[current_layer] - 1);
      // Loop through each input.
      for (int current_input = 0; current_input < neurons_per_layer[current_layer- 1]; current_input++) {
	// Multiplying input by the corresponding weight.
	output_buffer[current_neuron] += NETWORK(current_layer, current_neuron, current_input) * input_buffer[current_input];
      }
      // Applying the activation function to each output.
      output_buffer[current_neuron] = sigmoid(output_buffer[current_neuron]);
    }
    // Transfer the output buffer into the input buffer.
    for (int i = 0; i < neurons_per_layer[current_layer]; i++) {
      input_buffer[i] = output_buffer[i];
    }
  }

  // Copy the output from the output buffer to the output location.
  for (int i = 0; i < neurons_per_layer[number_of_layers - 1]; i++) {
    output_location[i] = output_buffer[i];
  }
}

int main ()
{
  srand(time(0));

  int max_neurons_per_layer = -1;
  for (int i = 0; i < sizeof(neurons_per_layer)/sizeof(neurons_per_layer[0]); i++) {
    if (neurons_per_layer[i] > max_neurons_per_layer) {
      max_neurons_per_layer = neurons_per_layer[i];
    }
  }
  b_max = max_neurons_per_layer;
  c_max = max_neurons_per_layer + 1;
  
  // Allocate enough memory for the network.
  double network_alloc[number_of_layers][b_max][c_max];
  // Assign the variable used to access the network.
  network = (double *) network_alloc;
  
  initialize_network();
  
  // Printing out the network.
  print_network();

  // Crate input.
  double input[neurons_per_layer[0]] = {1, 0};
  // Create output.
  double output[neurons_per_layer[number_of_layers - 1]];
  // Propogate forward.
  forward_propagate(input, output);
  // Print the output.
  for (int i = 0; i < neurons_per_layer[number_of_layers - 1]; i++) {
    printf("Output[%d] = %.20f\n", i, output[i]);
  }

  return 0;
}
