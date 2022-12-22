#include <iostream>
#include <cmath>
#include <cstdlib>
#include <time.h>

using namespace std;

/*
The steps to this neural network:
Each neuron gets its corresponding bias added to it.
Each neuron in the next layer becomes all the neurons in the previous layer multiplied to there corresponding weights, added up.
The activation function is applied to each neuron of the new layer.
Repeat for the new layer.
*/

double learn_amount = 0.1; // The amount that the weights and biases will be adjusted by.

// 0 is an empty square, 1 is black, and -1 is white.
// The second to last number is the move's location, and the last number is the color of the move.
int input_layer[66] = {0, 0, 0, 0, 0, 0, 0, 0,
		       0, 0, 0, 0, 0, 0, 0, 0,
		       0, 0, 0, 0, 0, 0, 0, 0,
		       0, 0, 0, -1, 1, 0, 0, 0,
		       0, 0, 0, 1, -1, 0, 0, 0,
		       0, 0, 0, 0, 0, 0, 0, 0,
		       0, 0, 0, 0, 0, 0, 0, 0,
		       0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1};

double output[1]; // An output layer with one neuron.
double weight_input_output[65][1]; // Weights from the hidden layer to the output layer.
double output_bias[1]; // Biases on the output layer.

double leaky_ReLU (double x) {
  // Set the leaky ReLU slope parameter
  double alpha = 0.01;
  if (x > 0) {
    // If x is positive, return x
    return x;
  } else {
    // If x is negative, return alpha * x
    return alpha * x;
  }
}

void compute_output (int move)
{
  input_layer[64] = move;
  for (int j = 0; j < 65; j++) {
    output[0] += input_layer[j] * weight_input_output[j][0];
  }
  output[0] += output_bias[0];
  output[0] = leaky_ReLU(output[0]);
}

void change_parameters_randomly ()
{
  for (int i = 0; i < 65; i++) {
    if ((rand() % 2) == 1) {
      weight_input_output[i][0] += learn_amount;
    } else {
      weight_input_output[i][0] -= learn_amount;
    }
  }
  if ((rand() % 2) == 1) {
    output[0] += learn_amount;
  } else {
    output[0] -= learn_amount;
  }
}

int main ()
{
  srand(time(0));
  // Randomizing the weight_input_output values, and the output bias.
  for (int i = 0; i < 200; i++) {
    change_parameters_randomly();
  }
  
  // Computing the output for the move 19, and 26.
  compute_output(19);
  cout << "The evaluation of the move to the spot " << input_layer[64] << " with the color " << input_layer[65] << " is: " << output[0] << "\n";
  
  compute_output(26);
  cout << "The evaluation of the move to the spot " << input_layer[64] << " with the color " << input_layer[65] << " is: " << output[0] << "\n";
  return 0;
}
