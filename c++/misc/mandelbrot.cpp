#include <iostream>
#include <complex>
#include <SDL2/SDL.h>

using namespace std;

int vertical_resolution = 100;
int horizontal_resolution = 100;
int iteration_number = 5000;
bool brightness_condition = false;

// Geting inputs from the user.
void get_inputs (void)
{ 
  string input;
  cout << "If you want the value to be the default value, enter default\n";
  cout << "The default values are:\n";
  cout << "Vertical resolution: " << vertical_resolution << "\n";
  cout << "Number of Iterations: " << iteration_number << "\n";

  string brightness_string = brightness_condition? "true" : "false";
  cout << "Brightness mapping: " << brightness_string << "\n";
  
  cout << "Enter the vertical resolution: ";
  getline (cin, input);
  if (input != "default all") {
    if (input != "default") {
      stringstream (input) >> vertical_resolution;
      horizontal_resolution = vertical_resolution;
    }
    
    cout << "Enter the number of iterations: ";
    getline (cin, input);
    if (input != "default") {
      stringstream (input) >> iteration_number;
    }

    cout << "Would you like to have the brightness be displayed (yes/no): ";
    getline (cin, input);
    if (input == "yes") {
      brightness_condition = true;
    }
  }
}

int main ()
{
  // Setting up all the variables.
  bool in_set;
  int horizontal_stretch = 1;
  int brightness;
  char brightness_text[] = {'%', '#', 'F', '*', '+', '=', '-', ':', '.', ' '};
  get_inputs();
  complex <double> z[iteration_number + 1];
  complex <double> top_left = -2.1 + 1.5i;
  complex <double> bottom_right = 0.9 - 1.5i;
  complex <double> current_coordinate = top_left;
  
  float height = top_left.imag() - bottom_right.imag();
  float width = bottom_right.real() - top_left.real();  
  double step_x = width/(horizontal_resolution * horizontal_stretch);
  double step_y = height/vertical_resolution;
  int pixel_x, pixel_y;
  
  // Setting up the screen.
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Could not init SDL: %s\n", SDL_GetError());
    return 1;
  }
  SDL_Window *screen = SDL_CreateWindow ("My application", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, horizontal_resolution, vertical_resolution, 0);
  if (!screen) {
    fprintf (stderr, "Could not create window\n");
    return 1;
  }
  SDL_Renderer *renderer = SDL_CreateRenderer(screen, -1, SDL_RENDERER_SOFTWARE);
  if (!renderer) {
    fprintf (stderr, "Could not create renderer\n");
    return 1;
  }

  
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderPresent(renderer);
  
  z[0] = 0;
  
  // Calculating the set.
  for (current_coordinate.imag(top_left.imag()), pixel_y = 1; current_coordinate.imag() > bottom_right.imag(); current_coordinate -= step_y * 1i, pixel_y++) {
    for (current_coordinate.real(top_left.real()), pixel_x = 1 ; current_coordinate.real() < bottom_right.real(); current_coordinate += step_x, pixel_x++) {
      in_set = true;
      brightness = 0;
      // Iterating the coordinates.
      for (int i = 0; i < iteration_number; i ++) {
	// Calculating the next coordinate.
	z[i + 1] = z[i] * z[i] + current_coordinate;

	if (brightness < 255) {
	  brightness ++;
	}
	
	// Checking if the coordinate is outside of the set.
	if (abs(z[i + 1]) > 2) {
	  in_set = false;
	  break;
	}
      }
      if (in_set) {
	SDL_RenderDrawPoint(renderer, pixel_x, pixel_y);
	//cout << "@";
      } else if (brightness_condition) {
	//cout << brightness_text[brightness - 1];
      } else {
	//cout << " ";
      }
    }
    SDL_RenderPresent(renderer);
    // cout << "\n";
  }
  
  SDL_RenderPresent(renderer);
  SDL_Delay(10000);  
  SDL_DestroyWindow(screen);
  SDL_Quit();
  
  return 0;
}
