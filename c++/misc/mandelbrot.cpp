#include <iostream>
#include <complex>
#include <cstring>

#include <SDL2/SDL.h>

using namespace std::complex_literals;

int main (int argc, char *argv[]) {
  int vertical_resolution = 300;
  int horizontal_resolution = 300;
  int max_num_iterations = 10;
  bool brightness_displayed = false;
  
  if (argc != 5 && argc != 2) {
    std::cout << "Correct usages:\n" << argv[0] << " default\nfor default values, or:\n" << argv[0] << " <Vertical resolution> <Horizontal resolution> <Max nun iterations> <Brightness displayed yes/no>\n";
    exit(EXIT_FAILURE);
  } else if (std::strcmp(argv[1], "default") != 0) {
    vertical_resolution = std::stoi(argv[1]);
    horizontal_resolution = std::stoi(argv[2]);
    max_num_iterations = std::stoi(argv[3]);
    // 1 if yes, 0 if anything else.
    brightness_displayed = !std::strcmp(argv[4], "yes");
  }

  // Setting up all the variables.
  bool in_set;
  int horizontal_stretch = 1;
  int brightness;
  char brightness_text[] = {'%', '#', 'F', '*', '+', '=', '-', ':', '.', ' '};

  std::complex<double> z[max_num_iterations + 1];
  std::complex<double> top_left = -2.1 + 1.5i;
  std::complex<double> bottom_right = 0.9 - 1.5i;
  std::complex<double> current_coordinate = top_left;
  
  float height = top_left.imag() - bottom_right.imag();
  float width = bottom_right.real() - top_left.real();  
  double step_x = width/(horizontal_resolution * horizontal_stretch);
  double step_y = height/vertical_resolution;
  int pixel_x, pixel_y;
  
  // Setting up the screen.
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "Could not init SDL: %s\n", SDL_GetError());
    exit(EXIT_FAILURE);
  }
  SDL_Window *screen = SDL_CreateWindow ("My application", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, horizontal_resolution, vertical_resolution, 0);
  if (!screen) {
    fprintf (stderr, "Could not create window\n");
    exit(EXIT_FAILURE);
  }
  SDL_Renderer *renderer = SDL_CreateRenderer(screen, -1, SDL_RENDERER_SOFTWARE);
  if (!renderer) {
    fprintf (stderr, "Could not create renderer\n");
    exit(EXIT_FAILURE);
  }

  
  for (int i = 0; i < max_num_iterations; i++) {
    int num_iterations = i;
    
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
	for (int i = 0; i < num_iterations; i ++) {
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
	  //std::cout << "@";
	} else if (brightness_displayed) {
	  //std::cout << brightness_text[brightness - 1];
	} else {
	  //std::cout << " ";
	}
      }
      // std::cout << "\n";
    }
  
    SDL_RenderPresent(renderer);
    SDL_Delay(1000);
  }
  
  SDL_DestroyWindow(screen);
  SDL_Quit();
  
  return 0;
}
