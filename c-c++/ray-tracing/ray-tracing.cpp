#include <iostream>
#include <fstream>
#include <vector>

#include "vector3.h"

void twod_vector_to_ppm (std::vector<std::vector<Vector3>> &colors, std::string &file_name) {
  int width {static_cast<int>(colors.size())};
  int height {static_cast<int>(colors[0].size())};
  
  std::ofstream my_file;
  my_file.open(file_name);
  
  my_file << "P3\n" << width << " " << height << "\n255\n";

  // The top left of the image is [0][0] in the array, and the top right is [width - 1][0].
  for (int y {0}; y < height; y++) {
    for (int x {0}; x < width; x++) {
      my_file << colors[x][y].r() << " " << colors[x][y].g() << " " << colors[x][y].b() << "\n";
    }
  }

  my_file.close();
}

int main () {
  //int width {1000};
  //int height {1000};
  int width {256};
  int height {256};

  std::vector<std::vector<Vector3>> colors;
  colors.resize(width);
  
  for (int x {0}; x < width; x++) {
    colors[x].resize(height);
    for (int y {0}; y < height; y++) {
      colors[x][y][0] = 3;
    }
  }

  std::string file_name {"example.ppm"};
  twod_vector_to_ppm(colors, file_name);
  
  return 0;
}
