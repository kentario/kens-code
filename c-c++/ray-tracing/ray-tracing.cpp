#include <iostream>
#include <fstream>
#include <vector>

void twod_vector_to_ppm (std::vector<std::vector<Ray>> &rays, std::string &file_name) {
  int width {static_cast<int>(rays.size())};
  int height {static_cast<int>(rays[0].size())};
  
  std::ofstream my_file;
  my_file.open(file_name);
  
  my_file << "P3\n" << rays.size() << " " << rays[0].size() << "\n255\n";

  // The top left of the image is [0][0] in the array, and the top right is [width - 1][0].
  for (int y {0}; y < height; y++) {
    for (int x {0}; x < width; x++) {
      //      my_file << 
    }
  }

  my_file.close();
}

int main () {
  /*  int width {1000};
      int height {1000};*/
  int width {256};
  int height {256};

  std::vector<std::vector<Ray>> rays;
  rays.resize(width);
  
  for (int x {0}; x < width; x++) {
    rays[x].resize(height);
    for (int y {0}; y < height; y++) {
      rays[x][y].red = x;
      rays[x][y].green = 0;
      rays[x][y].blue = 0;
    }
  }

  std::string file_name {"example.ppm"};
  twod_vector_to_ppm(colors, file_name);
  
  return 0;
}
