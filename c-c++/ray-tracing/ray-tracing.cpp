#include <iostream>
#include <fstream>
#include <vector>

#include "camera.hpp"
#include "sphere.hpp"
#include "plane.hpp"

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
  //                 origin, look_direction, horizontal_fov, horizontal_resolution, vertical_resolution.
  Camera my_camera{{0, 0, 0}, {1, 0, 0},              90,             800,                 800};
  
  Vector3 sphere_color {31, 107, 220};
  Vector3 sphere_center {100, 0, 0};
  double sphere_radius = 50;
  
  std::vector<Sphere> spheres;
  spheres.push_back(Sphere{sphere_color, sphere_center, sphere_radius});
  
  sphere_color = {230, 32, 158};
  sphere_center += {25, 75, 25};
  sphere_radius = 50;
  
  spheres.push_back(Sphere{sphere_color, sphere_center, sphere_radius});
  
  std::vector<Shape*> shapes;
  
  for (auto &sphere : spheres) {
    shapes.push_back(&sphere);
  }
  
  std::vector<std::vector<Vector3>> colors = my_camera.take_picture(shapes);
  
  std::string file_name {"example.ppm"};
  twod_vector_to_ppm(colors, file_name);
  
  return 0;
}
