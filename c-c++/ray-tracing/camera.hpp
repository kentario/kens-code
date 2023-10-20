#include <vector>
#include <cmath>
#include <limits>

#ifndef __CAMERA_HPP_
#define __CAMERA_HPP_

#include "ray.hpp"
#include "shape.hpp"

#define PI 3.141592653

double tan_degrees (const double x) {
  return tan(x * PI/180);
}

class Camera {
private:
  Vector3 origin;
  Vector3 look_direction;

  Vector3 up_direction;
  Vector3 left_direction;

  double horizontal_fov;
  
  double horizontal_resolution;
  double vertical_resolution;
  
  std::vector<std::vector<Ray>> rays;

  Hit_Info calculate_ray_collision (const std::vector<Shape*> &shapes, const Ray &ray) const;
public:
  Camera (Vector3 origin, Vector3 look_direction, double horizontal_fov, double horizontal_resolution, double vertical_resolution);

  // Returns a 2d vector of all the pixel color values.
  std::vector<std::vector<Vector3>> take_picture (const std::vector<Shape*> &shapes) const;
};

Camera::Camera (Vector3 origin, Vector3 look_direction, double horizontal_fov, double horizontal_resolution, double vertical_resolution) :
  origin{origin}, look_direction{look_direction},
  horizontal_fov{horizontal_fov},
  horizontal_resolution{horizontal_resolution}, vertical_resolution{vertical_resolution} {
  
  // To get the left direction, take the cross product of the look and the target up direction. a should be target up, b should be look_direction (see right hand rule).
  // The target up direction is the direction that I want the up direction to be closest to. In this project, I want the y axis to be up, so the target up direction will be (0, 1, 0), which is a unit vector pointing in the direction of the y-axis.
  left_direction = cross(Vector3(0, 1, 0), look_direction);
  
  // To get the up_direction, take the cross of the left and look directions. a is look_direction, b is left_direction (see right hand rule).
  up_direction = cross(look_direction, left_direction);
  
  // Normalize each direction to make it easier to deal with them later.
  look_direction.normalize();
  left_direction.normalize();
  up_direction.normalize();
  
  // The rays array is accessed by using rays[column][row]. The top left is rays[0][0].
  rays.resize(horizontal_resolution);
  
  // The top left Vector3 will start at the origin,
  Vector3 top_left{origin};
  // move forward in the look direction,
  top_left += look_direction;
  // in the left direction by half the angle of the horizontal fov.
  // To get the distance to move to the left:
  /*
    tan(fov/2) = left_distance/forward_length
    the forward length is look_direction.distance(), and since look_direction is normalized it is just 1.
    left_distance = tan(fov/2)
   */
  double width = 2 * tan_degrees(horizontal_fov/2);
  top_left += left_direction * width/2;
  // The top left is also half of the vertical fov up.
  // To get half of the vertical fov length:
  /*
    The ratio between the horizontal_fov and horizontal_resolution is equal to the ratio between the vertical_fov and the vertical_resolution.
    horizontal_fov/horizontal_resolution = vertical_fov/vertical_resolution
    cross multiplication:
     vertical_fov * horizontal_resolution = horizontal_fov * vertical_resolution
     vertical_fov = (horizontal_fov * vertical_resolution)/horizontal_resolution
   */
  double vertical_fov {(horizontal_fov * vertical_resolution)/horizontal_resolution};
  // Use the same method to get the vertical distance from the vertical fov as the horizontal distance from the horizontal fov.
  top_left += up_direction * tan_degrees(vertical_fov/2);

  // When iterating over each ray, it is necessary to have a grid of points that each ray passes through.
  // The distance between grid points is the width of the grid divided by the horizontal resolution.
  double pixel_spacing {width/horizontal_resolution};
  
  Vector3 direction;
  
  rays.resize(horizontal_resolution);
  for (int x {0}; x < horizontal_resolution; x++) {
    rays[x].resize(vertical_resolution);
    for (int y {0}; y < vertical_resolution; y++) {
      // The direction that the ray goes is to the top left, minus pixel_spacing times x times left_direction, minus pixel_spacing time x times up_direction.
      direction = top_left;
      direction -= pixel_spacing * x * left_direction;
      direction -= pixel_spacing * y * up_direction;
      
      rays[x][y] = Ray(origin, direction);      
    }
  }
}

Hit_Info Camera::calculate_ray_collision (const std::vector<Shape*> &shapes, const Ray &ray) const {
  unsigned long num_shapes {shapes.size()};

  // Anything will be closer than the max double.
  Hit_Info closest {false, std::numeric_limits<double>::max()};

  // Loop over each shape and find if it hits the ray.
  Hit_Info temp_hit_info;
  for (const auto *shape : shapes) {
    temp_hit_info = shape->hit(ray, false);

    if (temp_hit_info.hit) {
      // If the ray hits the shape, check if it is the closest hit.
      if (temp_hit_info.distance < closest.distance) {
	closest = temp_hit_info;
      }
    }
  }
  
  return closest;
}

std::vector<std::vector<Vector3>> Camera::take_picture (const std::vector<Shape*> &shapes) const {
  unsigned long num_shapes {shapes.size()};
  
  std::vector<std::vector<Vector3>> colors;
  colors.resize(horizontal_resolution);
  
  // Loop over each ray and find if it hits a shape.
  for (int x {0}; x < horizontal_resolution; x++) {
    colors[x].resize(vertical_resolution);
    for (int y {0}; y < vertical_resolution; y++) {
      colors[x][y] = calculate_ray_collision(shapes, rays[x][y]).material.color;
      /*      if (colors[x][y][0] != 0) {
	std::cout << "colors[" << x << "][" << y << "] = " << colors[x][y] << "\n";
	}*/
    }
  }
  
  return colors;
}

#endif // __CAMERA_HPP_ not defined
