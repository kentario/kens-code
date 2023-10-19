#include <iostream>
#include <random>

#ifndef __OBJECT_HPP_
#define __OBJECT_HPP_

#include "ray.hpp"

struct Material {
  Vector3 color;
};

inline std::ostream& operator<< (std::ostream &os, const Material &material) {
  os << "Color: " << material.color;
  return os;
}

struct Hit_Info {
  bool hit;
  double distance;
  Vector3 hit_point;
  Vector3 normal;
  Material material;

  // Returns true if the distance of this hit info is lower than the distance of the other hit info.
  bool is_closer_than (const Hit_Info &hit_info) const {
    return distance < hit_info.distance;
  }
};

inline std::ostream& operator<< (std::ostream &os, const Hit_Info &hit_info) {
  os << "Hit: " << (hit_info.hit ? "true" : "false") << " Distance: " << hit_info.distance << " Hit Point: " << hit_info.hit_point << " Normal: " << hit_info.normal << " Material: " << hit_info.material;
  return os;
}

class Shape {
protected:
  Material material;
public:
  Shape () : material{Vector3{0, 0, 0}} {}

  Shape (Material material) : material{material} {}
  
  virtual Hit_Info hit (const Ray &ray, const bool debug) const = 0;
};

// Returns a random number in the range [0, 1).
double random_number () {
  std::random_device generator;
  std::uniform_real_distribution<double> distribution{0.0, 1.0};
  
  return distribution(generator);
}

Vector3 rand_on_unit_sphere () {
  // Found the method here: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/#better-choice-of-spherical-coordinates
  double u {random_number()};
  double v {random_number()};
  double theta {u * 2 * M_PI};
  double phi {std::acos(2 * v - 1)};
  double r {std::pow(random_number(), 1/3.0)};
  double sin_theta {std::sin(theta)};
  double cos_theta {std::cos(theta)};
  double sin_phi {std::sin(phi)};
  double cos_phi {std::cos(phi)};
  double x {r * sin_phi * cos_theta};
  double y {r * sin_phi * sin_theta};
  double z {r * cos_phi};

  Vector3 point{x, y, z};
  return point;
}

#endif // __OBJECT_HPP_ not defined
