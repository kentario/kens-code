#include <cmath>

#ifndef __PLANE_HPP_
#define __PLANE_HPP_

#include "shape.hpp"

class Plane : public Shape {
private:
  // Because the plane is infininte, it won't actually have a center, but this is usefull just for moving it around.
  Vector3 center;
  Vector3 normal;
public:
  Plane () : center{}, normal{1, 1, 1} {}

  Plane (Vector3 center, Vector3 normal) : center{center}, normal{normal} {}

  Plane (Vector3 color, Vector3 center, Vector3 normal) : Shape{color}, center{center}, normal{normal} {}

  Hit_Info hit (const Ray &ray, const bool debug) const;
};

Hit_Info Plane::hit (const Ray &ray, const bool debug) const {
  Hit_Info hit_info {false};
  
  return hit_info;
}

#endif // __PLANE_HPP_ not defined
