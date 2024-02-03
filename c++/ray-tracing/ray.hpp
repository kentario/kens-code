#include <iostream>

#include "vector3.hpp"

#ifndef __RAY_HPP_
#define __RAY_HPP_

class Ray {
private:
  Vector3 start;
  Vector3 slope;
public:
  Ray () : start(), slope() {}
  
  Ray (Vector3 origin, Vector3 direction) : start(origin), slope(direction) {
    slope.normalize();
  }

  Vector3 origin () const {return start;}
  Vector3 direction () const {return slope;}

  void set_origin (Vector3 origin) {start = origin;}
  void set_direction (Vector3 direction) {slope = direction;}

  Vector3 point_at_distance (const double &distance) const {
    return slope * distance + start;
  }
};

inline std::ostream& operator<< (std::ostream &os, const Ray &ray) {
  os << "Origin: " << ray.origin() << " Direction: " << ray.direction();
  return os;
}

#endif // __RAY_HPP_ not defined
