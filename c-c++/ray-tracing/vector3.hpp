#include <iostream>
#include <cmath>

#ifndef __VECTOR3_HPP_
#define __VECTOR3_HPP_

class Vector3 {
public:
  double v[3];

  Vector3 () : v{0, 0, 0} {}
  Vector3 (double a, double b, double c) : v{a, b, c} {}
  Vector3 (const Vector3 &a) : v{a[0], a[1], a[2]} {}

  inline double x () const {return v[0];}
  inline double y () const {return v[1];}
  inline double z () const {return v[2];}
  inline double r () const {return v[0];}
  inline double g () const {return v[1];}
  inline double b () const {return v[2];}

  // +Vector3
  inline Vector3 operator+ () {return *this;}
  // -Vector3
  inline Vector3 operator- () {return Vector3(-v[0], -v[1], -v[2]);}
  
  inline double operator[] (int i) const {return v[i];}
  inline double& operator[] (int i) {return v[i];}
  
  inline Vector3& operator+= (const Vector3 &other);
  inline Vector3& operator-= (const Vector3 &other);
  inline Vector3& operator*= (const Vector3 &other);
  inline Vector3& operator/= (const Vector3 &other);
  
  inline Vector3& operator*= (const double &value);
  inline Vector3& operator/= (const double &value);

  inline Vector3& operator= (const Vector3 &other);

  inline double distance_squared () const {return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];}
  inline double distance () const {return std::sqrt(distance_squared());}

  inline void normalize ();

  inline Vector3 cross (const Vector3 &b) const;
};

inline std::ostream& operator<< (std::ostream &os, const Vector3 &a) {
  os << a[0] << " " << a[1] << " " << a[2];
  return os;
}

// (Vector3 + Vector3) -> Vector3
inline Vector3 operator+ (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

// (Vector3 - Vector3) -> Vector3
inline Vector3 operator- (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// (Vector3 * Vector3) -> Vector3
inline Vector3 operator* (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

// (Vector3 * value) -> Vector3
inline Vector3 operator* (const Vector3 &a, const double &value) {
  return Vector3(a[0] * value, a[1] * value, a[2] * value);
}

// (value * Vector3) -> Vector3
inline Vector3 operator* (const double &value, const Vector3 &a) {
  return Vector3(value * a[0], value * a[1], value * a[2]);
}

// (Vector3/Vector3) -> Vector3
inline Vector3 operator/ (const Vector3 &a, const Vector3 &b) {
  return Vector3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}

// (Vector3/value) -> Vector3
inline Vector3 operator/ (const Vector3 &a, const double &value) {
  return Vector3(a[0]/value, a[1]/value, a[2]/value);
}

// (value/Vector3) -> Vector3
inline Vector3 operator/ (const double &value, const Vector3 &a) {
  return Vector3(value/a[0], value/a[1], value/a[2]);
}

inline Vector3& Vector3::operator+= (const Vector3 &other) {
  v[0] += other[0];
  v[1] += other[1];
  v[2] += other[2];
  return *this;
}

inline Vector3& Vector3::operator-= (const Vector3 &other) {
  v[0] -= other[0];
  v[1] -= other[1];
  v[2] -= other[2];
  return *this;
}

inline Vector3& Vector3::operator*= (const Vector3 &other) {
  v[0] *= other[0];
  v[1] *= other[1];
  v[2] *= other[2];
  return *this;
}

inline Vector3& Vector3::operator/= (const Vector3 &other) {
  v[0] /= other[0];
  v[1] /= other[1];
  v[2] /= other[2];
  return *this;
}

inline Vector3& Vector3::operator*= (const double &value) {
  for (int i = 0; i < 3; i++) {
    v[i] *= value;
  }
  return *this;
}

inline Vector3& Vector3::operator/= (const double &value) {
  for (int i = 0; i < 3; i++) {
    v[i] /= value;
  }
  return *this;
}

inline Vector3& Vector3::operator= (const Vector3 &other) {
  v[0] = other[0];
  v[1] = other[1];
  v[2] = other[2];
  return *this;
}

inline void Vector3::normalize () {
  const double length = distance();

  if (!(length == 0 || length == 1)) {
    v[0] /= length;
    v[1] /= length;
    v[2] /= length;
  }
}

inline Vector3 normalize (const Vector3 &a) {
  const double length = a.distance();

  return Vector3{a[0]/length, a[1]/length, a[2]/length};
}

inline double dot (const Vector3 &a, const Vector3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vector3 Vector3::cross (const Vector3 &b) const {
  double output_x = y() * b.z() - z() * b.y();
  double output_y = z() * b.x() - x() * b.z();
  double output_z = x() * b.y() - y() * b.x();

  return Vector3(output_x, output_y, output_z);
}

inline Vector3 cross (const Vector3 &a, const Vector3 &b) {
  double x = a.y() * b.z() - a.z() * b.y();
  double y = a.z() * b.x() - a.x() * b.z();
  double z = a.x() * b.y() - a.y() * b.x();
  
  return Vector3(x, y, z);
}

#endif // __VECTOR3_HPP_ not defiend
