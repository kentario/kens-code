#include "tune.hpp"

#include <random>

#include "heuristic.hpp"

double random_double (const double min, const double max) {
  static std::default_random_engine generator;
  std::uniform_real_distribution<double> distr {min, max};

  return distr(generator);
}

Eval_Params tweak_params (const Eval_Params &in) {
  Eval_Params res {in};
  
  res.center += random_double(-1, 1);
  res.corner += random_double(-1, 1);
  res.edge += random_double(-1, 1);
  res.move_weight += random_double(-1, 1);
  res.win_options_weight += random_double(-1, 1);
  res.incomplete_subboard_weight += random_double(-1, 1);
  
  return res;
}
