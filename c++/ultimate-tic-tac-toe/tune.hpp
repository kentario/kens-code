#pragma once

#include "game.hpp"
#include "bots.hpp"
#include "heuristic.hpp"

double random_double (const double min, const double max);

Eval_Params tweak_params (const Eval_Params &in);

template <Bot_T T>
Eval_Params evolve_eval_params_for_bot (const size_t generations, const size_t gen_size);

#include "tune.tpp"
