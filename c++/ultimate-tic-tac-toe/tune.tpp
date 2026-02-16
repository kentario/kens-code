#pragma once

#include "game.hpp"
#include "bots.hpp"
#include "heuristic.hpp"

template <Bot_T B>
requires requires (B bot, const Eval_Params params) { bot.set_params(params); }
Eval_Params evolve_eval_params_for_bot () {
  Eval_Params params {};

  return params;
}
