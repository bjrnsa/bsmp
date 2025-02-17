functions {
  int all_ones(int N, vector weights) {
    int all_ones = 1;
    for (n in 1 : N) {
      if (weights[n] != 1) {
        all_ones = 0;
        break;
      }
    }
    return all_ones;
  }
}
data {
  int<lower=0> N; // Number of games
  int<lower=1> T; // Number of teams
  array[N] int home_team; // Home team index
  array[N] int away_team; // Away team index
  array[N] int home_goals; // Home goals
  array[N] int away_goals; // Away goals
  vector[N] weights; // match weights
}
parameters {
  real home_advantage; // Home advantage
  real goal_mean; // Goal mean
  real<lower=0.1> sigma_attack; // OFF sigma
  real<lower=0.1> sigma_defence; // defence sigma
  vector[T] attack_raw; // Raw OFF
  vector[T] defence_raw; // Raw defence
}
transformed parameters {
  vector[T] attack;
  vector[T] defence;
  vector[N] log_lambda_home;
  vector[N] log_lambda_away;
  
  attack = attack_raw - mean(attack_raw);
  defence = defence_raw - mean(defence_raw);
  
  for (i in 1 : N) {
    log_lambda_home[i] = home_advantage + attack[home_team[i]]
                         + defence[away_team[i]] + goal_mean;
    log_lambda_away[i] = attack[away_team[i]] - defence[home_team[i]]
                         + goal_mean;
  }
}
model {
  // Priors for global parameters
  home_advantage ~ normal(0, 1); // Home advantage prior
  goal_mean ~ normal(0, 1); // Goal mean prior
  
  // Priors for standard deviations
  sigma_attack ~ cauchy(0, 2); // Sigma OFF prior
  sigma_defence ~ cauchy(0, 2); // Sigma defence prior
  
  // Priors for team abilities
  attack_raw ~ normal(0, sigma_attack); // Raw OFF prior
  defence_raw ~ normal(0, sigma_defence); // Raw defence prior
  
  if (all_ones(N, weights) == 1) {
    // Likelihood
    home_goals ~ poisson_log(home_advantage + attack[home_team]
                             + defence[away_team] + goal_mean);
    away_goals ~ poisson_log(attack[away_team] - defence[home_team]
                             + goal_mean);
  } else {
    // Likelihood with weights
    for (i in 1 : N) {
      target += weights[i]
                * poisson_log_lpmf(home_goals[i] | log_lambda_home[i])
                + weights[i]
                  * poisson_log_lpmf(away_goals[i] | log_lambda_away[i]);
    }
  }
}
