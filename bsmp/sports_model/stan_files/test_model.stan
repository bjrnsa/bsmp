data {
  int<lower=0> N; // Number of games
  int<lower=1> T; // Number of teams
  array[N] int home_team; // Home team index
  array[N] int away_team; // Away team index
  array[N] int home_goals; // Home goals
  array[N] int away_goals; // Away goals
}
parameters {
  real home_advantage; // Home advantage
  real goal_mean; // Goal mean
  real<lower=0.01> sigma_att; // OFF sigma
  real<lower=00.1> sigma_def; // DEF sigma
  vector[T] att_raw; // Raw OFF
  vector[T] def_raw; // Raw DEF
}
transformed parameters {
  vector[T] att; // centered OFF
  vector[T] def; // centered DEF
  
  // Center OFF & DEF to have mean zero
  att = att_raw - mean(att_raw);
  def = def_raw - mean(def_raw);
}
model {
  // Priors for global parameters
  home_advantage ~ normal(0, 1); // Home advantage prior
  goal_mean ~ normal(0, 1); // Goal mean prior
  
  // Priors for standard deviations
  sigma_att ~ normal(0, 1); // Sigma OFF prior
  sigma_def ~ normal(0, 1); // Sigma DEF prior
  
  // Priors for team abilities
  att_raw ~ normal(0, sigma_att); // Raw OFF prior
  def_raw ~ normal(0, sigma_def); // Raw DEF prior
  
  // Likelihood
  home_goals ~ poisson_log(home_advantage + att[home_team] + def[away_team]
                           + goal_mean);
  away_goals ~ poisson_log(att[away_team] - def[home_team] + goal_mean);
}
generated quantities {
  vector[N] log_lambda_home;
  vector[N] log_lambda_away;
  vector[N] lambda_home;
  vector[N] lambda_away;
  vector[N] home_goals_sim;
  vector[N] away_goals_sim;
  for (i in 1 : N) {
    log_lambda_home[i] = home_advantage + att[home_team[i]]
                         + def[away_team[i]] + goal_mean;
    log_lambda_away[i] = att[away_team[i]] - def[home_team[i]] + goal_mean;
    lambda_home[i] = exp(log_lambda_home[i]);
    lambda_away[i] = exp(log_lambda_away[i]);
    home_goals_sim[i] = poisson_rng(lambda_home[i]);
    away_goals_sim[i] = poisson_rng(lambda_away[i]);
  }
}
