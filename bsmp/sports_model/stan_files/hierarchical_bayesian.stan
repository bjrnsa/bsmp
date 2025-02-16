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
  real<lower=0.1> sigma_att; // OFF sigma
  real<lower=0.1> sigma_def; // DEF sigma
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
  sigma_att ~ cauchy(0, 2); // Sigma OFF prior
  sigma_def ~ cauchy(0, 2); // Sigma DEF prior
  
  // Priors for team abilities
  att_raw ~ normal(0, sigma_att); // Raw OFF prior
  def_raw ~ normal(0, sigma_def); // Raw DEF prior
  
  // Likelihood
  home_goals ~ poisson_log(home_advantage + att[home_team] + def[away_team]
                           + goal_mean);
  away_goals ~ poisson_log(att[away_team] - def[home_team] + goal_mean);
}
