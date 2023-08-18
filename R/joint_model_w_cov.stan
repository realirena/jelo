
data {
  int<lower=2> P; // df for basis expansion on time_fmp
  int<lower=1> N; // no of data points 
  int<lower=1> I; //  no of subjects
  int<lower=1> id[N]; //array of subject ids (length N in order to do the longitudinal estimation)
  int<lower=1> n_cov;
  // data for the longitudinal submodel:
  matrix[N, P] f_time; // basis expansion on time 
  vector[N] x_pred; // this should be X_i at time t (Q-length vector of hormone values)
  //data for the outcome submodel:
  vector[N] y;
  matrix[N,n_cov] other_covariates;
  // Whether or not to evaluate the likelihood
  int<lower = 0, upper = 1> simulate; 
}

parameters {
  //for the longitudinal submodel 
  vector[P] alpha; // mean on the B_ijs 
  vector[P] B[I]; // a vectors of the individual parameters 
  // parameters for Sigma (for the covariance matrix of Bijs)
  corr_matrix[P] L_Omega;  // prior correlation 
  vector<lower=0>[P] tau;
  real log_indiv_sigma[I]; 
  real ran_eff_raw[I];
  real<lower=0, upper=pi()/2> ran_tau_unif;
  //parameters for individual variances 
  real hyper_mu;
  real <lower=0> hyper_sigma;
 //for the outcome submodel 
  real beta_out_int;
  real beta_time_int;
  real  beta_B_out; //coefficient for the mean (xhat)
  real beta_S_out;
  real beta_B_time; //coefficient for the mean (xhat) with time
  real beta_S_time;
  real<lower=0> outcome_sigma; //parameter for the variance of the outcomes
  vector[n_cov] phi; 
}

transformed parameters{
  real<lower=0> indiv_sigma[I];
  real<lower=0> ran_eff_tau;
  real ran_eff [I];
  matrix[P,P] Sigma;
  ran_eff_tau  = 2.5*tan(ran_tau_unif);
  Sigma = quad_form_diag(L_Omega, tau);
  for (i in 1:I){
     ran_eff[i] = ran_eff_tau*ran_eff_raw[i];
     indiv_sigma[i] = exp(log_indiv_sigma[i]);
  }

}

model {
  tau ~ cauchy(0,2.5);
  L_Omega ~ lkj_corr(1);
  alpha ~ normal(0,10); 
  hyper_mu ~ normal(0, 10);
  hyper_sigma ~ cauchy(0,1);
  //priors on outcome coefficients
   beta_out_int ~ normal(0,10);
   beta_time_int ~ normal(0,10);
   beta_B_out ~ normal(0,10);
   beta_B_time ~ normal(0,10);
   beta_S_out ~ normal(0,10);
   beta_S_time ~ normal(0,10);
   phi ~ normal(0,10);
   outcome_sigma ~ cauchy(0,2.5); //weak prior on the sigma for outcome model
  for(i in 1:I) {
   log_indiv_sigma[i] ~ normal(hyper_mu, hyper_sigma); 
    ran_eff_raw[i] ~ std_normal();
    B[i] ~ multi_normal(alpha, Sigma);
    }
  {
  if(simulate == 0){ 
    real x_mu[N]; 
    real y_mu[N];
    for(n in 1:N){
      x_mu[n] = dot_product(B[id[n]], f_time[n,]);
      x_pred[n] ~ normal(x_mu[n], sqrt(indiv_sigma[id[n]]));
       y_mu[n] =  beta_out_int  + x_mu[n]* beta_B_out + indiv_sigma[id[n]]*beta_S_out +  beta_time_int*f_time[n,2]  + (x_mu[n]* beta_B_time*f_time[n,2]) + indiv_sigma[id[n]]*beta_S_time*f_time[n,2] + dot_product(other_covariates[n,], phi)+ ran_eff[id[n]];
       y[n] ~ normal(y_mu[n], outcome_sigma);
     }
   }
 } 
}

 generated quantities {
      vector[N] sim_x;
      vector[N] x_mu;
      vector[N] sim_y;
      vector[N] sim_y_mu;
//   //   simulate hormone data instead of using the data to generate it
      for(n in 1:N){ 
        x_mu[n] = dot_product(B[id[n]], f_time[n,]);
        sim_x[n] = normal_rng(x_mu[n], sqrt(indiv_sigma[id[n]]));
        sim_y_mu[n] =  beta_out_int  + x_mu[n]* beta_B_out + indiv_sigma[id[n]]*beta_S_out +  beta_time_int*f_time[n,2]  + (x_mu[n]* beta_B_time*f_time[n,2]) + indiv_sigma[id[n]]*beta_S_time*f_time[n,2]+ dot_product(other_covariates[n,], phi) + ran_eff[id[n]];
        sim_y[n] = normal_rng(sim_y_mu[n], outcome_sigma);
   }
 }

