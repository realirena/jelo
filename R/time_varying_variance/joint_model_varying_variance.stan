
data {
  int<lower=2> P; // df for basis expansion on time_fmp
  int<lower=2>P_Si;
  int<lower=1> N; // no of data points 
  int<lower=1> I; //  no of subjects
  int<lower=1> id[N]; //array of subject ids (length N in order to do the longitudinal estimation)
  // data for the longitudinal submodel:
  matrix[N, P] f_time; // basis expansion on time 
  vector[N] x_pred; // this should be X_i at time t (Q-length vector of hormone values)
  //data for the outcome submodel:
   vector[N] y;
  // Whether or not to evaluate the likelihood
  int<lower = 0, upper = 1> simulate; 
}

parameters {
  //for the longitudinal submodel 
  vector[P] alpha; // mean on the B_ijs 
  //vector[P] alpha_raw;
  vector[P] B[I]; // a vectors of the individual parameters 
  // parameters for Sigma (for the covariance matrix of Bijs)
  corr_matrix[P] L_Omega;  // prior correlation 
  vector<lower=0>[P] tau;
  vector<lower=0>[P_Si] S_tau;
  vector[P_Si] log_S[I];
  //parameters for individual variances 
  vector[P_Si] S_mu;
  corr_matrix[P_Si] S_Omega;
  vector[I] raw_ran_eff;
  real<lower=0, upper=pi()/2> ran_tau_unif;
  real beta_out_int;
  real beta_time_int;
  real beta_B_out; //coefficients for the mean 
  real beta_S_out;
  real beta_B_time; //coefficients for the (co)variances, size of lower triangle of S
  real beta_S_time;
  real<lower=0> outcome_sigma; //parameter for the variance of the outcomes
}

transformed parameters{
  //v-cov parameters for the B's 
 // matrix[I,P+1] B_data;
  matrix[P,P] Sigma;
  matrix[P_Si, P_Si] S_Sigma;
  vector[I] ran_eff;
  real<lower=0> ran_eff_tau;
  S_Sigma = quad_form_diag(S_Omega, S_tau);
  Sigma = quad_form_diag(L_Omega, tau);
  ran_eff_tau = 2.5*ran_tau_unif;
  ran_eff = ran_eff_tau*raw_ran_eff;

}

model {
  //set up the variance parameters for each beta: 
      // rather than put an inverse wishart on Sigma, follow stan recommendations for LKJ: 
   L_Omega ~ lkj_corr(1);
   tau ~ cauchy(0, 1);
   S_tau ~ cauchy(0, 0.5);
   // draw betas from a nmultinormal distribution
   alpha ~ normal(0,2.5); //put a WEAKLY informative prior
   S_mu[1] ~ normal(0, 1);
   S_mu[2] ~ normal(0, 1);
   S_Omega ~ lkj_corr(1);
 // outcome_sigma ~ cauchy(0, 2.5);
   beta_out_int ~ normal(0,2.5);
   beta_time_int ~ normal(0,2.5);
   beta_B_out ~ normal(0,2.5);
   beta_B_time ~ normal(0,2.5);
   beta_S_out ~ normal(0,2.5);
   beta_S_time ~ normal(0,2.5);
  
  raw_ran_eff ~ std_normal();
  for(i in 1:I) {
   log_S[i] ~ multi_normal(S_mu, S_Sigma); 
    B[i] ~ multi_normal(alpha, Sigma);
    }
  // this estimates the parameters using the time fmp and hormone data 
  {
  if(simulate == 0){ 
    vector[N] x_mu; // for each person, the mean of the predictor 
    vector[N] x_s; // for each person, the variance of the predictor
    vector[N] y_mu;
    for(n in 1:N){
      x_mu[n] = dot_product(B[id[n]], f_time[n,]);  // mean trend 
      x_s[n] = exp(dot_product(log_S[id[n]], f_time[n,])); // variance trend 
      y_mu[n] =  beta_out_int  +x_mu[n]*beta_B_out + x_s[n]*beta_S_out +  beta_time_int*f_time[n,2]  + ((x_mu[n]*beta_B_time)*f_time[n,2]) + (x_s[n]*beta_S_time)*f_time[n,2] + ran_eff[id[n]];
     }
     x_pred ~ normal(x_mu,sqrt(x_s));
     y ~ normal(y_mu, outcome_sigma);
   }
 } 
}





