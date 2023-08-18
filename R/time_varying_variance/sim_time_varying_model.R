## cluster file 

rm(list=ls())
library(rstan)
library(dplyr)
library(reshape2)
library(splines)
library(MASS)
options(mc.cores = parallel::detectCores(logical= FALSE))
rstan_options(auto_write = TRUE)
slurm_arrayid <- Sys.getenv('SLURM_ARRAY_TASK_ID')

taskid <- as.numeric(slurm_arrayid)

#setwd("/home/irena/")
## set up the data simulation parameters:
P = 2 # no of basis functions
P_Si = 2
I = 500  # no of subjects ## increase subjects

alpha <- c(0,-2)

Sigma <- matrix(c(1, -0.25, -0.25, 0.5), ncol=P, nrow=P)

S_mu <- c(0, 0)
S_Sigma <- matrix(c(0.5, -0.01 ,-0.01, 0.05), ncol=P, nrow=P)
seed = 301 + taskid*28
set.seed(seed)

Ti = sample(3:12, I, replace=T) ##no visits per subject

## placeholder for the simulated time to FMPs
time_fmp = rep(0, sum(Ti))

ids <- unlist(lapply(seq_along(Ti), function(i) rep(i, Ti[i])))
N <- length(ids)
## simulate the time variable
range01 <- function(x){(x-min(x))/(max(x)-min(x))}


for(i in 1:I){
  if(i==1){
    time_fmp =  range01(seq(1:Ti[i])  + abs(rnorm(Ti[i],0,0.5)))## addnoise
    # apply basis function to time fmp
    #f_time_fmp = bs(time_fmp, df=P)
    f_time_fmp = cbind(1, time_fmp)
  } else {
    time_fmp_i = range01(seq(1:Ti[i]) + abs(rnorm(Ti[i],0,0.5)))
    time_fmp = c(time_fmp, time_fmp_i)
    #f_time_fmp = rbind(f_time_fmp, bs(time_fmp_i, df=P))
    f_time_fmp = rbind(f_time_fmp, cbind(1, time_fmp_i))
  }
}

### Generate the individual level coefficients
B  <- lapply(1:I, function(x){
  mat <- MASS::mvrnorm(1, alpha, Sigma)
  return(mat)
})

log_S <- lapply(1:I, function(x){
 mat <- MASS::mvrnorm(1, S_mu, S_Sigma)
  return(mat)
})

## get the individual variances 

## compute the means of the hormones
sim_mu <- t(sapply(seq(nrow(f_time_fmp)), function(i){
  B_i <- B[[ids[i]]]
  return(B_i %*% f_time_fmp[i,])
}))


## compute the means of the hormones
sim_var <- t(sapply(seq(nrow(f_time_fmp)), function(i){
  S_i <- log_S[[ids[i]]]
  return(exp(S_i %*% f_time_fmp[i,]))
}))


### generate the hormone data
sim_x <- t(sapply(seq_along(ids), function(i){rnorm(1, sim_mu[i],sd=sqrt(sim_var[i]))}))
x_pred <- drop(sim_x) 
ind_var <- drop(sim_var)
ind_mu <- drop(sim_mu)


B_S_data = cbind(1, ind_mu, ind_var)
colnames(B_S_data) <- c("intercept", "pred_x", "pred_var")

beta_out <- c(2, -1.5, 0.25)
beta_out_time <- c(1, 0.75, -0.1)

ran_eff_tau <- 0.25
ran_eff <- sapply(seq_along(1:I), function(i){rnorm(1, 0,ran_eff_tau)})
## get the mean of the outcome variable: 
mu_outcome = sapply(seq_along(1:N), function(n){
  B_S_data[n,]%*%beta_out  + (B_S_data[n,]%*%beta_out_time)*f_time_fmp[n,2] + ran_eff[ids[n]]
})

outcome_sigma <- 0.1
sim_outcome <- sapply(seq_along(1:N), function(n){rnorm(1, mu_outcome[n],sd=outcome_sigma)})


compiled_model <- stan_model("/home/irena/gsra/joint_long_models/one_predictor/time_varying_variance/500_ids/joint_model_varying_variance.stan")
## sample from the model:
sim_out <- sampling(compiled_model,
                    sample_file=paste0("/home/irena/gsra/joint_long_models/one_predictor/time_varying_variance/500_ids/results/rep_", taskid, "_model_samples.csv"), #writes the samples to CSV file
                    iter = 2000,
                    warmup=1000, #BURN IN
                    chains = 2,
                    seed = seed,
                    control = list(max_treedepth = 50,
                                   adapt_delta=0.99),
                    data = list(
                      P = P,
                      P_Si = P_Si,
                      I = I,
                      N = length(time_fmp),
                      id=ids,
                      f_time = f_time_fmp,
                      x_pred =x_pred,
                      y=sim_outcome,
                      simulate = 0
                    ))

