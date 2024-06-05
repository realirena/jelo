## cluster file

rm(list=ls())
library(rstan)
library(dplyr)
library(reshape2)
library(splines)
library(MASS)
options(mc.cores = parallel::detectCores(logical= FALSE))
# rstan_options(auto_write = TRUE)
slurm_arrayid <- Sys.getenv('SLURM_ARRAY_TASK_ID')

taskid <- as.numeric(slurm_arrayid)

#setwd("/home/irena/")
## set up the data simulation parameters:
P = 2 # no of basis functions
I = 200 # no of subjects ## increase subjects

alpha <- c(0,-2)

Sigma <- matrix(c(1, -0.05, -0.05, 0.5), ncol=P, nrow=P)

compiled_model <- stan_model("joint_model.stan")
seed = 10*taskid + 128

set.seed(seed)
S <- lapply(1:I, function(x){
  q = rnorm(1,0, 0.75)/2
  sigma_x = exp(q)
  return(sigma_x)
})

Ti = sample(3:12, I, replace=T) ##no visits per subject

## placeholder for the simulated time to FMPs
time_fmp = rep(0, sum(Ti))

ids <- unlist(lapply(seq_along(Ti), function(i) rep(i, Ti[i])))
N <- length(ids)
## simulate the time variable
range01 <- function(x){(x-min(x))/(max(x)-min(x))}

for(i in 1:I){
  if(i==1){
    time_fmp = range01((seq(1:Ti[i])  + abs(rnorm(Ti[i],0,0.5)))) ## addnoise
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
  mat <- mvrnorm(1, alpha, Sigma)
  return(mat)
})

## compute the means of the hormones
sim_mu <- t(sapply(seq(nrow(f_time_fmp)), function(i){
  B_i <- B[[ids[i]]]
  return(B_i %*% f_time_fmp[i,])
}))

### generate the hormone data
sim_x <- t(sapply(seq_along(ids), function(i){rnorm(1, sim_mu[i], S[[ids[i]]])}))
x_pred <- drop(sim_x)

## unlist B into a matrix with nrow = I and ncol = P
B_design <- matrix(unlist(B), ncol=P, byrow=T)
B_design = cbind(1, B_design)

## unroll S into a design matrix:
S_design <- matrix(unlist(S), byrow=T)

B_S_data = cbind(B_design, S_design)


## set the true coefficient parameters for the means and variances
beta_out <- c(5, 1, 2, -4)
beta_time_out <- c(-1, 1, 0.5, -0.5)

ran_eff_sigma <- 0.5
ran_eff <- sapply(seq_along(1:I), function(i){rnorm(1, 0,ran_eff_sigma)})
## get the mean of the outcome variable:
mu_outcome = sapply(seq_along(1:N), function(n){
B_S_data[ids[n],]%*%beta_out  + (B_S_data[ids[n],]%*%beta_time_out)*f_time_fmp[n,2]+ran_eff[ids[n]]
})

## set the variance of the outcome:
outcome_sigma <- 0.1

### generate the outcome data
sim_outcome <- sapply(seq_along(1:N), function(n){rnorm(1, mu_outcome[n],outcome_sigma)})

  sim_out  <- sampling(compiled_model,
                        # include = TRUE,
                        sample_file=paste0('rep_',taskid, '_model_samples.csv'),
                        iter = 2000,
                        warmup=1000, #BURN IN
                        chains = 2,
                        seed = seed,
                        control = list(max_treedepth = 40,
                                       adapt_delta=0.98),
                        data = list(
                                    P = P,
                                    I = I,
                                    N = length(time_fmp),
                                    id=ids,
                                    f_time = f_time_fmp,
                                    x_pred =x_pred,
                                    y  = sim_outcome,
                                    simulate = 0
                        ))

