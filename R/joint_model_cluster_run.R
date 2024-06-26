## cluster file 

rm(list=ls())
library(rstan)
library(dplyr)
library(MASS)
options(mc.cores = parallel::detectCores(logical= FALSE))
rstan_options(auto_write = TRUE)

data_dir <- "U:/Documents/jelo/hydra/"
model_dir <- "U:/Documents/repos/jelo/R/"
results_dir <- "G:/irena/jelo/"
seed = 220
#setwd("/home/irena/")
## set up the data simulation parameters:
P = 2 # no of basis functions

model_data <- read.csv(paste0(data_dir, "prepped_bmd_e2_data_06042024.csv"))

ids <- model_data$new_id
I <- length(unique(ids))
time_fmp <- cbind(1, model_data$time_fmp_bone)

other_covs <- cbind(model_data$bmi_std, model_data$age_std) 

x_pred <- model_data$lag_fsh
y <- model_data$bmd_resid
 
compiled_model <- stan_model(paste0(model_dir, "joint_model_w_cov.stan"))

 ## sample from the model:
sim_out <- sampling(compiled_model,
                      sample_file=paste0(results_dir, '0604_fsh_model_samples.csv'), #writes the samples to CSV file
                      iter = 4000,
                      warmup=2000, #BURN IN
                      chains =4,
                      save_warmup=FALSE,
                      seed = seed,
                      control = list(max_treedepth = 60,
                                     adapt_delta=0.99),
                      data = list(
                                  P = P,
                                  I = I,
                                  N = nrow(time_fmp),
                                  id=ids,
                                  f_time = time_fmp,
                                  x_pred =x_pred,
                                  n_cov = 2,
                                  other_covariates = other_covs,
                                  y  = y,
                                  simulate = 0
                                  )
                    )

