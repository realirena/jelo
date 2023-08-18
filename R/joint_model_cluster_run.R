## cluster file 

rm(list=ls())
library(rstan)
library(dplyr)
library(MASS)
options(mc.cores = parallel::detectCores(logical= FALSE))
rstan_options(auto_write = TRUE)
slurm_arrayid <- Sys.getenv('SLURM_ARRAY_TASK_ID')

taskid <- as.numeric(slurm_arrayid)
seed = taskid + 128
#setwd("/home/irena/")
## set up the data simulation parameters:
P = 2 # no of basis functions

model_data <- read.csv("model_data_05222023.csv")

ids <- model_data$new_id
I <- length(unique(ids))
time_fmp <- cbind(1, model_data$time_fmp)

other_covs <- cbind(model_data$bmi_std, model_data$age_std) 

x_pred <- model_data$lag_e2
y <- model_data$bmd_resid
 
compiled_model <- stan_model("joint_model_w_cov.stan")

 ## sample from the model:
sim_out <- sampling(compiled_model,
                      sample_file=paste0('0629_model_samples.csv'), #writes the samples to CSV file
                      iter = 2000,
                      warmup=1000, #BURN IN
                      chains =4,
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

