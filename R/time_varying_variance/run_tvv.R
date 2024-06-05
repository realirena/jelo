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
P_Si = 2
data_dir <- "U:/Documents/repos/joint_longitudinal_models/R/one_predictor/joint_model/"
model_dir <- "U:/Documents/repos/jelo/R/time_varying_variance/"
results_dir <- "G:/irena/tvv/"

model_data <- read.csv(paste0(data_dir, "model_data_01062023.csv"))

ids <- model_data$new_id
I <- length(unique(ids))
time_fmp <- cbind(1, model_data$time_fmp)

other_covs <- cbind(model_data$bmi_std, model_data$age_std) 

x_pred <- model_data$lag_e2
y <- model_data$bmd_resid


 
compiled_model <- stan_model(paste0(model_dir, "joint_model_varying_variance.stan"))

 ## sample from the model:
bmd_out <- sampling(compiled_model,
                      sample_file=paste0(results_dir, '_model_samples.csv'), #writes the samples to CSV file
                      iter = 2000,
                      warmup=1000, #BURN IN
                      chains =4,
                      seed = seed,
                      control = list(max_treedepth = 60,
                                     adapt_delta=0.99),
                      data = list(
                                  P = P,
                                  P_Si=P_Si,
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



rstan::traceplot(bmd_out, pars=c("S_mu","S_Sigma"))
