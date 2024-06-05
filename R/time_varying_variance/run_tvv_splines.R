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
P = 3 # no of basis functions
P_Si = 2
data_dir <- "U:/Documents/repos/joint_longitudinal_models/R/one_predictor/joint_model/"
model_dir <- "U:/Documents/repos/jelo/R/time_varying_variance/"
results_dir <- "G:/irena/tvv/"

model_data <- read.csv(paste0(data_dir, "model_data_01062023.csv"))

## total dataset 
b_spline<- read.csv(paste0(data_dir, "b_spline_05102024.csv"))
S_b_spline<- read.csv(paste0(data_dir, "S_b_spline_05102024.csv"))

ids <- model_data$new_id
I <- length(unique(ids))

other_covs <- cbind(model_data$bmi_std, model_data$age_std) 

x_pred <- model_data$lag_e2
y <- model_data$bmd_resid

compiled_model <- stan_model(paste0(model_dir, "joint_tvv_cov.stan"))

 ## sample from the model:
bmd_out <- sampling(compiled_model,
                      sample_file=paste0(results_dir, Sys.Date(), '_e2_tvv_model_samples_splines.csv'), #writes the samples to CSV file
                      iter = 4000,
                      warmup=2000, #BURN IN
                    save_warmup=FALSE,
                      chains =4,
                      seed = seed,
                      control = list(max_treedepth = 60,
                                     adapt_delta=0.99),
                      data = list(
                                  P = P,
                                  P_Si=P_Si,
                                  I = I,
                                  N = nrow(model_data),
                                  id=ids,
                                  f_time = b_spline,
                                  s_time = S_b_spline,
                                  time_fmp = model_data$time_fmp,
                                  x_pred =x_pred,
                                  n_cov = 2,
                                  other_covariates = other_covs,
                                  y  = y,
                                  simulate = 0
                                  )
                    )
# 
# bmd_out <-  "G:/irena/tvv/"
# 
# sample_file_names <- c("tvv_model_samples_2")
# 
# bmd_out <- read_stan_csv(paste0(results_dir, sample_file_names,".csv"))
# 
# rstan::traceplot(bmd_out, pars=c("S_mu","S_Sigma"))
