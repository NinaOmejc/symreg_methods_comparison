require("GPoM")  # check if the method is installed and install if not 

###############################################################################
# FUNCTIONS
###############################################################################


mse <- function(x, y){
  means = colMeans(y)
  sds = apply(y,2,sd)
  xnorm = scale(x, center = means, scale = sds)
  ynorm = scale(y, center = means, scale = sds)
  return(mean((xnorm-ynorm)^2))
}

smape <- function(x,y){
  return(2/length(x) * sum(abs(x-y)/(abs(x)+abs(y))))
}


fit_gpomo <- function(time, ladata, observed_var, max_time_derivative, poly_degree, steps, plots_filename, verbose = TRUE,
                      write_results_folder=write_results_folder, out_filename_base = out_filename_base){
  # output: coefficients of best gpomo model as the minimum smape metric in train prediction series.
  n_vars = length(observed_var)
  
  out <- gPoMo(data = ladata,
               tin = time,
               dMax = poly_degree, 
               nS = rep(max_time_derivative, n_vars),
               show = 0,
               IstepMin = 10, 
               IstepMax = steps,
               nPmin = 0, 
               nPmax = 20,
               method ='lsoda',
               verbose = TRUE)
  
  # plot folder
  #if (verbose){
  #  plots_folder = paste0(plots_filename)
  #  dir.create(plots_folder, showWarnings = FALSE)
  #}
  
  #visuEq(out$models$model5, n_vars, poly_degree, approx = 1, substit = c('x','y'))
  
                                                                                                              
  #visuOutGP(out)
  # calculate mse for all the tested models
  loss <- list()
  for(model_name in names(out$stockoutreg)){
    model_data = out$stockoutreg[[model_name]][,seq(2,n_vars+1,1)] #keep the column corresponding to the observed variables
    data_posta = ladata
    
    tryCatch({
      coeffs <- out$models[[model_name]]
      colnames(coeffs) <- rev(poLabs(nVar=max_time_derivative*n_vars, dMax = 1, Xnote=toupper(observed_var))[-1])
      rownames(coeffs) <- poLabs(nVar=max_time_derivative*n_vars, dMax = poly_degree, Xnote=toupper(observed_var))
      
      out_filename3 <- paste0(write_results_folder, '/params_', out_filename_base, '_', model_name, '.csv')
      dir.create(write_results_folder)
      write.csv(coeffs, paste(out_filename3, sep='/'))
    },
    error = function(e){
      print("Error occured")
      write.csv("Error", paste(out_filename3, sep='/'))
    })
    
    # TODO: filtrar con el tiempo y no quedarse con los priemros como ahora
    if (any(is.na(model_data))){
      loss[[model_name]] <- Inf
    }else{
      if  (length(model_data)<length(data_posta)){
        data_posta_loss = data_posta[1:length(model_data)]
        model_data_loss = model_data
      }else{
        model_data_loss = model_data[1:length(time)]
        data_posta_loss = data_posta
      }
      
      #colnames(model_data) = colnames(data_posta)
      loss[[model_name]] <- smape(model_data_loss, data_posta_loss)
    }
    
    # ploting
    #if (verbose){
    #  png(paste0(plots_folder,'/',model_name, '.png'))
    #  plot(data_posta,col = 'blue', main = paste(model_name,'\n', loss[[model_name]]), type='l')
    #  lines(model_data, col = 'red')
    #  dev.off()
    #}
  }
  

  best_model = names(loss)[which.min(loss)]
  #visuEq(out$models[[best_model]], nVar=max_time_derivative, dMax = poly_degree, approx = 2)
  coefs <- out$models[[best_model]]
  colnames(coefs) <- rev(poLabs(nVar=max_time_derivative*n_vars, dMax = 1, Xnote=toupper(observed_var))[-1])
  rownames(coefs) <- poLabs(nVar=max_time_derivative*n_vars, dMax = poly_degree, Xnote=toupper(observed_var))

  return(coefs)
  
}


experiment <- function(path_data, fname_data, write_results_folder, observed_var, keep2test, steps_list, 
                       max_time_derivatives, poly_degrees, verbose, sys_name, data_size, snr, init){
  
  data_raw <- read.csv(paste0(path_data, fname_data))
  n <- dim(data_raw)[1]
  time <- data_raw[1:(n-keep2test),1]
  data <- data_raw[1:(n-keep2test),observed_var]
  # print(data)
  # fit data and time it
  for(steps in steps_list){
    for (max_time_derivative in max_time_derivatives){
      for (poly_degree in poly_degrees){
        
        print(paste('Doing steps:', steps, 'max time derivative', max_time_derivative, 'poly degree', poly_degree))
        final_directory = paste0('gpom_', sys_name, '_', data_size, '_dmax', max_time_derivative, '_poly', poly_degree,'_steps', steps, '_obs', paste(observed_var, collapse=''), '_snr', snr, '_init', iinit)
        write_results_folder_final <- paste0(write_results_folder, final_directory, sep = "/")
        out_filename_base <- paste0('gpom_', sys_name, '_', data_size, '_dmax', max_time_derivative, '_poly', poly_degree,'_steps', steps, '_obs', paste(observed_var, collapse=''), '_snr', snr, '_init', iinit)
        out_filename1 <- paste0(write_results_folder_final, 'params_', out_filename_base, '_best.csv')
        out_filename2 <- paste0(write_results_folder_final, 'duration_', out_filename_base, '.csv')
        plots_filename <- paste0(write_results_folder_final,'plots_',out_filename_base, '.png')
        
        t0 <- Sys.time()
        tryCatch({  
          coeffs <- fit_gpomo(time=time, 
                              ladata=data,
                              observed_var = observed_var,
                              max_time_derivative=max_time_derivative, 
                              poly_degree=poly_degree, 
                              steps=steps,
                              plots_filename = plots_filename,
                              verbose = verbose,
                              write_results_folder=write_results_folder_final, 
                              out_filename_base = out_filename_base)
          
          duration <- round(as.numeric(difftime(time1 = Sys.time(), time2 = t0, units = "secs")), 3)
          write.csv(coeffs, paste(out_filename1, sep='/'))
          write.csv(duration, paste(out_filename2, sep='/'))
        },
        error = function(e){
          print("Error occured")
          write.csv("Error", paste(out_filename1, sep='/'))
          write.csv("Error", paste(out_filename2, sep='/'))
          
        }
        )
      }
    }
  }
}

###############################################################################
# SET AND RUN GPOM
###############################################################################

exp_version = 'e1' # e1: constrained search space (we did not do e2 version (unconstrained space) for gpom)
exp_type = "sysident_partial"
system_name = "vdp"
data_version = "train"
snrs = c('None', '30', '13')
inits = c(0:3)
obss = list(c('x', 'y'), c('x'),c('y'))
targets = c(1, 2, 3)
maxpolys = 3
steps = 5120 
targets_to_plot = c(1,2)
data_sizes = c("small", "large")

main_path = "D:/Experiments/symreg_methods_comparison" # adjust accordingly

for (data_size in data_sizes){
  
  if (data_size == "large"){
    data_length = 2000 # in timepoints
    test_size = 200 
    time_end = 20 # in seconds
    time_step = 0.01
  }else{
    data_length = 100 
    test_size = 10 
    time_end = 10
    time_step = 0.1
  }
  
  for(isnr in snrs){
    for (iinit in inits){
      for (iobs in obss){
        print(paste("Doing: size: ", data_size, "| snr: ", isnr, " | init: ",  iinit, " | obs: ", iobs))
        
        if (length(iobs) == 1){
          max_time_derivative = c(2)
        } else{
          max_time_derivative = c(1)
        }
        experiment(
          path_data = paste0(main_path, "/data/", data_version, "/", data_size, "_for_lodefind_gpom/", system_name, "/"),
          fname_data = paste0("data_", system_name, "_len", as.character(time_end), "_rate", gsub("\\.", "", as.character(time_step)), "_snr", isnr, "_init", iinit, ".csv"),
          write_results_folder = paste(main_path, "/results/", exp_type, "/gpom/", exp_version, "/", system_name, "/", sep=""),
          observed_var = iobs,
          keep2test = test_size,
          steps_list = steps,
          max_time_derivatives = max_time_derivative, 
          poly_degrees = c(3),
          verbose = TRUE,
          sys_name = system_name,
          data_size = data_size,
          snr = isnr,
          init = iinit
        )
        
        print(paste("Finished size: ", data_size, "| snr ", isnr, " | init: ",  iinit, " | obs: ", iobs))
      }
    }
  }
}
