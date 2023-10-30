require("GPoM")

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
      
      out_filename3 <- paste0(write_results_folder, 'params_', out_filename_base, '_', model_name, '.csv')
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


experiment <- function(filename, write_results_folder, observed_var, keep2test, steps_list, max_time_derivatives, poly_degrees, verbose, sys_name, snr, init){
  
  data_raw <- read.csv(filename)
  n <- dim(data_raw)[1]
  time <- data_raw[1:(n-keep2test),1]
  data <- data_raw[1:(n-keep2test),observed_var]
  print(data)
  #fit data and time it
  for(steps in steps_list){
    for (max_time_derivative in max_time_derivatives){
      for (poly_degree in poly_degrees){
        
        print(paste('Doing steps:', steps, 'max time derivative', max_time_derivative, 'poly degree', poly_degree))
        
        out_filename_base <- paste0('gpomo_myvdp_dmax', max_time_derivative, '_poly', poly_degree,'_steps', steps, '_obs', paste(observed_var, collapse=''), '_snr', snr, '_init', iinit)
        out_filename1 <- paste0(write_results_folder, 'params_', out_filename_base, '_best.csv')
        out_filename2 <- paste0(write_results_folder, 'duration_', out_filename_base, '.csv')
        plots_filename <- paste0(write_results_folder,'plots_',out_filename_base, '.png')
        
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
                              write_results_folder=write_results_folder, 
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

exp_version = 'e6'
system = 'myvdp'
data_version = "allonger"
data_length = 2000
test_length = 1
snrs = c('inf', 30, 13)
inits = c(0:3)
obss = list(c('x', 'y'), c('x'),c('y'))
# obss = list(c('x', 'y'))
targets = c(1, 2, 3)
maxpolys = 3
steps = 5120 
targets_to_plot = c(1,2)


#isnr = 'inf'
#iinit = 0
#iobs = c('x', 'y')


for(isnr in snrs){
  for (iinit in inits){
    for (iobs in obss){
      print(paste("Doing: snr ", isnr, " | init: ",  iinit, " | obs: ", iobs))
      
      if (length(iobs) == 1){
        max_time_derivative = c(2)
      } else{
        max_time_derivative = c(1)
      }
      experiment(
        filename = paste("D:/Experiments/MLJ23/data/lodefind/data_lodefind_", system, "_", data_version, "_len", data_length, "_snr", isnr, "_init", iinit, ".csv", sep=""),
        write_results_folder = paste("D:/Experiments/MLJ23/results/gpomo/", exp_version, "/", system, "/", sep=""),
        observed_var = iobs,
        keep2test = test_length,
        steps_list = steps,
        max_time_derivatives = max_time_derivative, 
        poly_degrees = c(3),
        verbose = TRUE,
        sys_name = system,
        snr = isnr,
        init = iinit
      )
      
      print(paste("Finished: snr ", isnr, " | init: ",  iinit, " | obs: ", iobs))
    }
  }
}
