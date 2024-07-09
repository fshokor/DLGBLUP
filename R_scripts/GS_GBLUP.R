# Clear all existing objects from the R environment
rm(list=ls())

# Load necessary libraries
library(GenEval)
library(dplyr)
library(data.table)
library(ggplot2)
library(gridExtra)
library(purrr)
library(grid)

# Set parameters for simulation
nb_G1 <- 5000            # Number of individuals in generation G1
M_SelPerc <- 0.1         # Selection percentage for males
F_SelPerc <- 1           # Selection percentage for females (100%)
pmale <- 0.2             # Proportion of males
rho <- 0.5               # Correlation coefficient between traits
h2 <- 0.3                # Heritability of the trait
Vy <- 20                 # Total phenotypic variance


# Read genotype data from a file
geno <- as.matrix(fread("path/data_25k_geno.txt"))
# Calculate allele frequencies
p <- apply(geno, 2, sum) / (2 * nrow(geno))


# Define a function to compute a selection index for individuals based on traits
compute_selection_index <- function(dataframe, rho, weights=NULL) {
  
  # Function to compute the Selection Index (SI) for each individual in a given dataframe.
  # The SI is calculated based on the weighted sum of traits in the dataframe, 
  # with an additional correlation factor.
  # Arguments:
  #   dataframe: A dataframe where each row represents an individual and each column represents a trait.
  #   rho: correlation between traits 
  #   weights: A vector of weights for each trait. If NULL, all weights are set to 1.
  
  num_traits <- ncol(dataframe)
  if (is.null(weights)) {
    # Initialize weights to 1 for each trait
    weights <- rep(1, num_traits)  # All weights are 1
  }
  
  SI <- rep(0, nrow(dataframe))
  
  # Calculate the Selection Index (SI) for each row
  for (i in 1:num_traits) {
    rho_weights_sum <- sum(weights[1:(i-1)] * rho) + sum(weights[(i):num_traits] * rho)
    SI <- SI + (weights[i] + rho_weights_sum) * dataframe[, i]
  }
  
  return(SI)
}

# Define a function to select top individuals by a percentage
select_samples_by_percentage <- function(SI, selection_percentage) {
  SI_idx <- order(SI, decreasing=TRUE)
  num_top_points <- ceiling(selection_percentage * length(SI_idx))
  top_indices <- head(SI_idx, num_top_points)
  
  return(top_indices)
}

# Define a function for different transformations based on the chosen function
cor_function <- function(x, FUN="linear", par=c(1,1)){
  switch(FUN,
         "linear" = par[1] + par[2]*x,
         "quadratic" = par[1] + par[2]*x + par[3]*x^2,
         "exp" = exp(0.4*x),
         "sin" = sin(par[1]*x),
         "logistic" = par[2]*exp(par[1]*x)/(1+exp(par[1]*x)) + (1-par[2])/2,
         stop("Unknown function")
  )
}

# Define a function to simulate other dependent traits based on the first trait's genetic values
simulate_trait <- function(g1, geno, QTL.poly, QtlEff.poly, FUN, par, rho_g, h2, Vy) {
  # Transform g1 according to a given function
  g_tmp <- cor_function(g1, FUN = FUN, par = par)
  
  # Adjust the variance of transformation according to correlation
  Mean_g <- rho_g*sqrt(h2/ var(g_tmp)) * g_tmp
  
  # Generate polygenetic effect (genetic error)
  Err_g <- geno[,QTL.poly]%*%QtlEff.poly
  
  # Adjust the linear correlation
  # if(FUN == "linear"){
  #   lambda <- ((sd(g1)/sd(Err_g))*sqrt((1-rho_g**2)/rho_g**2)) %>%  as.numeric
  #   Err_g <- lambda*Err_g
  #   Mean_g <- g_tmp
  # }
  # Sum transformed g and the genetic error & Adjust the variance to be equal to g1's
  g <- ((Mean_g+Err_g)/sd(Mean_g+Err_g))*sqrt(Vy*(h2))
  
  # Compute the error of phenotype
  e <- rnorm(length(g), 0, sqrt((1 - h2) * var(g) / h2))
  
  list(y = g + e, g = g, QtlEff_poly= QtlEff.poly, QTL_poly = QTL.poly)
}

# Simulate traits using SNP effects
Sim_traits_from_SNPeffect_fn <- function(genotype, Qtl.pos, p, SNPeffect, ErrQTLpos_df, ErrQTLEffect_df, rho_g, h2, Vy) {
  centred_genoG1 = genotype - 2 * p
  g1 <- centred_genoG1[, Qtl.pos] %*% SNPeffect 
  
  # Simulate other traits
  trait_names <- c("linear", "quadratic", "sin", "logistic", "exp")
  pheno_list <- list()
  
  for (trait in trait_names) {
    pheno_list[[trait]] <- simulate_trait(as.vector(g1), genotype, ErrQTLpos_df[[trait]],
                                          ErrQTLEffect_df[[trait]], trait, c(1, 1, 2), rho_g, h2, Vy)
  }
  
  e1 <- rnorm(length(g1), 0, sqrt((1 - h2) * var(g1) / h2))
  pheno1 <- g1 + e1 
  
  pheno_df <- data.frame(
    pheno1 = pheno1,  # Explicitly named the column
    do.call(cbind, lapply(pheno_list, function(x) x$y))
  )
  
  g_df <- data.frame(
    g1 = g1,  # Explicitly named the column
    do.call(cbind, lapply(pheno_list, function(x) x$g))
  )
  
  # Return a list containing both dataframes
  list(pheno_df = pheno_df, g_df = g_df)
}

            

NextGeneration_Geno <- function(MP, genoPHASE, lenSelecG0) {  
  # Function to generate the phase and genotype of next generation
  # Arguments:
  #   MP: A dataframe of population info
  #   genoPHASE: the phase of first generation 
  #   lenSelecG0: the length first generation 
  
  print('Starting GenoPhase generation')
  
  # Generate genoPhase for the first generation
  genoPhaseG0_1_list <- vector("list", length = length(genoPHASE))  
  for (i in 1:length(genoPHASE)) {
    sub_list_chrs <- genoPHASE[[i]]
    genoPhaseG1 = popBreed(sub_list_chrs, MP)
    genoPhaseG0_1_list[[i]] <- genoPhaseG1
  }
  rm(genoPhaseG1)
  
  # Select the indices for the next generation
  lenG0 = lenSelecG0+1
  genoPhaseG1_list <- vector("list", length = length(genoPhaseG0_1_list))
  for (i in 1:length(genoPhaseG0_1_list)) {
    # Select indices within each sub-list
    sub_list <- genoPhaseG0_1_list[[i]]
    sub_list_selected <- lapply(sub_list, function(x) x[lenG0:dim(genoPhaseG0_1_list[[1]][[1]])[1], ])
    
    # Update the sub-list in the main list
    genoPhaseG1_list[[i]] <- sub_list_selected
  }
  
  print('Starting Geno generation')
  # Combine genetic data across chromosomes
  genoG1 <- numeric(0)
  for(chr in 1:29){
    genoG1 <- cbind(genoG1,genoPhaseG1_list[[chr]][[1]]+genoPhaseG1_list[[chr]][[2]])
  };rm(chr)
  
  result <- list(genoPhase_list = genoPhaseG1_list, geno = genoG1)
  return(result)
}


# Simulate Phenotype and genetic value of next generation
NextGeneration_Pheno <- function(genoG1, Qtl.pos, p, SNPeffect, ErrQTLpos_df, ErrQTLEffect_df, rho_g, h2, Vy) {
  # Simulate BV
  print('Starting GV simulation')
  # simulate
  results <- Sim_traits_from_SNPeffect_fn(genoG1, Qtl.pos, p, SNPeffect, ErrQTLpos_df, ErrQTLEffect_df, rho_g, h2, Vy)
  
  phenotypeG1 = results$pheno_df
  
  TGV_G1 = results$g_df
  
  #ghat <- (1 + sqrt(h2)) * TGV_G1 + (1 + sqrt(h2)) * phenotypeG1
  
  result <- list(phenotype = phenotypeG1, TGV = TGV_G1)#, ghat = ghat )
  return(result)
}
                                
GenderSelection <- function(TGV_G, rho, weights = NULL, gender, F_SelPerc, M_SelPerc) {
  
  # Function to select individuals based on gender and Selection Index (SI).
  # It computes SI for each individual and selects a specified percentage of each gender.
  # Arguments:
  #   TGV_G: A dataframe representing individuals, with each column as a trait.
  #   rho: Correlation between traits.
  #   weights: A vector of weights for each trait. If NULL, all weights are set to 1.
  #   gender: A vector indicating the gender of each individual (0 for female, 1 for male).
  #   F_SelPerc: The percentage of females to be selected.
  #   M_SelPerc: The percentage of males to be selected.
  
  # Compute the Selection Index for each individual
  SI_G = compute_selection_index(TGV_G, rho, weights)
  
  # Create a dataframe with ID, Selection Index, and gender
  df_SI_sex_G <- data.frame(ID = rownames(TGV_G), SI = SI_G, sex = gender)
  
  # Separate the dataframe by gender
  F_SI_G = df_SI_sex_G[df_SI_sex_G$sex == 0,]
  M_SI_G = df_SI_sex_G[df_SI_sex_G$sex == 1,]
  
  # Select top individuals based on Selection Index for each gender
  selected_idx_F_G <- F_SI_G[select_samples_by_percentage(F_SI_G$SI, F_SelPerc),]
  selected_idx_M_G <- M_SI_G[select_samples_by_percentage(M_SI_G$SI, M_SelPerc),]
  
  # Combine the selected individuals and sort by ID
  Selected_G_df <- rbind(selected_idx_M_G, selected_idx_F_G)
  Selected_G_df$ID <- as.numeric(as.character(Selected_G_df$ID))
  Selected_G_df_sorted <- Selected_G_df[order(Selected_G_df$ID), ]
  
  return(Selected_G_df_sorted)
}


PopInfoSim <- function (n, n.gen, p.male, pop.info = NULL,
                        sex = NULL, ID = NULL,  Selected_idx = NULL){
  # Function to simulate population information across generations.
  # Arguments:
  #   n: A vector indicating the number of individuals in each generation.
  #   n.gen: The number of generations to simulate.
  #   p.male: The probability of an individual being male.
  #   pop.info: An optional dataframe containing existing population information.
  #   sex: An optional vector indicating the sex of individuals.
  #   ID: An optional vector of individual IDs.
  #   Selected_idx: Optional, contains row indices of individuals that can be considered as parents.
  
  # Helper function to zero-pad IDs.
  char0 <- function(n) {
    return(paste(rep(0, 6 - n), collapse = ""))
  }
  
  # Initial population setup if pop.info is not provided.
  if (is.null(pop.info)) {
    if (is.null(ID)){
      ID = 1:n[1]}
    if (is.null(sex)){
      sex = rep(0, n[1])
      sex[sample(n[1], round(p.male * n[1]))] <- 1}
    pop.info <- data.frame(ID = ID, parent1 = rep(NA,n[1]), 
                           parent2 = rep(NA, n[1]), sex = sex, 
                           generation = rep(0, n[1]))
    
    # Formatting the IDs for the initial population.
    nID <- matrix(nchar(as.character(pop.info$ID[pop.info$generation == 0])), ncol = 1)
    pop.info$ID <- paste0("", "G", paste(rep(0, 2), collapse = ""), "0", 
                          c("M", "F")[2 - pop.info$sex[pop.info$generation == 0]], 
                          apply(nID, 1, char0), pop.info$ID[pop.info$generation == 0])
  }
  
  # Generating additional generations if required.
  if (n.gen > 0) {
    for (g in 1:n.gen) {
      ng = 1 + max(pop.info$generation)
      
      # Creating a temporary dataframe for the new generation.
      tmp <- data.frame(ID = 1:n[g + 1], 
                        parent1 = rep(NA, n[g + 1]), 
                        parent2 = rep(NA, n[g + 1]), 
                        sex = rep(0, n[g + 1]), 
                        generation = rep(ng, n[g + 1]))
      
      # Assigning sex to the new generation.
      tmp$sex[sample(n[g + 1], round(p.male * n[g + 1]))] <- 1
      
      # Formatting the IDs for the new generation.
      nID <- matrix(nchar(as.character(tmp$ID[tmp$generation == 
                                                ng])), ncol = 1)
      tmp$ID <- paste0("", "G", 
                       paste(rep(0, 3 - nchar(as.character(ng))), collapse = ""), 
                       ng, c("M", "F")[2 - tmp$sex[tmp$generation == ng]], 
                       apply(nID, 1, char0), tmp$ID[tmp$generation == ng])
      
      # Check if Selected_ID is provided and not NULL.
      if (!is.null(Selected_idx)) {
        # Use the row indices in Selected_ID to filter potential parents.
        potential_parents <- pop.info[Selected_idx, ]
      } else {
        # If Selected_ID is NULL, consider all individuals in pop.info as potential parents.
        potential_parents <- pop.info
      }
      
      # Assigning parents to the new generation from the filtered potential parents.
      tmp$parent1 <- sample(potential_parents$ID[potential_parents$sex == 1 & potential_parents$generation >= ng - 4], 
                            n[g + 1], replace = TRUE, 
                            prob = (potential_parents$generation[potential_parents$sex == 
                                                                   1 & potential_parents$generation >= ng - 4] + 1)/sum(potential_parents$generation[potential_parents$sex == 
                                                                                                                                                       1 & potential_parents$generation >= ng - 4] + 1))
      tmp$parent2 <- sample(potential_parents$ID[potential_parents$sex == 0 & potential_parents$generation >= ng - 2], 
                            n[g + 1], replace = TRUE, 
                            prob = (potential_parents$generation[potential_parents$sex == 
                                                                   0 & potential_parents$generation >= ng - 2] + 1)/sum(potential_parents$generation[potential_parents$sex == 
                                                                                                                                                       0 & potential_parents$generation >= ng - 2] + 1))
      # Combining the new generation with the existing population.
      pop.info <- rbind(pop.info, tmp)
      rm(tmp)
    }
  }
  # Returning the complete population information.
  return(pop.info)
}

                                
# Select pased genetype for seleted indices                               
SelectPhase <- function(selected_idx, genoPHASE){
  genoPHASEG0 <- vector("list", length = length(genoPHASE)) 
  for (i in 1:length(genoPHASE)) {
    # Select indices within each sub-list
    sub_list <- genoPHASE[[i]]
    sub_list_selected <- lapply(sub_list, function(x) x[selected_idx, ])
    
    # Update the sub-list in the main list
    genoPHASEG0[[i]] <- sub_list_selected
  }
  return(genoPHASEG0)
}

generateMP <- function(pop.info.G0, nb_G1, p.male, gen_nb, Selected_idx=NULL) {
  # Function to generate mating plan
  # Arguments:
  #   pop.info.G0: A dataframe containing the initial population information. 
  #                This is used as a basis for generating the new population.
  #   nb_G1: The number of individuals in the first generation.
  #   p.male: The probability of an individual being male. Used in sex assignment.
  #   gen_nb: The generation number.
  
  if (gen_nb == 0) {
    # Generates a new population info for generation 0 using the PopInfoSim function.
    # It uses the ID and sex from pop.info.G0, and the number of individuals in generation 1 (nb_G1).
    MP <- PopInfoSim(c(dim(pop.info.G0)[1], nb_G1), n.gen = 1, p.male = p.male, 
                     ID = pop.info.G0$ID, sex = pop.info.G0$sex, Selected_idx = Selected_idx)
    new_MP <- NULL
  } else {
    # For generations other than 0, generates a new population info including the first generation.
    # modify an existing population.
    
    MP <- PopInfoSim(c(dim(pop.info.G0)[1], nb_G1), n.gen = 1, p.male = p.male,
                     pop.info = pop.info.G0, Selected_idx = Selected_idx)
    # Modifies MP to update parent information and generation numbers for the new generation.
    new_MP <- MP %>%
      mutate(
        parent1 = if_else(row_number() <= dim(pop.info.G0)[1], NA_character_, parent1),
        parent2 = if_else(row_number() <= dim(pop.info.G0)[1], NA_character_, parent2),
        generation = if_else(row_number() <= dim(pop.info.G0)[1], 0, 1)  
      )
  }
  # MP_gen <- MP[MP$generation != 0, ]
  # 
  # pop.info <- data.frame(ID = ID, parent1 = rep(NA,n[1]), 
  #                        parent2 = rep(NA, n[1]), sex = sex, 
  #                        generation = rep(0, n[1]))
  # 
  # # Formatting the IDs for the initial population.
  # nID <- matrix(nchar(as.character(pop.info$ID[pop.info$generation == 0])), ncol = 1)
  # pop.info$ID <- paste0("", "G", paste(rep(0, 2), collapse = ""), "0", 
  #                       c("M", "F")[2 - pop.info$sex[pop.info$generation == 0]], 
  #                       apply(nID, 1, char0), pop.info$ID[pop.info$generation == 0])
  # 
  # All_MP <- rbind(pop.info.G0, MP_gen)
  # # Extract numeric part of ID for indexing.
  # MP$idx <- as.numeric(str_extract(MP$ID, "(?<=[FM])\\d+"))
  return(list(MP = MP, new_MP = new_MP))
}

plot_Traits <- function(...) {
  dataframes <- list(...)
  n_dataframes <- length(dataframes)
  plots <- list()
  
  for (i in 2:ncol(dataframes[[1]])) {
    p <- ggplot()
    for (j in 1:n_dataframes) {
      p <- p + geom_point(data = dataframes[[j]], aes_string(x = names(dataframes[[j]])[1], y = names(dataframes[[j]])[i]), color = scales::hue_pal()(n_dataframes)[j])
    }
    p <- p + xlab("Trait 1") + ylab(paste("Trait", i))
    plots[[i-1]] <- p
  }
  
  do.call(grid.arrange, c(plots, ncol = length(plots)))
  
  # Save the plot
  # ggsave(file_name, plot_object, height = 4, width = 15)  # You can adjust width as needed
}

merge_2Phase <- function(Selected_PhaseG0, Selected_PhaseG1) {
  genoPhaseG01_list <- vector("list", length = length(Selected_PhaseG1)) 
  for (i in 1:length(Selected_PhaseG1)) {
    sub_list <- Map(rbind, Selected_PhaseG0[[i]], Selected_PhaseG1[[i]])
    genoPhaseG01_list[[i]] <- sub_list
  }
  return(genoPhaseG01_list)}

MergePhases <- function(...) {
  phases <- list(...)
  mergedPhases <- Reduce(function(x, y) merge_2Phase(x, y), phases)
  return(mergedPhases)
}

GS_NextGeneration <- function(PBV, rho, weights = NULL, G_info, Phase, F_SelPerc, 
                              M_SelPerc, nb_G1, p.male, gen_nb, Qtl.pos, p, SNPeffect, 
                              ErrQTLpos_df, ErrQTLEffect_df, h2, Vy, out_path){
  print('Starting Genomic Selection for G0')
  # Select individuals based on gender and PBV
  Selected_G = GenderSelection(TGV_G=PBV, rho=rho, gender=G_info$sex,
                               F_SelPerc=F_SelPerc, M_SelPerc=M_SelPerc)
  # Save selected individuals' IDs and data
  # write.csv(Selected_G$ID, paste0(out_path, "/selected_indices_G", gen_nb, "_PBVselection.csv"), row.names=FALSE, quote=FALSE)
  write.csv(Selected_G, paste0(out_path, "/Selected_G", gen_nb, ".csv"), row.names=FALSE, quote=FALSE)
  # 
  print('Starting Generating Mating Plan')
  # Generate mating plan for the next generation
  MP = generateMP(pop.info.G=G_info, nb_G=nb_G1, p.male=p.male, gen_nb=gen_nb,
                  Selected_idx = Selected_G$ID)
  write.csv(MP[["MP"]], paste0(out_path, "/MP", gen_nb + 1, ".csv"), row.names=FALSE, quote=FALSE)
  
  # Selected_Phase = SelectPhase(Selected_G$ID, Phase)
  
  # Generate genetic information for the next generation
  if (gen_nb == 0) {
    MP1 <- MP[["MP"]]
  } else {
    MP1 <- MP[["new_MP"]]
  }
  resultsGNext = NextGeneration_Geno(MP1, Phase, dim(G_info)[1])
  
  resultsG_Pheno =  NextGeneration_Pheno(resultsGNext[["geno"]], Qtl.pos, p, SNPeffect, ErrQTLpos_df, ErrQTLEffect_df, rho, h2, Vy)
  write.csv(resultsG_Pheno$TGV, paste0(out_path, "/TGV_G", gen_nb + 1, ".csv"), row.names=FALSE, quote=FALSE)
  write.csv(resultsG_Pheno$phenotype, paste0(out_path, "/Phenotype_G", gen_nb + 1, ".csv"), row.names=FALSE, quote=FALSE)
  # Extract and save specific genetic phase information
  print('Starting saving Phase Data')
  GenophaseGNext = resultsGNext[["genoPhase_list"]]
  # Male_PhaseGNext = SelectPhase(which(MP[MP$generation == gen_nb + 1,]$sex == 1), GenophaseGNext)
  # save(Male_PhaseGNext, file=paste0(out_path, "/Male_PhaseG", gen_nb + 1, ".Rdata"))
  save(GenophaseGNext, file=paste0(out_path, "/GenophaseG", gen_nb + 1, ".Rdata"))
  # 
  # Save genetic information for the next generation
  print('Starting saving Geno Data')
  write.table(resultsGNext[["geno"]], file=paste0(out_path, "/genoG", gen_nb + 1, ".txt"), row.names=FALSE, col.names=FALSE)

  result <- list(Selected_G = Selected_G, resultsGNext = resultsGNext, MP = MP, resultsG_Pheno = resultsG_Pheno)
  return(result)
  
}

# G1
TGV_G0 = read.csv("path/TGV_G0_25K.csv")
PBV_GBLUP_G0 = read.csv("path/GS/PBVGBLUP/PBV_GBLUPDL_G0_df.csv")
loaded_names <- load('path/genoPHASE_25k.Rdata') 
assign('GenoPhase0', get(loaded_names[1]))  

QTL_df <- read.csv("path/QTL_df.csv")
ErrQTLpos_df <- read.csv("path/ErrQTLpos_df.csv")
ErrQTLEffect_df <- read.csv("path/ErrQTLEffect_df.csv")
G0_info = read.csv("path/G0_info.csv")

G1results <- GS_NextGeneration(PBV=PBV_GBLUP_G0, rho=rho, G_info=G0_info, Phase=GenoPhase0, 
                               F_SelPerc=F_SelPerc, M_SelPerc=M_SelPerc, nb_G1=nb_G1, 
                               p.male=pmale, gen_nb=0, Qtl.pos=QTL_df$QTLpos, p=p, 
                               SNPeffect=QTL_df$QTLeffect, ErrQTLpos_df=ErrQTLpos_df, 
                               ErrQTLEffect_df=ErrQTLEffect_df, h2=h2, Vy=Vy, 
                               out_path="path/GS/PBVGBLUP/")
TGV_G1 = G1results$resultsG_Pheno$TGV
colMeans(TGV_G1) - colMeans(TGV_G0)
plot_Traits(TGV_G1)

plot_Traits(TGV_G0, TGV_G1)

compareGeneticGain <- function(df_list) {
  library(stats)
  
  # Extract the first dataframe as df2
  df2 <- df_list[[1]]
  
  # Create a list of new column names based on the number of columns
  new_col_names <- paste0('Trait_', seq_len(ncol(df2)))
  
  # Rename columns in all dataframes in the list
  df_list <- lapply(df_list, function(df) {
    colnames(df) <- new_col_names
    return(df)
  })
  
  # Calculate column means for df2
  colmeans_df2 <- colMeans(df2)
  
  # Initialize empty lists to store results
  colmean_diff_list <- list()
  t_statistic_list <- list()
  p_value_list <- list()
  
  # Loop through the other dataframes and compare them to df2
  for (df1 in df_list[-1]) {
    # Calculate column means for df1
    colmeans_df1 <- colMeans(df1)
    
    # Compute the difference between column means
    colmean_diff <- colmeans_df1 - colmeans_df2
    
    # Perform paired t-test
    test_result <- t.test(as.matrix(df1), as.matrix(df2))
    
    # Append results to lists
    colmean_diff_list <- c(colmean_diff_list, list(colmean_diff))
    t_statistic_list <- c(t_statistic_list, list(test_result$statistic))
    p_value_list <- c(p_value_list, list(test_result$p.value))
  }
  
  # Create a dataframe to store the results
  colmean_diff_df <- do.call(rbind, colmean_diff_list)
  t_statistic_df <- do.call(rbind, t_statistic_list)
  p_value_df <- do.call(rbind, p_value_list)
  
  # Change the row names to 'Generation 1', 'Generation 2', ...
  row_names <- paste0('Generation ', seq_len(length(colmean_diff_list)))
  rownames(colmean_diff_df) <- row_names
  rownames(t_statistic_df) <- row_names
  rownames(p_value_df) <- row_names
  
  return(list(colmean_diff_df, t_statistic_df, p_value_df))
}

TGVs_df <- list(TGV_G0, TGV_G1)
GG = compareGeneticGain(TGVs_df)
View(GG[[1]])


