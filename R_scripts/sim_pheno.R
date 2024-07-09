# Clear the environment of previously stored objects
rm(list=ls())

# Load necessary libraries for data manipulation and plotting
library(GenEval)
library(data.table)
library(dplyr)
library(ggplot2)
library(gridExtra)


# Define a function to apply linear and non linear transformations based on the specified type 
cor_function <- function(x, FUN="linear", par=c(1,1)){
  switch(FUN,
         "linear" = par[1] + par[2]*x,  # y = a + bx (linear function)
         "quadratic" = par[1] + par[2]*x + par[3]*x^2,  # y = a + bx + cx^2 (quadratic function)
         "exp" = exp(0.4*x),  # y = exp(0.4x) (exponential function)
         "sin" = sin(par[1]*x),  # y = sin(ax) (sinusoidal function)
         "logistic" = par[2]*exp(par[1]*x)/(1+exp(par[1]*x)) + (1-par[2])/2,  # logistic growth model
         stop("Unknown function")  # error handling for unknown functions
  )
}

# Load genotype data 
geno <- as.matrix(fread("path/data_25k_geno.txt"))

# Define genetic parameters
rho_g <- 0.5 # Genetic correlation between traits
h2 <- 0.6 # Heritability of the trait
Vy <- 20 # Total phenotypic variance

# Define positions for QTL influencing the trait
Qtl.pos <- c(seq(750, 1102, by=2), seq(1830,2040, by =2), seq(5950,6102, by=2), seq(7700,7902, by=2))
nbQTL.poly <- 50
QTL.poly <- setdiff(1:dim(geno)[2], Qtl.pos) %>% sample(.,size = nbQTL.poly)
allQtl.pos <- sort(c(Qtl.pos, QTL.poly))

# Simulate phenotypic values using the genotype and QTL positions
pheno1 <- simPheno(M=list(geno[,allQtl.pos]),h2=h2,Vy=Vy)

# Extract the genetic values (g1)
g1 <- pheno1$g[[1]]


# Create data frame with QTL positions and effects, and save it
QTL_df <- data.frame(QTLpos = allQtl.pos, QTLeffect = pheno1$a[[1]])
write.csv(QTL_df, file = "path/QTL_df.csv", row.names = FALSE, quote = FALSE)
#QTL_df <- read.csv("path/QTL_df.csv")


# Define a function to simulate other dependent traits based on the first trait's genetic values
simulate_trait <- function(g1, geno, Qtl.pos, nbQTL.poly, FUN, par, rho_g, h2, Vy) {
  # Transform g1 according to a given function 
  g_tmp <- cor_function(g1, FUN = FUN, par = par) 
  
  # Adjust the variance of transformation according to correlation
  Mean_g <- rho_g*sqrt(h2/ var(g_tmp)) * g_tmp
  
  # Generate polygenetic effect (genetic error)
  QTL.poly <- setdiff(1:dim(geno)[2], Qtl.pos) %>% sample(.,size = nbQTL.poly)
  QtlEff.poly <- runif(n = nbQTL.poly, min = -0.2, max = 0.2)
  Err_g <- geno[,QTL.poly]%*%QtlEff.poly
  
  # # Adjust the linear correlation 
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



# Simulate dependent traits
trait_names <- c("linear", "quadratic", "sin", "logistic", "exp")
pheno_list <- list()
for (trait in trait_names) {
  pheno_list[[trait]] <- simulate_trait(g1, geno, allQtl.pos, nbQTL.poly, 
                                        trait, c(1, 1, 2), rho_g, h2, Vy)
}

# Convert lists to data frames
pheno_df <- data.frame(
  pheno1$y, # Add pheno1$y
  do.call(cbind, lapply(pheno_list, function(x) x$y))

)

g_df <- data.frame(
  g1,  # Add g1
  do.call(cbind, lapply(pheno_list, function(x) x$g))
  
)

ErrQTLEffect_df <- data.frame(lapply(pheno_list, function(x) x$QtlEff_poly))
ErrQTLpos_df <- data.frame(lapply(pheno_list, function(x) x$QTL_poly))

# Save the data frames to CSV files
write.csv(ErrQTLEffect_df, "path/ErrQTLEffect_df.csv", row.names = FALSE, quote = FALSE)

write.csv(ErrQTLpos_df, "path/ErrQTLpos_df.csv", row.names = FALSE, quote = FALSE)

write.csv(pheno_df, "path/Pheno_G0_25K.csv", row.names = FALSE, quote = FALSE)

write.csv(g_df, "path/TGV_G0_25K.csv", row.names = FALSE, quote = FALSE)

# Define a function to plot trait relationships 
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

# Call the function
plot_Traits(g_df)
cor(g_df)

