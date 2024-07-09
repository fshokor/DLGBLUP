# Clear the environment of previously stored objects
rm(list=ls())


# Load the GenEval library for genetic data simulation
library(GenEval)

# Define the number of markers and individuals
m <- 10000  # Total number of markers
n <- 25000  # Total number of individuals

# Define the proportion of markers on each of the 29 chromosomes
p <- c(0.063,0.053,0.048,0.048,0.040,0.050,0.043,0.046,0.040,0.043,0.042,0.034,0.034,0.035,0.032,0.033,0.032,0.026,0.027,0.030,0.027,0.024,0.022,0.025,0.020,0.022,0.019,0.020,0.022)

# Initialize a list to store genotype data for each chromosome (phased)
genoPHASE <- list()
# Initialize a matrix to store total genotype data (non-phased)
geno <- numeric(0)

# Loop over each chromosome to simulate genotypes
for(chr in 1:29){
  # simGeno function simulates genotype data; LD=TRUE introduces linkage disequilibrium
  genoPHASE[[chr]] <- simGeno(n, AlleleFreq=runif(round(m*p[chr]), 0.05, 0.5), LD=TRUE, phased=TRUE)
  # Combine the two phased genotypes to create a total genotype and append to the geno matrix
  geno <- cbind(geno, genoPHASE[[chr]][[1]] + genoPHASE[[chr]][[2]])
}

# Save the phased genotype data to an R data file
save(genoPHASE, file="path/data_25k_genoPHASE.Rdata")

# Write the total genotype matrix to a text file without row or column names
write.table(geno, file="path/data_25k_geno.txt", row.names=FALSE, col.names=FALSE)
