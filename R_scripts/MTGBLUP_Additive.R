# Clear all objects from the current R environment to start fresh
rm(list=ls())

# Load necessary libraries
library(GenEval)
library(cgwtools)
library(gdata)
library(combinat)
library(BGLR)
library(data.table)

# Load genotype data
geno <- as.matrix(fread("path/data_25k_geno.txt"))

# Load QTL positions
QTL_df <- read.csv("path/QTL_df.csv")
ErrQTLpos_df <- read.csv("path/ErrQTLpos_df.csv")

# Combine unique QTL positions from two sources into a single vector
ErrQTLpos <- unique(unlist(ErrQTLpos_df))
Qtl.pos <- c(QTL_df$QTLpos, ErrQTLpos)

# Subset the genotype matrix to include only the QTL positions of interest
n_indv_G0 <- 25000
geno1 = geno[1:n_indv_G0,Qtl.pos]

# Create a genomic relationship matrix (GRM)
G1 <- mkGRM(geno1)

# Load phenotypic data
pheno <- read.csv("path/Pheno_G0_25K.csv")

# n_rows <-  2500
# n_cols <- ncol(pheno)
# na_matrix <- matrix(NA, n_rows, n_cols, dimnames = list(NULL, colnames(pheno)))
# y <- as.matrix(rbind(pheno[1:22500,], na_matrix))

y <- as.matrix(pheno[1:n_indv_G0,])

#y <- as.matrix(rbind(pheno[1:22500,],matrix(NA,2500,ncol(pheno))))


# Define and run a multitrait GBLUP model 
MT <- Multitrait(y=y,ETA=list(list(K=G1,model="RKHS")), nIter=25000,burnIn=10000,thin=10,verbose=TRUE)
#save( MT, file = 'path/MTG0_25K_AllTrain.RData')

# Extract Predicted Breeding Values (PBV) from the model output
PBV = as.data.frame(sapply(1:ncol(MT$ETA[[1]]$u), function(i) MT$ETA[[1]]$u[, i]))
write.csv(PBV, "path/PGV_MTGBLUP_G0_25K.csv", row.names=FALSE, quote=FALSE)


