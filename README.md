# Deep Learning and GBLUP Integration: An Approach that Identifies Nonlinear Genetic Relationships Between Traits

## Summary
DLGBLUP is a hybrid model combining deep learning and GBLUP (genomic best linear unbiased prediction) to predict multiple correlated traits' genetic value (GV). It consists of two steps: first, the GVs are predicted using a multi-trait GBLUP based on the genomic data, then GVs are re-predicted using a DL model that captures potential non-linear genetic correlations between traits. [[Link](https://www.biorxiv.org/content/10.1101/2024.03.23.585208v1)]

![Demo Video](https://github.com/fshokor/DLGBLUP/blob/main/DLGBLUP_Illustration.gif)

## Requirements 
### R
- GenEval
- data.table
- dplyr
- ggplot2
- gridExtra
- cgwtools
- gdata
- combinat
- BGLR
- purrr

To install the required packages, run the following command
```shell
install.packages(c("GenEval", "data.table", "dplyr", "ggplot2", "gridExtra", "cgwtools", "gdata", "combinat", "BGLR", "purrr"))
```

### Python
- numpy: 1.23.5
- pandas: 2.0.1
- matplotlib: 3.7.1
- scipy: 1.10.1
- seaborn: 0.12.2
- scikit-learn: 1.3.1
- tqdm: 4.65.0
- torch: 2.0.1+cu117
  
To install the required packages, run the following command in your terminal or command prompt:
```shell
pip install numpy==1.23.5 pandas==2.0.1 matplotlib==3.7.1 seaborn==0.12.2 scipy==1.10.1 scikit-learn==1.3.1 tqdm==4.65.0 torch==2.1.2
```
## Data simulation 
To simulate the genomic data run: 

[R_scripts/sim_geno.R](https://github.com/fshokor/DLGBLUP/blob/main/R_scripts/sim_geno.R)

--> Outputs: 
- Genomic matrix coded as 0, 1, and 2 (.txt file)
- Phased data (.Rdata)

To simulate the phenotypic and genetic values data run [R_scripts/sim_pheno.R](https://github.com/fshokor/DLGBLUP/blob/main/R_scripts/sim_pheno.R)

--> Output: (.csv files) 
- Phenotypes
- True Genetic Value
- QTL_df with the position and the effect of the common QTL
- ErrQTLEffect_df with the effect of specific QTLs for all dependent traits
- ErrQTLpos_df the positions of specific QTLs for all dependent traits 

All this files can be found in [Data](https://github.com/fshokor/DLGBLUP/tree/main/Data) directory

## Run GBLUP
Run a GBLUP multitrait model using [R_scripts/MTGBLUP_Additive.R](https://github.com/fshokor/DLGBLUP/blob/main/R_scripts/MTGBLUP_Additive.R) 

--> Output: a data frame of predicted genetic value for all traits

## DLGBLUP model
To train DLGBLUP model and analyse the results use [DL_Models/DLGBLUP.ipynb](https://github.com/fshokor/DLGBLUP/blob/main/DL_Models/DLGBLUP.ipynb)

## Citation
If you find the work helpful, you can cite it using the following citation:
```shell
@article{
shokor2024predicting,
title={Predicting nonlinear genetic relationships between traits in multi-trait evaluations by using a GBLUP-assisted Deep Learning model},
author={Fatima Shokor and Pascal Croiseau and Hugo Gangloff and Romain Saintilan and Thierry Tribout and Tristan Mary-Huard and Beatriz C.D. Cuyabano},
journal={bioRxiv},
year={2024},
doi={10.1101/2024.03.23.585208},
url={https://doi.org/10.1101/2024.03.23.585208},
note={}
}
```
