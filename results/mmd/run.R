#!/usr/bin/env Rscript
library(tidyverse)
library(R.matlab)

args = commandArgs(trailingOnly=TRUE)
tissue = args[1]
phenotype_file = args[2]
num_features = args[3]

# create gene expression matrix
xpr <- read_tsv(paste0('esets_', tissue, '_exprs.txt'))
X <- as.matrix(select(xpr, -X1))
genes <- xpr$X1

# create phenotype vector
pheno <- read_tsv(phenotype_file)
y <- as.numeric(pheno$`Characteristics[disease]` == 'tumor')
names(y) <- pheno$`Source Name`

y <- y[colnames(X)]

# sample minor class to balance dataset
cnt <- table(y)
resamples <- sample(names(y)[y == names(which.min(cnt))], max(cnt) - min(cnt), replace = TRUE)

X <- cbind(X, X[,resamples])
y <- c(y, y[resamples])

writeMat(paste0(tissue, '.mat'), X = t(X), Y = y, genes = genes)

system2('python', args = c('../../../exp.py', './', paste0(tissue, '.mat'), 10, num_features))
