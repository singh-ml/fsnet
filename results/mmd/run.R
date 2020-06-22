#!/usr/bin/env Rscript
library(tidyverse)
library(R.matlab)

args = commandArgs(trailingOnly=TRUE)
tissue = args[1]
phenotype_file = args[2]

xpr <- read_tsv(paste0('esets_', tissue, '_exprs.txt'))
X <- as.matrix(select(xpr, -X1))
genes <- xpr$X1

pheno <- read_tsv(phenotype_file)
y <- as.numeric(pheno$`Characteristics[disease]` == 'tumor')
names(y) <- pheno$`Source Name`

y <- y[colnames(X)]

writeMat(paste0(tissue, '.mat'), X = t(X), Y = y, genes = genes)

system2('python', args = c('../../../exp.py', './', paste0(tissue, '.mat), 10, 16,))
