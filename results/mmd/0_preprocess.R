#!/usr/bin/env Rscript
library(tidyverse)
library(R.matlab)

xpr <- read_tsv('esets_breast_exprs_genes.txt')
X <- as.matrix(select(xpr, -X1))
genes <- xpr$X1

pheno <- read_tsv('E-MTAB-6703.sdrf.txt')
y <- as.numeric(pheno$`Characteristics[disease]` == 'tumor')
names(y) <- pheno$`Source Name`

y <- y[colnames(X)]

writeMat('mmd_brca.mat', X = t(X), Y = y, genes = genes)
