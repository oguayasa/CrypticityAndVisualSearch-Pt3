
# eyemovementLDA.R
R version 3.2.4 Revised
8/5/2017
e-mail: oguayasa@gmail.com

General Notes: Needs to be modified according to current data frame. 

Summary:
Applying linear discriminant analysis (LDA) on eye-movement behaviors for 
dimensionality reduction and to determine if visual searches can be 
separated based on search outcome (True Positive, False Positive, 
True Negative, False Negative), and experimental manipulation of search task
difficulty (Cryptic Targets vs. Non-Cryptic Targets). Determine LDA 
components, and visualize results using cluster plots color coded by class. 

PCA is an unsupervised learning technique (doesn't use class information) 
while LDA is a supervised technique (uses class information), but both provide
the possibility of dimensionality reduction. LDA finds a linear combination 
of the predictors that gives maximum separation between the centers of the 
data while at the same time minimizing the variation within each group of data.

Requires: Data files, MASS, ggplot2, readxl, scales, plyr

Outputs: Results of LDA dimensionality reduction in .doc file containing 
console output, jpeg files of histograms and cluster plots of LDA components 
by class

