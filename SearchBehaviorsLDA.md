Dimensionality Reduction of Search Behaviors with LDA
================
Olivia Guayasamin
May 10, 2017

This code demonstrates an application of linear discriminant analysis ([LDA](http://sebastianraschka.com/Articles/2014_python_lda.html)) for dimensionality reduction. Using a data set containing several different eye-movement measures taken during easy and difficult visual search tasks, this code uses LDA to determine the combination of variables that best separates the different classes. LDA finds a linear combination of the predictors that gives maximum separation between the centers of the data while at the same time minimizing the variation within each group of data.Do LDA Analysis on search behavior data to determine if different search types and difficulties are unique classes.

Outputs the results of LDA anlaysis as a cluster plot with data points colored by search type and difficulty treatment.

Converts individual behaviors to LDA scores. Compare scores across classes to see if there are significantly different search behaviors exhibited across search types and difficulty treatments.

Initializing Steps
==================

Load libraries from CRAN
------------------------

``` r
library(scales)
library(MASS)
library(ggplot2)
library(readxl)
library(plyr)
library(knitr)
```

    ## [1] "C:/Users/oguayasa/Documents/R/val_SearchBehavLDA"

Import and format data
----------------------

Include variables identifying experimental treatments and search outcomes
-------------------------------------------------------------------------

| search.Class | PartID | ExpVer |  MediaOrder|  BlockOrder|  NumSamplingTrips|  NumFixComp|
|:-------------|:-------|:-------|-----------:|-----------:|-----------------:|-----------:|
| Tp           | P01    | least  |           1|           1|             -0.93|        0.38|
| Tp           | P01    | least  |           1|           2|             -1.10|        0.89|
| Tp           | P01    | least  |           1|           3|             -0.58|        6.43|
| Tp           | P01    | least  |           1|           4|             -0.82|        0.52|
| Tp           | P01    | most   |           2|           1|             -1.61|       -0.48|
| Tp           | P01    | most   |           2|           2|             -1.57|       -0.29|
| Tp           | P01    | most   |           2|           3|             -1.25|       -0.29|
| Tp           | P01    | most   |           2|           4|             -1.57|       -0.60|

Creating the LDA model
======================

Check model performance using cross validation
----------------------------------------------

Specify LDA model

Classification results

|               |     Fn|     Fp|     Tn|     Tp|
|---------------|------:|------:|------:|------:|
| FalseNegative |  46.90|   7.59|  43.45|   2.07|
| FalsePositive |  14.29|  53.85|   0.00|  31.87|
| TrueNegative  |  24.66|   0.34|  73.65|   1.35|
| TruePositive  |   1.69|   7.46|   0.34|  90.51|

Model performs best at classifying True Positive search outcomes, with a success rate of 91%, and the worse at correctly classifying False Negatives (47%)

Examine contents of our LDA model
---------------------------------

The number of observations per Search Outcome class

|     |     |
|:----|----:|
| Fn  |  290|
| Fp  |   91|
| Tn  |  296|
| Tp  |  295|

The prior probabilities of each class

|     |      |
|:----|-----:|
| Fn  |  0.30|
| Fp  |  0.09|
| Tn  |  0.30|
| Tp  |  0.30|

Means of original variables by class

|     |  NumSamplingTrips|  NumFixComp|  NumGazeComp|  TotNumFix|  AvgDurFix|
|-----|-----------------:|-----------:|------------:|----------:|----------:|
| Fn  |              0.27|       -0.45|        -0.40|      -0.33|       0.05|
| Fp  |             -1.05|        0.06|        -0.13|      -0.85|      -0.29|
| Tn  |              0.82|       -0.34|        -0.26|       0.27|       0.12|
| Tp  |             -0.76|        0.76|         0.69|       0.31|      -0.08|

Variance explained by each LD

|     |  VarianceExplained|
|-----|------------------:|
| LD1 |              0.922|
| LD2 |              0.076|
| LD3 |              0.001|

LD1 explains a whopping 92% of the variance in the orginal data set. Combined with LD2 (~8%), these two LDs explain almost all of the observed variance.

Visualize Class Separations
===========================

Create data frame with LD projections (scores) based on original data

| SearchType | DifTreatment |   LDA.LD1|     LDA.LD2|
|:-----------|:-------------|---------:|-----------:|
| Tp         | least        |  2.405417|   0.6194023|
| Tp         | least        |  2.329040|   0.2060947|
| Tp         | least        |  4.201326|   0.8660769|
| Tp         | least        |  3.028117|   1.1994827|
| Tp         | most         |  2.593909|  -0.4063113|
| Tp         | most         |  2.288270|  -0.7972150|

LD Scores by Search Outcome
---------------------------

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-16-1.png)

**Figure 1** Histogram of LD1 scores by Search Outcome class with visible means. Note separation between positive and negative search classes.

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-17-1.png)

**Figure 2** Histogram of LD2 scores by Search Outcome class with visible means. Note slight spearation between true and false search classes.

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-19-1.png)

**Figure 3** Cluster plot showing division of LD1 and LD2 across Search Outcome classes. Note "decent" separation among classes.

LD Scores by Difficulty Treatment
---------------------------------

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-20-1.png)

**Figure 4** Histogram of LD1 scores by experimental Difficulty Treatment with visible class means. Not much separation here.

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-21-1.png)

**Figure 5** Histogram of LD2 scores by experimental Difficulty Treatment with visible class means. Slightly better seperation than achieved with LD1 transformation.

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-22-1.png)

**Figure 6** Cluster plot showing division of LD1 and LD2 across experimental Difficulty Treatment classes. It's not really fantastic.

Relating original variables to LD1 and LD2
==========================================

Coefficient values of each variable in the discriminant function equations for LD1 and LD2
------------------------------------------------------------------------------------------

|                  |    LD1|    LD2|
|------------------|------:|------:|
| NumSamplingTrips |  -1.21|  -0.41|
| NumFixComp       |  -0.03|  -0.57|
| NumGazeComp      |   0.32|  -0.03|
| TotNumFix        |   0.04|   0.56|
| AvgDurFix        |  -0.05|   0.14|
| Fix2End          |  -0.55|   0.59|
| ActiveSearchTime |   1.03|   0.87|

Interpreting LD1 and LD2 with variable loadings
-----------------------------------------------

Correlate the original variable values with the LD "scores" to determine how each original variable loads onto each LD function

|                  |    LD1|   LD2|
|------------------|------:|-----:|
| NumSamplingTrips |  -0.77|  0.54|
| NumFixComp       |   0.57|  0.30|
| NumGazeComp      |   0.49|  0.38|
| TotNumFix        |   0.07|  0.84|
| AvgDurFix        |  -0.11|  0.16|
| Fix2End          |  -0.78|  0.57|
| ActiveSearchTime |   0.52|  0.80|

Summary
=======

After defining our classes as search outcomes, we identified two LD functions that accounted for nearly all of the variance in our data by using LDA for dimensionality reduction (Table 6). Along LD1, the search outcomes were distinctly clustered by positive and negative search outcomes (Fig. 1 & 3), but there was little separation between difficulty treatment classes (Fig. 4 & 6). LD2 appeares to search outcomes based on whether a search resulted in a false or true finding (Fig. 2 & 3), but it too does poorly (althought slightly better than LD1) when separating by difficulty treatment (Fig. 5 & 6).