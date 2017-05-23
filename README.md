This code demonstrates an application of linear discriminant analysis ([LDA](http://sebastianraschka.com/Articles/2014_python_lda.html)) for dimensionality reduction on eye-movement behaviors recording during a visual search experiment. This data set contains several different eye-movement measures organized by search outcome (Tp, Fp, Tn, Fn). This code implements LDA analysis using the ([MASS](https://cran.r-project.org/web/packages/MASS/MASS.pdf)) package to determine what combinations of variables best separate the different classes (search outcomes & experimental treatments), and if different classes are characterized by different combinations of behaviors. This analysis is meant to answer questions like, *"Are distinct search classes described by different patterns of eye movement behaviors?"*, or *"How do subjects differ in their behavior during successful and unsuccessful searches?"*, while also describing the behavior combinations of interest.

While PCA is a more commonly used technique for dimensionality reduction, LDA is more appropriate for our current questions. This is because LDA finds linear combinations of the predictors (original variables) that gives *maximum separation* between the centers of the data classes while at the same time minimizing the variation within each group of data. LDA is able to do this because it is a "supervised" technique, meaning that each entry (data row) of predictors is associated with a class, and LDA takes the class into account when doing dimensionality reduction.

In contrast, PCA is "unsupervised", meaning that it is given no information about the data class to which an entry of predictors belong. Instead, blind to data class, it tries to find the linear combinations of predictors that account for the most variation within the entire data set.

Summary: Conduct two separate LDA on search behavior data to determine if different search types and difficulties can be separated into unique classes. Converts individual behaviors to LDA scores. Compare scores across 1.) search outcomes to see if there are significantly different search behaviors depending on search success, and 2.) difficulty treatments.

Requires: Included R libraries, data files

Outputs: the results of LDA anlaysis as a cluster plot with data points colored by search type and difficulty treatment.

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

Import and format data
----------------------

``` r
tp.Data <- read_excel("val_TP_SearchBehav.xlsx")
fn.Data <- read_excel("val_Fn_SearchBehav.xlsx")
fp.Data <- read_excel("val_FP_SearchBehav.xlsx")
tn.Data <- read_excel("val_Tn_SearchBehav.xlsx")
```

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

Creating the LDA model using Search Outcome Classes
===================================================

Check model performance using cross validation
----------------------------------------------

Specify LDA model

``` r
# training model with cross validation
m1.lda <- lda(search.Class ~ NumSamplingTrips + NumFixComp + NumGazeComp + 
              TotNumFix + AvgDurFix + Fix2End + ActiveSearchTime, 
              data = search.Data, CV = TRUE)
```

Classification results

|               |     Fn|     Fp|     Tn|     Tp|
|---------------|------:|------:|------:|------:|
| FalseNegative |  46.90|   7.59|  43.45|   2.07|
| FalsePositive |  14.29|  53.85|   0.00|  31.87|
| TrueNegative  |  24.66|   0.34|  73.65|   1.35|
| TruePositive  |   1.69|   7.46|   0.34|  90.51|

Correct classification rates go along the diagonal. This model performs best at classifying True Positive search outcomes, with a success rate of 91%, and is pretty dismal at correctly classifying False Negatives (47%).

Examine contents of our LDA model
---------------------------------

The number of observations per Search Outcome class

|     |Count|
|:----|----:|
| Fn  |  290|
| Fp  |   91|
| Tn  |  296|
| Tp  |  295|

The prior probabilities of each class

|     |Priors|
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

### Variance explained by each LD

An easy way to check that you didn't enter incorrect class information into the model is to check the number of output LDs. The number of LDs produced by an LDA model is always equal to the number of classes minus 1. Here I have three LD's, which is expected given the fact that I have four classes. In addition, you should check to make sure that the cumulative variance explained by all of the LDs sums to 100%, otherwise something funky is going on with how your model treats variance (although this last check is much more likely to be relevant for those who are writing their own functions for performing LDA).

|     |  VarianceExplained|
|-----|------------------:|
| LD1 |              0.922|
| LD2 |              0.076|
| LD3 |              0.001|

LD1 explains a whopping ~92% of the variance in the orginal data set, and combined with LD2 (~7%), these two LDs explain almost all of the observed variance. LD3 does explain much of anything, so we won't include it in later analysis.

Visualize Class Separation based on Search Outcome
==================================================

Now, we transform our set of original predictors using LD projections to get each data row's predicted LD "value"" according to the LD1 and LD2 functions. If you are more familiar with PCA, LD "scores" are the equivalent to PC scores. By transforming our original data using the LD projections, and then plotting the LD scores values by class, we can visualize how well the LDA separates data belonging to distinct classes.

Calculate LD "scores" using LD projections
------------------------------------------

Create data frame with LD projections (scores) by using the LDA model to estimate LD1 and LD2 values from the original data.

``` r
# model predictions
m2.pred <-predict(m2.lda, newdata = search.Data[, c(6, 7, 8, 9, 11, 12, 13)])
# data frames for histograms
h1.Data = data.frame(SearchType = search.Data$search.Class, DifTreatment = 
                     search.Data$ExpVer, LDA = m2.pred$x)  
```

| SearchType | DifTreatment |   LDA.LD1|     LDA.LD2|
|:-----------|:-------------|---------:|-----------:|
| Tp         | least        |  2.405417|   0.6194023|
| Tp         | least        |  2.329040|   0.2060947|
| Tp         | least        |  4.201326|   0.8660769|
| Tp         | least        |  3.028117|   1.1994827|
| Tp         | most         |  2.593909|  -0.4063113|
| Tp         | most         |  2.288270|  -0.7972150|

Plot LD Scores by Search Outcome
--------------------------------

![](https://github.com/oguayasa/eyemovement-lda/blob/master/imgs/hist1.jpg)

**Figure 1** Histogram of LD1 scores by Search Outcome class with visible means. Note separation between positive and negative search classes.

![](https://github.com/oguayasa/eyemovement-lda/blob/master/imgs/hist2.jpg)

**Figure 2** Histogram of LD2 scores by Search Outcome class with visible means. Note slight spearation between true and false search classes.

![](https://github.com/oguayasa/eyemovement-lda/blob/master/imgs/clust1.jpg)

**Figure 3** Cluster plot showing division of LD1 and LD2 across Search Outcome classes. Note "decent" separation among classes.

From examining these plots, it appears that LD1 is responsible for separating search outcomes into by positive (actioned) and negative (inactioned) searches. In contrast LD2 represents separation between true (successful) and false (unsucessful) search outcomes.

From examining these plots, it appears that LD1 is responsible for separating search outcomes into by positive (actioned) and negative (inactioned) searches. In contrast LD2 represents separation between true (successful) and false (unsucessful) search outcomes. Despite clear clusters (Fig. 1), the different search outcome classes overlap along the LD1 and LD2 axes. However, the separation among between classes is *much* better along LD1 than LD2 (Fig. 2 & 3 respectively), which is not a surprising result given that LD1 explained ~92% of the variation in this data, while LD2 only accounted for ~7%.

Relating original variables to LD1 and LD2
==========================================

For those more familiar with linear regression, LDA conducted with more than two classes is similar to a Multivariate Multiple Linear Regression or Mutltivariate General Linear Model, where there are multiple dependent variables (outcomes or classes) and multiple independent variables (predictors) that have interrationships described by significant linear correlations.

The eparate classes are synonymous with categorical outcome variables (or factors depending on your statistical software), and the coefficients of the linear discriminant functions are equivalent to, and perhaps more easily visualized as, the *β* values associated with each predictor variable in a multivariate linear model.

Therefore, just like in multivariate regression, the predictors with large absolute *β* values contributed the most to the model.

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

Here, For LD1, the orginal variables with the largest coefficients (*β*) in the LD1 function Were: NumSamplingTrips, ActiveSearchTime, Fix2End, and NumGazeComp, in that order. These orginal variables were of the greatest importance when finding the function (linear combination of original predictors) that gave the best separation between classes. In contrast, the original variables NumFixComp, TotNumFix, and AvgDurFix have *β* values close to zero, and therefore neighter contributed much to the resulting function for LD1 nor explained any variance between classes. Similar interpretations can be done for LD2.

Interpreting LD1 and LD2 with variable loadings
-----------------------------------------------

Correlate the original variable values with the LD "scores" to determine how each original variable relates each LD function.

|                  |    LD1|   LD2|
|------------------|------:|-----:|
| NumSamplingTrips |  -0.77|  0.54|
| NumFixComp       |   0.57|  0.30|
| NumGazeComp      |   0.49|  0.38|
| TotNumFix        |   0.07|  0.84|
| AvgDurFix        |  -0.11|  0.16|
| Fix2End          |  -0.78|  0.57|
| ActiveSearchTime |   0.52|  0.80|

Note that for both LD1 and LD2, some of the orginal variables that had negligible coefficient values now have sizeable "loadings" along the LDs. However, just because these predictors have high "loadings" does not mean that they made significant contributions to the class separation described by the LD functions. Remember, coefficients represent *β* values of the original variables in a multivariate linear model. On the other hand, these loadings only represent correlations between the original variables and the scores derived from the LD function. Because NumFixComp, TotNumFix, and AvgDur contributed so little to the LD1 function and explained so little variation, there really isn't much point in including their loadings in the interpretation of LD1. The same analysis can be repeated for LD2.

Summary
=======

Search Outcomes
---------------

After defining our classes as search outcomes, we identified two LD functions that accounted for nearly all of the variance (92%, 8%) in our data by using LDA for dimensionality reduction (Table 6). Along LD1, the search outcomes were distinctly clustered by positive and negative search outcomes (Figs. 1 & 2), while LD2 did a better job separating search outcomes based on whether search resulted in a false or true finding (Figs. 1 & 3).

We discovered that out of our seven original variables, four contributed to LD1 and gave the greatest separation between groups, so only these four contribted to explaining 92% of the variance between classes (Table 8). Looking at the coeffients and correlational loadings of these four original variables, we can conclude that searches that scored higher on LD1 were those where subjects made very few passes (NumSamplingTrips), with lots of local comparisons that lead to significant active search time (NumFixComp, ActiveSearchTime) and yielded a low total duration of search (Fix2End). On the other hand, interpreting LD2 is a bit more complicated, but it looks like searches with higher LD2 scores were more thorough, with greater amounts of all search behaviors.

On average, positive searches had higher LD1 values than negative searches (Fig. 2) indicating that positive searchers shorter and more intense than negative searches. Correct searches had slightly higher LD2 scores than incorrect searches (Fig. 3), suggesting that they were more thorough searches, but this difference is small.

In a future post, we will talk about how we could formally compare LD1 and LD2 scores across classes using GLMMs (as an alternative to the ANOVA approach).


