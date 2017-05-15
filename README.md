This code demonstrates an application of linear discriminant analysis ([LDA](http://sebastianraschka.com/Articles/2014_python_lda.html)) for dimensionality reduction on eye-movement behaviors recording during a visual search experiment. This data set containing several different eye-movement measures organized by search outcome (Tp, Fp, Tn, Fn) from easy and experimental treatment (easy, hard). This code uses LDA to determine the combination of variables that best separates the different classes. LDA finds a linear combination of the predictors that gives maximum separation between the centers of the data while at the same time minimizing the variation within each group of data. Here we conduction LD Analysis on searchbehavior data to determine if different search types and difficulties can be separated into unique classes.

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

Creating the LDA model
======================

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

|                  |        Fn|       Fp|        Tn|                                                                                                                                                                                   Tp|
|------------------|---------:|--------:|---------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| FalseNegative    |     46.90|     7.59|     43.45|                                                                                                                                                                                 2.07|
| FalsePositive    |     14.29|    53.85|      0.00|                                                                                                                                                                                31.87|
| TrueNegative     |     24.66|     0.34|     73.65|                                                                                                                                                                                 1.35|
| TruePositive     |      1.69|     7.46|      0.34|                                                                                                                                                                                90.51|
| Correct classifi |  cation r|  ates go|  along th|  e diagonal. This model performs best at classifying True Positive search outcomes, with a success rate of 91%, and is pretty dismal at correctly classifying False Negatives (47%).|

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

LD1 explains a whopping ~92% of the variance in the orginal data set. Combined with LD2 (~8%), these two LDs explain almost all of the observed variance.

Visualize Class Separations
===========================

Create data frame with LD projections (scores) based on original data

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

LD Scores by Search Outcome
---------------------------

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-18-1.png)

**Figure 1** Histogram of LD1 scores by Search Outcome class with visible means. Note separation between positive and negative search classes.

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-19-1.png)

**Figure 2** Histogram of LD2 scores by Search Outcome class with visible means. Note slight spearation between true and false search classes.

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-21-1.png)

**Figure 3** Cluster plot showing division of LD1 and LD2 across Search Outcome classes. Note "decent" separation among classes.

From examining these plots, it appears that LD1 is responsible for separating search outcomes into by positive (actioned) and negative (inactioned) searches. In contrast LD2 represents separation between true (successful) and false (unsucessful) search outcomes.

LD Scores by Difficulty Treatment
---------------------------------

``` r
#save difficulty treatment histogram as jpeg image
ggplot(data = h1.Data, aes(x = LDA.LD1, fill = DifTreatment)) + 
  # histogram
  geom_histogram(alpha = 0.4, position = "identity", breaks = seq(-5, 8, 0.5)) + 
  # mean lines
  geom_vline(data = h1.DifMus, aes(xintercept = LD1.mean, 
             colour = DifTreatment), linetype = "dashed") +  
  labs(x = "LD1 Score", y = "Count") + # axes labels and plot titles
  ggtitle("LD1 Score by Experimental Treatment: Easy vs Hard", 
          subtitle = "LDA estimated with 4 classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
```

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-22-1.png)

**Figure 4** Histogram of LD1 scores by experimental Difficulty Treatment with visible class means. Not much separation here.

``` r
#save difficulty treatment histogram as jpeg image
ggplot(data = h1.Data, aes(x = LDA.LD2, fill = DifTreatment)) + 
  # histogram
  geom_histogram(alpha = 0.4, position = "identity", breaks = seq(-5, 8, 0.5)) + 
  # mean lines
  geom_vline(data = h2.DifMus, aes(xintercept = LD2.mean, 
             colour = DifTreatment), linetype = "dashed") +  
  labs(x = "LD2 Score", y = "Count") + # axes labels and plot titles
  ggtitle("LD2 Score by Experimental Treatment: Easy vs Hard", 
          subtitle = "LDA estimated with 4 classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
```

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-23-1.png)

**Figure 5** Histogram of LD2 scores by experimental Difficulty Treatment with visible class means. Slightly better seperation than achieved with LD1 transformation.

``` r
# show division of ld1 and ld2 scores across classes as cluster plots
ggplot(cluster1.Data) + 
  geom_point(aes(LDA.LD1, LDA.LD2,colour = DifTreatment, shape = DifTreatment), 
             size = 2.0) + 
  labs(x = paste("LD1 (", percent(prop.VarExp1[1, 1]), ")"), y = 
       paste("LD2 (", percent(prop.VarExp1[2, 1]), ")")) +
  ggtitle("Experimental Treatments in LD Space: Easy vs Hard", 
          subtitle = "LDA Constructed with 4 Classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
```

![](SearchBehaviorsLDA_files/figure-markdown_github/unnamed-chunk-24-1.png)

**Figure 6** Cluster plot showing division of LD1 and LD2 across experimental Difficulty Treatment classes. It's not really fantastic.

LD2 does a slightly better job of separating the data based on difficulty treatment. While there is still quite a bit of overlap, Fig 5 shows searches during the Hard treatment ("least") had higher LD2 scores on average than searches during the Easy treatment ("most").

Relating original variables to LD1 and LD2
==========================================

Coefficient values of each variable in the discriminant function equations for LD1 and LD2
------------------------------------------------------------------------------------------

``` r
original.Components = data.frame(m2.lda$scaling)
kable(original.Components[,1:2], 
      caption = "Coefficients of linear discriminants", digits = 2)
```

|                  |    LD1|    LD2|
|------------------|------:|------:|
| NumSamplingTrips |  -1.21|  -0.41|
| NumFixComp       |  -0.03|  -0.57|
| NumGazeComp      |   0.32|  -0.03|
| TotNumFix        |   0.04|   0.56|
| AvgDurFix        |  -0.05|   0.14|
| Fix2End          |  -0.55|   0.59|
| ActiveSearchTime |   1.03|   0.87|

For LD1, the orginal variables with the largest coeffient (beta) in the LD1 function Were: NumSamplingTrips, ActiveSearchTime, Fix2End, and NumGazeComp, in that order. These orginal variables were of the greatest importance when finding the function that resulted in the best discrimination between classes (explained the most variance between classes). In contrast, the original variables NumFixComp, TotNumFix, and AvgDurFix have coefficients close to zero, and therefore neighter contributed much to the resulting function for LD1 nor explained any variance between classes. Similar interpretations can be done for LD2.

Interpreting LD1 and LD2 with variable loadings
-----------------------------------------------

Correlate the original variable values with the LD "scores" to determine how each original variable relates each LD function.

``` r
search.Data.4cor = data.frame(NumSamplingTrips = search.Data$NumSamplingTrips,
                              NumFixComp = search.Data$NumFixComp, 
                              NumGazeComp = search.Data$NumGazeComp, 
                              TotNumFix = search.Data$TotNumFix, 
                              AvgDurFix = search.Data$AvgDurFix, 
                              Fix2End = search.Data$Fix2End, 
                              ActiveSearchTime = search.Data$ActiveSearchTime)
loadings.Components = cor(search.Data.4cor, m2.pred$x)
kable(data.frame(loadings.Components)[,1:2], 
      caption = "Loadings of original variables", digits = 2)
```

|                  |    LD1|   LD2|
|------------------|------:|-----:|
| NumSamplingTrips |  -0.77|  0.54|
| NumFixComp       |   0.57|  0.30|
| NumGazeComp      |   0.49|  0.38|
| TotNumFix        |   0.07|  0.84|
| AvgDurFix        |  -0.11|  0.16|
| Fix2End          |  -0.78|  0.57|
| ActiveSearchTime |   0.52|  0.80|

Note that for both LD1 and LD2, some of the orginal variables that had negligible coefficient values now have sizeable "loadings" along the LDs. However, just because these predictors have high "loadings" does not mean that they contributed significantly to the LD. Remember, the coefficients represent predictor betas of the original variables that explain variation while these loadings only represent correlations between the original variables and the LD function. Because NumFixComp, TotNumFix, and AvgDur contributed so little to the LD1 function and explained so little variation, there really isn't much point in interpreting their loadings. Same goes for LD2.

Summary
=======

After defining our classes as search outcomes, we identified two LD functions that accounted for nearly all of the variance (92%, 8%) in our data by using LDA for dimensionality reduction (Table 6). Along LD1, the search outcomes were distinctly clustered by positive and negative search outcomes (Fig. 1 & 3), but there was little separation between difficulty treatment classes (Fig. 4 & 6). LD2 appeares to search outcomes based on whether a search resulted in a false or true finding (Fig. 2 & 3), but it too does poorly (althought slightly better than LD1) when separating by difficulty treatment (Fig. 5 & 6).

We discovered that out of our 7 original variables, only 4 really contributed to =LD1, so only these four contribted to explaining 92% of the variance between classes. Looking at the coeffients and loadings of these four original variables, we can conclude that relative to positive searches, negative searches are characterized by lower values of NumSamplingTrips and Fix2End, and higher values of ActiveSearchTime and NumGazeComp.
