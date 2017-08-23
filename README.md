Dimensionality Reduction of Eye-Movement Behaviors with LDA
==================

Olivia Guayasamin
5/24/2017

Summary
------------------------
This code demonstrates an application of linear discriminant analysis ([LDA](http://sebastianraschka.com/Articles/2014_python_lda.html)) for dimensionality reduction on eye-movement behaviors recording during a visual search experiment. This data set contains several different eye-movement measures organized by search outcome (Tp, Fp, Tn, Fn). This code implements LDA analysis using the ([MASS](https://cran.r-project.org/web/packages/MASS/MASS.pdf)) package to determine the combinations of variables that best separate the different classes (search outcomes & experimental treatments), and if different classes are characterized by different combinations of behaviors. This analysis is meant to answer questions like, *"Are distinct search classes described by different patterns of eye movement behaviors?"*, or *"How do subjects differ in their behavior during successful and unsuccessful searches?"*, while also describing the behavior combinations of interest.

While PCA is a more commonly used technique for dimensionality reduction, LDA is more appropriate for our current questions. This is because LDA finds linear combinations of the predictors (original variables) that gives *maximum separation* between the centers of the data classes while at the same time minimizing the variation within each group of data. These linear combinations are called Linear Discriminants, or LDs. LDA is able to do this because it is a "supervised" machine learning technique, meaning that each entry (data row) of predictors is associated with a class, and LDA takes the class into account when doing dimensionality reduction.

In contrast, PCA is an "unsupervised" form of machine learning, meaning that it is given no information about the data class to which an entry of predictors belong. Instead, blind to data class, it tries to find the linear combinations of predictors that account for the most variation within the entire data set. In PCA analysis, these linear combinaitons are know as Principal Components, or PCs. In this way, LDA can be considered a type of classifcation machine learning (especially when used to make predictions), while PCA is a method of clustering data.

**Summary:** LDA on search behavior data to determine if different search types and difficulties can be separated into unique classes. Converts individual behaviors to LDA scores. Visualize scores depending on search outcomes and experimental difficulty. **Requires:** Included R libraries, data files. **Outputs:** the results of LDA anlaysis as a cluster plot with data points colored by search type and difficulty treatment.

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
librar(broom)
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

Specify LDA model, using search outcomes as our "dependent" variable and our list of search behaviors as our "independent" variables. Because we first want to check the performance of our LDA model before we draw any conclusions from it, we need to indicate in our code that we want to include a cross-validation model validation technique. This will enable us to check how accurately the model can assign data to a particular search outcome.

``` r
# training model with cross validation
m1.lda <- lda(search.Class ~ NumSamplingTrips + NumFixComp + TotTimeFix +
              TotNumFix + AvgDurFix + Fix2End + ActiveSearchTime, 
              data = search.Data, CV = TRUE)
```

|               |     Fn|     Fp|     Tn|     Tp|
|---------------|------:|------:|------:|------:|
| FalseNegative |  46.90|   7.59|  43.45|   2.07|
| FalsePositive |  14.29|  53.85|   0.00|  31.87|
| TrueNegative  |  24.66|   0.34|  73.65|   1.35|
| TruePositive  |   1.69|   7.46|   0.34|  90.51|

Correct classification rates go along the diagonal. This model performs best at classifying True Positive search outcomes, with a success rate of 91%, and is pretty dismal at correctly classifying False Negatives (46%). Interestingly, when the model does misclassify, it tends to do so only with searches that share the same positive/negative outcome. For example, False Positives are correctly identified 53% of the time and mistaken as True Positives 32%, but *never* mistaken for True Negatives, and only rarely mis-labeled as False Negatives (15%). This actually let's us know that our model is working. Because search outcomes that are labelled as "positives" are those searches associated with a mouse-click, they should have a slightly different set of behaviors compared to the "negative" search outcomes.

Examine contents of our LDA model
---------------------------------

Now we are going to re-run our model, but without including cross-validation. This will enable us to check our model input to make sure we didn't make a mistake. More importantly, this will now cause the lda() function to return more information about the linear discriminants (LD) returned by our LDA model instead of just describing our model's performance. Now, we'll be able to see how much variation each LD is able to explain and how the search behaviors contribute to each LD.

### Quick check to make sure the input is correct.

Check the number of observations per Search Outcome class.

|     |     |
|:----|----:|
| Fn  |  290|
| Fp  |   91|
| Tn  |  296|
| Tp  |  295|

The prior probabilities of each class.

|     |      |
|:----|-----:|
| Fn  |  0.30|
| Fp  |  0.09|
| Tn  |  0.30|
| Tp  |  0.30|

Means of original variables by class.

|     |  NumSamplingTrips|  NumFixComp|  NumGazeComp|  TotNumFix|  AvgDurFix|
|-----|-----------------:|-----------:|------------:|----------:|----------:|
| Fn  |              0.27|       -0.45|        -0.40|      -0.33|       0.05|
| Fp  |             -1.05|        0.06|        -0.13|      -0.85|      -0.29|
| Tn  |              0.82|       -0.34|        -0.26|       0.27|       0.12|
| Tp  |             -0.76|        0.76|         0.69|       0.31|      -0.08|

These values match those of the orginal data set, so it looks like we entered everything correctly. Now let's examine the LDs returned by our analysis.

### Variance explained by each LD

An additional (not to mention easy) way to check that you didn't enter incorrect class information into the model is to check the number of output LDs. The number of LDs produced by an LDA model is always equal to the number of classes minus 1. Here I have three LD's, which is expected given the fact that I have four classes. In addition, you should check to make sure that the cumulative variance explained by all of the LDs sums to 100%, otherwise something funky is going on with how your model treats variance (although this last check is much more likely to be relevant for those who are writing their own functions for performing LDA).

|     |  VarianceExplained|
|-----|------------------:|
| LD1 |              0.922|
| LD2 |              0.076|
| LD3 |              0.001|

LD1 explains a whopping ~92% of the variance in the orginal data set. Since the other two don't explain much of anything, so we won't include it in later analysis. What little additional insight we would get probably won't be worth the increased chance of Type I error that would accompany each additional statistical test.

Visualize Class Separation based on Search Outcome
==================================================

Now, we transform our set of original predictors using LD projections to get each data row's predicted LD "value"" according to the LD1 and LD2 functions. If you are more familiar with PCA, LD "scores" are the equivalent to PC scores. By transforming our original data using the LD projections, and then plotting the LD scores values by class, we can visualize how well the LDA separates data belonging to distinct classes.

Calculate LD "scores" using LD projections
------------------------------------------

First, we create will create a data table containing our estimated LD scores.

``` r
# model predictions
m2.pred <-predict(m2.lda, newdata = search.Data[, c(6, 7, 10, 9, 11, 12, 13)])
# data frames for histograms
h1.Data = data.frame(cbind(search.Data[, c(1,2,4,5)], DifTreatment = 
                     search.Data$ExpVer, LDA = m2.pred$x))  
```

| SearchType | DifTreatment |   LDA.LD1|     LDA.LD2|
|:-----------|:-------------|---------:|-----------:|
| Tp         | least        |  2.405417|   0.6194023|
| Tp         | least        |  2.329040|   0.2060947|
| Tp         | least        |  4.201326|   0.8660769|
| Tp         | least        |  3.028117|   1.1994827|
| Tp         | most         |  2.593909|  -0.4063113|
| Tp         | most         |  2.288270|  -0.7972150|

Plot LD1 Scores by Search Outcome
--------------------------------

Then, we plot our LD1 scores, labelling each score by search outcome.

To visualize our data, we will first implement a "cluster" plot, so that we can see how the searches are separated along LD1 and LD2 (just in case LD2 ends up being interesting after all).

![](https://github.com/oguayasa/eyemovement-lda/blob/master/imgs/clust1.jpg)

**Figure 1** Cluster plot showing division of LD1 and LD2 across Search Outcome classes. Note "decent" separation among classes along LD1, but a lot of overlap along LD2. Turns out LD2 doesn't add much after all, so now we should feel pretty safe in excluding it from later analyses.

Now we will use a histogram to better visualize that separate along LD1, and include markers that indicate the average of each search outcome.

![](https://github.com/oguayasa/eyemovement-lda/blob/master/imgs/hist1.jpg)

**Figure 2** Histogram of LD1 scores by Search Outcome class with visible means. Here, we more clearly see the separation between positive and negative search classes.

The same patterns we saw when examining the model predictions are presenting themselves in the LD1 scores as well. Namely, the model does well at separating positive and negative search classes, but falters at distinguishing between true and false outcomes within those classes.. However, the separation among between classes is *much* better along LD1 than LD2 (Fig. 2 & 3 respectively), which is not a surprising result given that LD1 explained ~92% of the variation in this data, while LD2 only accounted for ~7%.

Relating original variables to LD1 and LD2
==========================================

For those more familiar with linear regression, LDA conducted with more than two classes is similar to a Multivariate Multiple Linear Regression or Mutltivariate General Linear Model, where there are multiple dependent variables (outcomes or classes) and multiple independent variables (predictors) that have interrationships described by significant linear correlations.

Examine coefficient values of each variable for LD1
---------------------------------------------------

The separate classes are synonymous with categorical outcome variables (or factors depending on your statistical software), and the coefficients of the linear discriminant functions are equivalent to, and perhaps more easily visualized as, the *β* values associated with each predictor variable in a multivariate linear model. Therefore, just like in multivariate regression, the predictors with large absolute *β* values contributed the most to the model.

|                  |    LD1|    LD2|
|------------------|------:|------:|
| NumSamplingTrips |  -1.11|  -0.44|
| NumFixComp       |   0.06|  -0.58|
| TotTimeFix       |  -0.20|  -0.03|
| TotNumFix        |   0.07|   0.58|
| AvgDurFix        |  -0.06|   0.15|
| Fix2End          |  -0.64|   0.61|
| ActiveSearchTime |   1.31|   0.86|

Here, For LD1, the orginal variables with the largest coefficients (*β*) in the LD1 function Were: NumSamplingTrips, ActiveSearchTime, Fix2End, and TotTimeFix, in that order. These orginal variables were of the greatest importance when finding the function (linear combination of original predictors) that gave the best separation between classes. In contrast, the original variables NumFixComp, TotNumFix, and AvgDurFix have *β* values close to zero, and therefore neither contributed much to the resulting function for LD1 nor explained any variance between classes.

-------------------------------------------------------

Now, we will correlate the original variable values with the LD1 "scores" to determine how each original variable relates each LD function. In effect, we are creating variable "loadings" for our LDs, much like is done with PCs in PCA.

|                  |    LD1|   LD2|
|------------------|------:|-----:|
| NumSamplingTrips |  -0.77|  0.54|
| NumFixComp       |   0.58|  0.29|
| TotTimeFix       |   0.38|  0.72|
| TotNumFix        |   0.07|  0.84|
| AvgDurFix        |  -0.11|  0.16|
| Fix2End          |  -0.78|  0.57|
| ActiveSearchTime |   0.53|  0.80|

Note that for both LD1 some of the orginal variables that had negligible coefficient values now have sizeable "loadings" along the LDs. However, just because these predictors have high "loadings" *does not mean that they made significant contributions to the class separation* described by the LD functions. Remember, coefficients represent *β* values of the original variables in a multivariate linear model. On the other hand, these loadings only represent correlations between the original variables and the scores derived from the LD function. Because NumFixComp, TotNumFix, and AvgDur contributed so little to the LD1 function and explained so little variation, there really isn't much point in including their loadings in the interpretation of LD1.

Having said that, let's look at the loadings of our relevant variables. NumSamplingTrips and Fix2End have strong negative loadings, while ActiveSearchTime and TotNumFix have strong positive loadings. This means that the larger the LD1 score, the more separate visiting and total search time *and* less time spent searching in the target areas and fewer fixations. Looking at how the positive searches have highter values (on avearge) than the negative searches (See Fig. 2), this indicates that positive searches required more visits and a longer total search time, but surprisingly less time spent in the target areas and fewer fixations.

Final Steps
===========

Within the positive and negative search outcomes, the True and False search outcomes overlap quite a bit, but they don't seem identical. To check if Tp and Fp, and Tn and Fn distributions are significnatly different, let's quickly examining the median values and running a paired significance test.

Compare LD1 Scores betwenn Tp & Fp, and Tn & Fn
-----------------------------------------------

We will first report the median values, and then use a Wilcoxon Signed-Rank test (a dependent, non-parametric t-test) to determine if the two distriubtions LD1 scores are significantly different from each other or not.

``` r
# run a wilcoxon signed rank test
ld1.Test1 <- wilcox.test(Tn.Scores, Fn.Scores, 
                        paired = TRUE, conf.int = TRUE, 
                        na.action = na.exclude)

# run a wilcoxon signed rank test
ld1.Test2 <- wilcox.test(Tp.Scores, Fp.Scores, 
                        paired = TRUE, conf.int = TRUE, 
                        na.action = na.exclude)
```

|     |  Medians|
|-----|--------:|
| Tp  |     2.32|
| Tn  |    -1.75|
| Fn  |    -1.16|
| Fp  |     1.12|

|  estimate|  statistic|  p.value|  conf.low|  conf.high| method                                               | alternative |
|---------:|----------:|--------:|---------:|----------:|:-----------------------------------------------------|:------------|
|     -0.44|      10961|        0|     -0.56|      -0.32| Wilcoxon signed rank test with continuity correction | two.sided   |

|  estimate|  statistic|  p.value|  conf.low|  conf.high| method                                               | alternative |
|---------:|----------:|--------:|---------:|----------:|:-----------------------------------------------------|:------------|
|      1.14|       3788|        0|      0.91|       1.38| Wilcoxon signed rank test with continuity correction | two.sided   |

It looks like the LD1 scores for Tn & Fn and Tp & Fp searches are significantly different.

Let's first look at the comparison between Tn and Fn searches. From the median values, we can see that Tn LD1 scores are slightly more negative than scores during Fn searches. This is interesting. Remember than Fn searches are cases where a target was present, but the subject failed to find it. The fact that Fn searches are more similar to Tp searches indicates that even though the search outcome was the same (negative), *subjects behaved differently depending on whether a target was or was not there*. This indicates that there are slightly different cognitive processes occuring during Tn and Fn searches, even if the subjects isn't entirely aware of it.

Now looking at Tp and Fp searches, we can see that Fp searches have a more negative LD1 score on average than Tp earches. This means that an Fp search (when a subject believed a target was present, but in reality there was nothing there) is not only different from a Tp search, but its *more similar to negative searches* tha Tp searches are (cases where subjects decided there was nothing there). Again, it appears that there are different cognitive processes going on during the time preceeding the decisions about whether a target was there or not, even if the outcome of the decision was the same.

Conclusions
===========

There's some really cool stuff here, but the story isn't done yet! In the next section, we're going to look more depth at how both search outcome *and* difficulty treatment influence LD1 (search behavior) score, using Glmms.
