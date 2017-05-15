# val_SearchBehavLDA.R
# 
# Summary:
# Do LDA Analysis on search behavior data to determine if different search
# types and difficulties are unique classes. 
#
# Plot output as cluster plot with data points colored by  by search type and 
# difficulty treatment.
#
# PCA is an unsupervised learning technique (doesn't use class information) 
# while LDA is a supervised technique (uses class information), but both provide
# the possibility of dimensionality reduction. 
#
# LDA finds a linear combination of the predictors that gives maximum separation 
# between the centers of the data while at the same time minimizing the 
# variation within each group of data.
#
# Requires: Data files, MASS, ggplot2, readxl, scales, plyr
#
# Outputs: Results of LDA dimensionality reduction in .doc file containing 
# console output, jpeg files of histograms and cluster plots of LDA components 
# by class
# 
# Date: 5/8/2017
# ----- Initializing steps -----

#clean up workspace and console
rm(list=ls())  # workspace
cat("\014")  # console

#load libraries
library(scales)
library(MASS)
library(ggplot2)
library(readxl)
library(plyr)

#set working directory
getwd()
setwd("~/R/val_SearchBehavLDA/")

# function for writing as percentages
percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}

# ----- Import and format data -----

# Import search behavior data files
tp.Data <- read_excel("val_TP_SearchBehav.xlsx")
fn.Data <- read_excel("val_Fn_SearchBehav.xlsx")
fp.Data <- read_excel("val_FP_SearchBehav.xlsx")
tn.Data <- read_excel("val_Tn_SearchBehav.xlsx")

search.Class <- rbind(matrix(rep('Tp', nrow(tp.Data))), matrix(rep('Tn', 
                                                                   nrow(tn.Data))), matrix(rep('Fn',  nrow(fn.Data))), 
                      matrix(rep('Fp', nrow(fp.Data))))

# Format data for LDA Analysis
raw.Data <- (rbind(tp.Data, tn.Data, fn.Data, fp.Data))
scaled.Var.Data <- data.frame(scale(raw.Data[, 5:12]))
id.Data <- data.frame(raw.Data[, 1:4])
search.Data <- cbind(search.Class, id.Data, scaled.Var.Data)

# Convert character vectors to 
# factors
search.Data$PartID <- factor(search.Data$PartID)  
search.Data$ExpVer <- factor(search.Data$ExpVer)

head(search.Data,3)  # check first 3 rows of data


# ----- Search Types: Create LDA and check performance with CV -----

# training model with cross validation
m1.lda <- lda(search.Class ~ NumSamplingTrips + NumFixComp + NumGazeComp + 
              TotNumFix + AvgDurFix + Fix2End + ActiveSearchTime, 
              data = search.Data, CV = TRUE)

head(m1.lda$posterior, 3) # class posterior probabilities
head(m1.lda$class, 3)# class predictions

# create and format table showing model performance
m1.Table <- table(search.Data$search.Class, m1.lda$class)
perf1.Table <- rbind(m1.Table[1, ]/sum(m1.Table[1, ]), 
                     m1.Table[2, ]/sum(m1.Table[2, ]),
                     m1.Table[3, ]/sum(m1.Table[3, ]), 
                     m1.Table[4, ]/sum(m1.Table[4, ]))
print(perf1.Table)


# ----- Search Types: Create LDA model and visualize class separation -----

# create model and examine model components

# prediction model
m2.lda <- lda(search.Class ~ NumSamplingTrips + NumFixComp + NumGazeComp + 
              TotNumFix + AvgDurFix + Fix2End + ActiveSearchTime, 
              data = search.Data, CV = FALSE)

m2.lda$N  # number of observations used in model
m2.lda$counts  # number of observations per class
m2.lda$prior  # prior probabilities of each class
m2.lda$means  # predictor means for each class
m2.lda$scaling  # lda, linear combination coefficients for each LD
m2.lda$svd  # singular values giving ratios of between and within group SDs 

# determine the proportion of variance explained by each LD
prop.VarExp1 <- m2.lda$svd^2/sum(m2.lda$svd^2)
print(prop.VarExp1)  # 4 classes so should be values for 3 ld

# create and format table showing model performance
m2.pred <-predict(m2.lda, newdata = search.Data[, c(6, 7, 8, 9, 11, 12, 13)])
m2.Table <- table(search.Data$search.Class, m2.pred$class)
perf2.Table <- rbind(m2.Table[1, ]/sum(m2.Table[1, ]), 
                     m2.Table[2, ]/sum(m2.Table[2, ]),
                     m2.Table[3, ]/sum(m2.Table[3, ]), 
                     m2.Table[4, ]/sum(m2.Table[4, ]))
print(perf2.Table)

# show division of ld1 scores across search types as histograms

# data frames for histograms
h1.Data = data.frame(SearchType = search.Data$search.Class, DifTreatment = 
                     search.Data$ExpVer, LDA = m2.pred$x)  
# means of groups 
# means of groups 
h1.TypeMus = ddply(h1.Data, "SearchType", summarise, LD1.mean = mean(LDA.LD1))
h1.DifMus = ddply(h1.Data, "DifTreatment", summarise, LD1.mean = mean(LDA.LD1))
h2.TypeMus = ddply(h1.Data, "SearchType", summarise, LD2.mean = mean(LDA.LD2))
h2.DifMus = ddply(h1.Data, "DifTreatment", summarise, LD2.mean = mean(LDA.LD2))

# save search type histogram as jpeg image
jpeg(filename = "hist1.jpg", units = "in", height = 6, width = 10, res  = 300)  
ggplot(data = h1.Data, aes(x = LDA.LD1, fill = SearchType)) + 
  # histogram
  geom_histogram(alpha = 0.4, position = "identity", breaks = seq(-5, 8, 0.5)) +  
  # mean lines
  geom_vline(data = h1.TypeMus, aes(xintercept = LD1.mean, 
             colour = SearchType), linetype = "dashed") +  
  labs(x = "LD1 Score", y = "Count") + # axes labels and plot titles
  ggtitle("LD1 Score by Search Type: TP, FP, TN, FN", 
          subtitle = "LDA estimated with 4 Classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
dev.off()


# save search type histogram as jpeg image
jpeg(filename = "hist2.jpg", units = "in", height = 6, width = 10, res  = 300)  
ggplot(data = h1.Data, aes(x = LDA.LD2, fill = SearchType)) + 
  # histogram
  geom_histogram(alpha = 0.4, position = "identity", breaks = seq(-5, 8, 0.5)) +  
  # mean lines
  geom_vline(data = h2.TypeMus, aes(xintercept = LD2.mean, 
                                    colour = SearchType), linetype = "dashed") +  
  labs(x = "LD2 Score", y = "Count") + # axes labels and plot titles
  ggtitle("LD2 Score by Search Type: TP, FP, TN, FN", 
          subtitle = "LDA estimated with 4 Classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
dev.off()

#save difficulty treatment histogram as jpeg image
jpeg(filename = "hist3.jpg", units = "in", height = 6, width = 10, res  = 300)  
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
dev.off()

#save difficulty treatment histogram as jpeg image
jpeg(filename = "hist4.jpg", units = "in", height = 6, width = 10, res  = 300)  
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
dev.off()

# show division of ld1 and ld2 scores across classes as cluster plots

# data frames for cluster plots
cluster1.Data <- data.frame(SearchType = search.Data$search.Class, 
                            DifTreatment = search.Data$ExpVer, LDA = m2.pred$x)
# cluster by search type, save as jpeg image
jpeg(filename = "clust1.jpg", units = "in", height = 6, width = 8, res  = 300) 
# cluster by search type
ggplot(cluster1.Data) + 
  geom_point(aes(LDA.LD1, LDA.LD2,colour = SearchType, shape = SearchType),
             size = 2.0) + 
  labs(x = paste0("LD1 (", percent(prop.VarExp1[1]), ")"), y = 
       paste0("LD2 (", percent(prop.VarExp1[2]), ")")) +
  ggtitle("Search Types in LD Space: TP, FP, TN, FN", 
          subtitle = "LDA Constructed with 4 Classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
dev.off()

# cluster by treatment difficulty, save as jpeg image
jpeg(filename = "clust2.jpg", units = "in", height = 6, width = 8, res  = 300) 
# cluster by treatment difficulty
ggplot(cluster1.Data) + 
  geom_point(aes(LDA.LD1, LDA.LD2,colour = DifTreatment, shape = DifTreatment), 
             size = 2.0) + 
  labs(x = paste0("LD1 (", percent(prop.VarExp1[1]), ")"), y = 
       paste0("LD2 (", percent(prop.VarExp1[2]), ")")) +
  ggtitle("Experimental Treatments in LD Space: Easy vs Hard", 
          subtitle = "LDA Constructed with 4 Classes (Search Types)") +
  theme(plot.title = element_text(hjust = 0.5, size = 16), 
        plot.subtitle = element_text(hjust = 0.5, size = 12))
dev.off()

# ----- Examine LD Coefficients and Orginal variable loadings -----

#look at the original coefficients
original.Coefficients = data.frame(m2.lda$scaling)
print(original.Coefficients)

#Correlate the original variable values with the LD "scores" to determine 
#how each original variable loads onto each LD function
search.Data.4cor = data.frame(NumSamplingTrips = search.Data$NumSamplingTrips,
                              NumFixComp = search.Data$NumFixComp, 
                              NumGazeComp = search.Data$NumGazeComp, 
                              TotNumFix = search.Data$TotNumFix, 
                              AvgDurFix = search.Data$AvgDurFix, 
                              Fix2End = search.Data$Fix2End, 
                              ActiveSearchTime = search.Data$ActiveSearchTime)
loadings.Components = cor(search.Data.4cor, m2.pred$x) #correlate
print(loadings.Components)

closeAllConnections()

# clean up
detach(package:scales)
detach(package:MASS)
detach(package:readxl)
detach(package:ggplot2)
detach(package:plyr)
