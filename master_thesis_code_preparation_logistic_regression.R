"
The following code is the first part of the code that constitutes this 
analysis. This first part deals with data preparation and exploration aswell
as the base linear regression.
The second part of the code is a Python-script containing the predictions using
machine-learning methods.

As there is an R-project-file, all paths are relative to the path the 
project-file is in.
"


# Section 1: Set up ------------------------------------------------------------
# clean up environment
rm(list = ls())


# load packages
library(readxl)
library(dplyr)
library(usefun)
library(tidyr)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(plyr)
library(sjPlot)
library(gtsummary)
library(gt)
library(ghibli)
library(tvthemes)
library(Amelia)
library(stringr)
library(caret)
library(ROSE)
library(lme4)
library(stargazer)
library(ModelMetrics)


# packages used for data exploration (not needed after exploration)
# library(klaR)
# library(cluster)
# library(poLCA)
# library(mltools)
# library(data.table)
# library(ggdendro)
# library(infotheo)
# library(FactoMineR)
# library(factoextra)
# library(ConfusionTableR)
# library(car)
# library(pscl)
# library(aod)
# library(ROCR)
# library(glmnet)
# library(glmmLasso)
# library(psych)
# library(GPArotation)
# library(nFactors)
# library(nnet)
# library(reshape)
# library(randomForest)
# library(party)



# load in data
data <- read_excel("data/Data_CarbonNeutral_compact.xlsx") 

# check number of participants
print(paste0("number of particpants: ", length(unique(data$lfdn))))



# Section 2: Data preparation : Part 1------------------------------------------

# filter for relevant variables
# create a list of variables that are relevant for the analysis
var_list <- c("lfdn", "purchase_CHO", "purchase_PA", "purchase_GB",
              "GB_S1", "GB_S2", "GB_S3", "GB_S4", "GB_S5", "GB_S6", "GB_S7", 
              "GB_S8", "GB_S9", "PA_S1", "PA_S2", "PA_S3", "PA_S4", "PA_S5", 
              "PA_S6", "PA_S7", "PA_S8", "PA_S9", "CHO_S1", "CHO_S2", 
              "CHO_S3", "CHO_S4", "CHO_S5", "CHO_S6", "CHO_S7", "CHO_S8", 
              "CHO_S9", "rank_GB", "rank_CHO", "rank_CHEESE", "rank_WHEAT",
              "rank_TOM", "age", "gender", "country", "education", 
              "occupation", "av_income", "environment", "Country_Name",
              "Generation", "final_comment", "browser", "datetime",
              "date_of_last_access", "date_of_first_mail", "Date_Start",
              "Date_End", "Time_Start", "Time_End", "Completion_Time")

# subset dataframe
data_clean <- data[,var_list]

# check how many variables are left
print(paste0("number of variables in the data: ",dim(data)[2], 
             ", number variables for further steps of the analysis: ", 
             dim(data_clean)[2]))



# recode missing values

# replace non-answers with NA's
data_clean[data_clean == -66 | data_clean == -77 | data_clean == -99] <- NA



# select observations that worked on the questionnaire for a reasonable time

# function to calculate completion times, technically they are in the given
# data set, but don't account for transitions between dates:

time_passed  <- function(dataset){
  # initiate column for completion time in seconds
  dataset$Completion_Time_sec = rep(NA, dim(dataset)[1])
  
  # calculate difference between Time_End and Time_Start in seconds, add seconds
  # for each day passed between start and finish time
  for (i in 1:dim(dataset)[1]){
    dataset$Completion_Time_sec = (dataset$Time_End - dataset$Time_Start) +
      (dataset$Date_End - dataset$Date_Start)
  }
  
  # make sure the column is numeric
  dataset$Completion_Time_sec <- as.numeric(dataset$Completion_Time_sec)
  
  return(dataset)
}


# apply function
data_clean <- time_passed(data_clean)


# check results
print(paste0("minimum time taken: ", min(data_clean$Completion_Time_sec),
             " seconds"))



# pre-select observations based on completion time
# long completion times will not be excluded, as participants might have been
# distracted but faithfully finished the questionnaire later
# very short completion times will be excluded, as participants might have not
# payed attention
# the approach might be greedy, as data-hungry methods incentivice keeping as
# many observations as possible, the author accounted for this by deep 
# philosophical contemplation of one's motives and biases... and a coin flip

# check dimension before filtering
print(paste0("number of observations before filtering: ", dim(data_clean)[1]))

# filter data set
data_clean <- data_clean[data_clean$Completion_Time_sec >= 120,]
# minimum time: 120 seconds, which seems reasonable when accounting for
# different questionnaire constellations, reading speeds, and decision making

# check dimensions after filtering
print(paste0("number of observations after filtering: ", dim(data_clean)[1]))



# factorize and label observations

# gender
data_clean$gender <- factor(data_clean$gender,
                            levels = c(1, 2, 5, 6),
                            labels = c("Female", "Male", "Non-binary",
                                       "Other"))


# education
data_clean$education <- factor(data_clean$education,
                               levels = c(1, 2, 3, 4, 5, 6, 7),
                               labels = c("Secondary School General", 
                                          "Secondary School", 
                                          "Secondary School Academic", 
                                          "Bachelor's Degree", 
                                          "Master's Degree", 
                                          "Ph.D. or higher", 
                                          "None"))


# occupation
data_clean$occupation <- 
  factor(data_clean$occupation, 
         levels = c(1, 2, 3, 4, 5, 6, 7), 
         labels = c("Student Highschool", 
                    "Student University", 
                    "In education or vocational training", 
                    "Employed", 
                    "Retired", 
                    "Looking for employment", 
                    "Other"))


# maybe factorize environment?
data_clean$environment <- factor(data_clean$environment,
                                 levels = c(1, 2),
                                 labels = c("Priority", "Not priority"))


# factorize Generation if the variable is used additionally to/instead of age
# might make sense because of the unequal age distribution
data_clean$Generation <- as.factor(data_clean$Generation)



# maybe factorize average income? ...as the possible answers were categories
# make new variable for that for now
data_clean$av_income_factorized <- factor(data_clean$av_income,
                                          levels = c(1, 2, 3, 4, 5, 6),
                                          labels = c("<1000", 
                                                     "1000-1500", 
                                                     "1500-3500", 
                                                     "3500-5000", 
                                                     ">5000", 
                                                     "No answer"))  
# this does not have clear distinctions between levels, e.g. "1500" could 
# belong to the second or the third level, depending on interpretation


data_clean$purchase_CHO <- factor(data_clean$purchase_CHO,
                                  levels = c(0, 1),
                                  labels = c("No",
                                             "Yes"))


data_clean$purchase_PA <- factor(data_clean$purchase_PA,
                                 levels = c(0, 1),
                                 labels = c("No",
                                            "Yes"))


data_clean$purchase_GB <- factor(data_clean$purchase_GB,
                                 levels = c(0, 1),
                                 labels = c("No",
                                            "Yes"))



# recoding outcome variables to get them into the right format

# recoding outcome so choosing the product with the eco-label always has the
# same coding
recode_vars <- function(dataset_choice, variables){
  dataset_choice <- dataset_choice %>%
    mutate_at(variables, funs(recode(., '1' = 2, '2' = 1, '3' = 3, 
                                     .default = NaN)))
  return(dataset_choice)
}


# copy of dataset
data_recoded <- data_clean


# check distribution of one variable for comparison
print("CHO_S3 before recoding:")
print(table(data_recoded$CHO_S3))
print("")



# recode dataset
data_recoded <- recode_vars(data_recoded, c("CHO_S3", "CHO_S5", "CHO_S7", "CHO_S8", 
                                            "CHO_S9", "PA_S3", "PA_S5", "PA_S7", 
                                            "PA_S8", "PA_S9", "GB_S3", "GB_S5", 
                                            "GB_S7", "GB_S8", "GB_S9"))


# check distribution for one variable after recoding
print("CHO_S3 after recoding:")
print(table(data_recoded$CHO_S3))




# Section 3: Figures------------------------------------------------------------


# Visualization 1: world map

# show map of where the answers are from (Costa Rica should be as visible as
# possible, since there were relatively a lot of observations from there, but
# it's a small land)
print(unique(data_recoded$country))
print(unique(data_recoded$Country_Name))


# get dataframe with countries and number of participants
participants_origin <- count(data_recoded, "Country_Name")
participants_origin

# adjust country names to the ones ggplot uses
participants_origin$Country_Name[participants_origin$Country_Name == "United Kingdom"] <- "UK"
participants_origin$Country_Name[participants_origin$Country_Name == "United States"] <- "USA"


# filter world data
world_data <- map_data("world") %>% filter(region != "Antarctica") %>% fortify
# no one from Antarctica participated... I'm shocked as well

# print and save map
png("figures/world_map.png", width = 1000, height = 600)
map_world <- ggplot() +
  geom_map(data = world_data, map = world_data,
           aes(long, lat, map_id = region), fill = "white", colour = "black",
           size = 0.5) +
  geom_map(data = participants_origin, map = world_data,
           aes(fill = freq, map_id = Country_Name), colour = "black") +
  scale_fill_continuous(low = "cornflowerblue", high = "midnightblue") +
  xlab("longitudinal") + ylab("lateral") +
  labs(fill = "Count") +
  guides(fill = "none") +
  theme_bw()
print(map_world)
dev.off()
print(map_world)


# countries for European map (not only European countries)
europe_country_list = c("Albania, Andorra", "Austria", "Belarus", "Belgium", 
                        "Bosnia and Herzegovina", "Bulgaria", "Croatia", 
                        "Czech Republic", "Denmark", "Estonia", "Finland", 
                        "France", "Germany", "Greece", "Hungary", "Iceland",
                        "Italy", "Latvia", "Liechtenstein", "Lithuania", 
                        "Luxembourg", "Malta", "Moldova", "Monaco", 
                        "Montenegro", "Netherlands", "North Macedonia", 
                        "Norway", "Poland", "Portugal", "Romania", "Russia", 
                        "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", 
                        "Sweden", "Switzerland", "Ukraine", "UK", 
                        "Turkey", "Cyprus", "Syria", "Lebanon", "Israel", 
                        "Jordan", "Morocco", "Tunesia", "Lybia", "Algeria", 
                        "Egypt", "Georgia", "Azerbaijan", "Iraq")
europe <- world_data[world_data$region == europe_country_list,]


# print map
png("figures/map_europe.png", width = 1000, height = 600)
map_europe <- ggplot() +
  geom_map(data = europe, map = world_data,
           aes(long, lat, map_id = region), fill = "white", colour = "black",
           size = 0.5) + 
  geom_map(data = participants_origin, map = world_data, 
           aes(fill = freq, map_id = Country_Name), colour = "black") +
  scale_fill_continuous(low = "cornflowerblue", high = "midnightblue") +
  xlab("longitudinal") + ylab("lateral") +
  labs(fill = "Count") +
  guides(fill = "none") +
  theme_bw() + xlim(-10,40) + ylim(35,70)
print(map_europe)
dev.off()
print(map_europe)


# countries for north- and middle-american map (not only north-and middle-
# american countries)
america_nm_country_list <- c("Canada", "Greendland", "Mexico", "USA",
                             "Antigua", "Bahamas", "Barbados", "Cuba", 
                             "Dominica", "The Dominican Republic", "Grenada", 
                             "Haiti", "Saint Kitts", "Saint Lucia", 
                             "Saint Vincent", "Trinidad",
                             "Costa Rica", "El Salvador", "Guatemala", 
                             "Honduras", "Nicaragua", "Panama",
                             "Colombia", "Venezuela", "Guyana", "Suriname")
america_nm <- world_data[world_data$region == america_nm_country_list,]
# it is also doable with only zooming in and out, but that would require more
# trial and error with the limits

# print map
png("figures/map_north_america.png", width = 1000, height = 600)
map_america_nm <- ggplot() +
  geom_map(data = america_nm, map = world_data,
           aes(long, lat, map_id = region), fill = "white", colour = "black",
           size = 0.5) + 
  geom_map(data = participants_origin, map = world_data, 
           aes(fill = freq, map_id = Country_Name), colour = "black") +
  scale_fill_continuous(low = "cornflowerblue", high = "midnightblue") +
  xlab("longitudinal") + ylab("lateral") +
  labs(fill = "Count") +
  guides(fill = "none") +
  theme_bw() + xlim(-175,-25) + ylim(10, 80)
print(map_america_nm)
dev.off()
print(map_america_nm)


# countries for south-american map (not only south-american countries)
america_s_country_list <- c("Argentina", "Bolivia", "Brazil", "Chile", 
                            "Colombia", "Eucador", "Guyana", "Paraguay", "Peru", 
                            "Suriname", "Uruguay", "Venezuela",
                            "Mexico", "Costa Rica", "El Salvador", "Guatemala", 
                            "Honduras", "Nicaragua", "Panama")
america_s <- world_data[world_data$region == america_s_country_list,]

# print map
png("figures/map_south_america.png", width = 1000, height = 600)
map_america_s <- ggplot() +
  geom_map(data = america_s, map = world_data,
           aes(long, lat, map_id = region), fill = "white", colour = "black",
           size = 0.5) + 
  geom_map(data = participants_origin, map = world_data, 
           aes(fill = freq, map_id = Country_Name), colour = "black") +
  scale_fill_continuous(low = "cornflowerblue", high = "midnightblue") +
  xlab("longitudinal") + ylab("lateral") +
  labs(fill = "Count") +
  guides(fill = "none") +
  theme_bw() + xlim(-105, -20) + ylim(-55,15)
print(map_america_s)
dev.off()
print(map_america_s)



# visualization 2: table with demographics
# create table one
table_one <- tbl_summary(data_recoded,
                         include = c(age, gender, education, occupation,
                                     av_income_factorized),
                         statistic = c(age ~ "{median} ({sd})"),
                         label = c(age ~ "Age, median (SD)",
                                   gender ~ "Gender, n (%)",
                                   education ~ "Education, n (%)",
                                   occupation ~ "Occupation, n (%)",
                                   av_income_factorized ~ "Income, n(%)"),
                         missing = "no") %>% 
  modify_header(label = "**Demographics (missing values excluded)**") %>%
  bold_labels()
print(table_one)

# save table one
table_one %>%
  as_gt() %>%
  gtsave("figures/table_one.png")




# visualization 3: Demographics

# check min and max ages
print(paste0("min age: ", min(data_recoded$age, na.rm = TRUE)))
print(paste0("max age: ", max(data_recoded$age, na.rm = TRUE)))



# histogram of age distributions overall
age_histogram <- ggplot(data_recoded, aes(x = age)) +
  geom_histogram(
    aes(y = ..density..), 
    color = "black", 
    fill = "cornflowerblue") +
  geom_density(alpha = 0.5, fill = "plum3") +
  xlab("Age") + ylab("Density") +
  #ggtitle("Age density") +
  scale_x_continuous(breaks = seq(0, 100, 10), 
                     limits = c(round_any(min(data_recoded$age, na.rm = TRUE), 
                                          10, f = floor) - 10, 
                                round_any(max(data_recoded$age, na.rm = TRUE), 
                                          10, f = ceiling) + 10)) +
  scale_y_continuous(breaks = seq(0.0, 0.1, 0.01), limits = c(0.0, 0.07)) +
  labs(caption = "There is some concentration  \naround 25 years and 55 years,
       one has to take this into consideration.") +
  theme_bw() +
  theme(text = element_text(size = 30)) 



# histograms of age by gender distributions
png("figures/gender_histogram.png", width = 1000, height = 700)
gender_histogram <- ggplot(data_recoded,
                           aes(x = age, fill = gender)) +
  geom_histogram(alpha = 1, 
                 position = "stack", 
                 color = "black") +
  xlab("Age") + ylab("Count") +
  #ggtitle("Age distribution by gender") +
  scale_x_continuous(breaks = seq(0, 100, 10), 
                     limits = c(round_any(min(data_recoded$age, na.rm = TRUE), 
                                          10, f = floor) - 10, 
                                round_any(max(data_recoded$age, na.rm = TRUE), 
                                          10, f = ceiling) + 10)) +
  guides(fill = guide_legend(title = "Gender")) +
  theme_bw() +
  theme(text = element_text(size = 30)) +
  scale_fill_ghibli_d("LaputaMedium", direction = -1) 

print(gender_histogram)
dev.off()
print(gender_histogram)


# age-distribution for most prevalant countries in the data
png("figures/country_histogram.png", width = 1000, height = 700)
country_histogram <- ggplot(data_recoded %>% 
                              filter(Country_Name %in% 
                                       c("Germany", "Costa Rica")), 
                            aes(x = age, fill = Country_Name)) +
  geom_histogram(alpha = 0.5,
                 position = "identity",
                 color = "black") +
  xlab("Age") + ylab("Count") +
  #ggtitle("Age distribution by most prevalent countries in the data") +
  scale_x_continuous(breaks = seq(0, 100, 10), 
                     limits = c(round_any(min(data_recoded$age, na.rm = TRUE), 
                                          10, f = floor) - 10, 
                                round_any(max(data_recoded$age, na.rm = TRUE), 
                                          10, f = ceiling) + 10)) +
  guides(fill = guide_legend(title = "Country")) +
  scale_fill_manual(values = c(ghibli_palettes$MononokeMedium[3], 
                               ghibli_palettes$MononokeMedium[6])) +
  theme_bw() +
  theme(text = element_text(size = 30)) 

print(country_histogram)
dev.off()
print(country_histogram)


# Violin-plot for gender
png("figures/age_gender_violin.png", width = 1000, height = 700)
age_gender_violin <- ggplot(data_recoded %>% 
                              filter(gender %in% 
                                       c("Female", "Male", "Non-binary")), 
                            aes(x = gender, y = age)) +
  geom_violin(fill = ghibli_palettes$LaputaMedium[6], size = 1) + 
  labs(title = "Age distribution for different genders",
       subtitle = "(too few observations for 'Other')") +
  xlab("Gender") + ylab("Age") +
  theme_bw() +
  theme(text = element_text(size = 30)) 

print(age_gender_violin)
dev.off()
print(age_gender_violin)


# density-plot of the relationship of age and gender
png("figures/age_gender_density.png", width = 1000, height = 700)
age_gender_denstiy <- ggplot(data_recoded %>% 
                               filter(gender %in% 
                                        c("Female", "Male", "Non-binary")),
                             aes(x = age, color = gender)) +
  geom_density(size = 2) +
  labs(title = "Age distribution by gender",
       subtitle = "(too few observations for 'Other')",
       color = "Gender") +
  scale_color_ghibli_d("LaputaMedium", direction = -1) +
  theme_bw() +
  theme(text = element_text(size = 30))

print(age_gender_denstiy)
dev.off()
print(age_gender_denstiy)


# contingency-table education and occupation

# create contingency-table
edu_occ <- data.frame(table(data_recoded$education, data_recoded$occupation))

# graphic
png("figures/edu_occ_balloon.png", width = 800, height = 500)
edu_occ_balloon <- ggballoonplot(edu_occ, fill = "Freq", ggtheme = theme_bw(),
                                 size = "Freq",
                                 shape = 21) +
  gradient_fill(c("lightskyblue1", 
                  "navyblue")) +
  #ggtitle("Co-occurence of education and occupation \ncategories") +
  theme(text = element_text(size = 20)) 

print(edu_occ_balloon)
dev.off()
print(edu_occ_balloon)


# distribution-plots for the relationship of age and income
png("figures/age_income_density.png", width = 1000, height = 700)
age_income_density <- ggplot(data_recoded %>% filter(av_income_factorized %in%
                                                       c("<1000", "1000-1500", "1500-3500",
                                                         "3500-5000", ">5000", "No answer")), 
                             aes(x = age, y = av_income_factorized)) +
  geom_density_ridges(aes(fill = av_income_factorized), alpha = 0.5) +
  scale_fill_ghibli_d("MononokeMedium", direction = -1) +
  #ggtitle("Income distribution by age group") +
  xlab("Age") + ylab("Income") +
  labs(fill = "Income category") +
  theme_bw() +
  theme(text = element_text(size = 30))

print(age_income_density)
dev.off()
print(age_income_density)


# barplot of environmental views
png("figures/environment_bar.png", width = 1000, height = 700)
environment_bar <- ggplot(data_recoded %>% filter(environment %in%
                                                    c("Priority", "Not priority")), 
                          aes(x = environment)) +
  geom_bar(fill = ghibli_palettes$LaputaMedium[6]) +
  #ggtitle("View on environmental issues") +
  xlab("") + ylab("Count") +
  theme_minimal() +
  theme(text = element_text(size = 30))

print(environment_bar)
dev.off()
print(environment_bar)


# bosplot of environmental views
png("figures/environment_boxplot.png", width = 700, height = 700)
environment_boxplot <- ggplot(data_recoded %>% filter(environment %in% 
                                                        c("Priority", "Not priority")), 
                              aes(x = environment, y = age)) +
  geom_boxplot(color = "black") +
  geom_jitter(width = 0.2, size = 5, color = ghibli_palettes$LaputaMedium[6]) +
  xlab("") + ylab("Age") +
  ylim(0, 100) +
  #ggtitle("View on environmental issues by age") +
  theme_minimal() +
  theme(text = element_text(size = 30))

print(environment_boxplot)
dev.off()
print(environment_boxplot)



# Section 4: Data preparation - Part 2------------------------------------------

# check country-variable
# frequency
print(table(data_recoded$Country_Name))

# a lot of countries do not have many observations, let's recode them to a 
# different value
change_label_countries <- unique(data_recoded$Country_Name)
change_label_countries <- change_label_countries[!(change_label_countries %in%
                                                     c("Costa Rica", "Germany",
                                                       NA))]
data_recoded$Country_Name[data_recoded$Country_Name %in% 
                            change_label_countries] <- "Other"

# check results
table(data_recoded$Country_Name)

# factorize
data_recoded$Country_Name <- as.factor(data_recoded$Country_Name)



# Interlude: Cluster and factor analyses for data exploration-------------------

# Analyses commented out as they were used for exploration and did not appear in
# the paper

# requires outcommented packages from the set-up-section

# # kmodes clustering for categorical data
# data_cluster <- data_recoded[, c("age", "gender", "Country_Name", "education",
#                                  "occupation", "av_income_factorized",
#                                  "environment")]
# data_cluster <- as.data.frame(na.omit(sapply(data_cluster, as.numeric)))
# 
# cluster_explore <- kmodes(data_cluster, 5)
# 
# data_clustered <- data_cluster
# data_clustered$cluster <- cluster_explore$cluster
# head(data_clustered)
# 
# 
# 
# # helper function to get percentages from table()
# table_perc <- function(column){
#   table_output <- table(column)
#   return(round(table_output / sum(table_output), 2))
# }
# 
# 
# 
# # check whether some clusters can be found
# for (i in 1:length(unique(data_clustered$cluster))){
#   print(paste0("cluster: ", i))
#   data_temp <- data_clustered[data_clustered$cluster == i,]
#   data_temp <- data_temp %>% dplyr::select(gender, education, occupation,
#                                            av_income_factorized, environment, 
#                                            cluster)
#   print(sapply(data_temp[, -which(names(data_temp) %in% c("cluster"))], 
#                table_perc))
#   print("num obs:")
#   print(table(data_temp$cluster))
#   for(i in 1:3){
#     print_empty_line()
#   }
#   print("--------------------------------------------------------------------")
# }
# 
# 
# 
# # hierarchical clustering
# 
# set.seed(100)
# 
# gower_distance <- daisy(data_cluster[, c("age", "gender", "education", 
#                                          "occupation", "av_income_factorized", 
#                                          "environment")], 
#                         metric = c("gower"))
# 
# 
# # divisive clustering
# divisive_clustering <- diana(as.matrix(gower_distance),
#                              diss = TRUE,
#                              keep.diss = TRUE)
# plot(divisive_clustering, main = "Divisive")
# 
# 
# # agglomerative clustering
# agglomorative_clustering <- hclust(gower_distance, method = "complete")
# plot(agglomorative_clustering, main = "Agglomerative")
# 
# 
# 
# # latent class analysis
# formula_1 <- cbind(gender, education, occupation, av_income_factorized, 
#                    environment) ~ 1
# 
# lca_1 <- poLCA(formula_1, data_cluster, nclass = 3,
#                maxiter = 10000)
# print(lca_1)
# 
# plot(lca_1)
# 
# 
# 
# # cluster analysis using a dendrogram
# data_cluster <- data_recoded[, c("age", "gender", "Country_Name", "education",
#                                  "occupation", "av_income_factorized")]
# 
# 
# # one-hot-encode data for cluster normalization
# data_cluster_dummy <- data_cluster %>% select_if(is.factor)
# 
# 
# data_cluster_dummy_one_hot <- one_hot(as.data.table(data_cluster_dummy))
# 
# 
# # put dataset together again (a.k.a playing Frankenstein)
# data_cluster_one_hot <- cbind(data_cluster %>% select_if(is.numeric),
#                               data_cluster_dummy_one_hot)
# 
# 
# # normalize data for cluster analysis
# data_cluster_one_hot_means <- apply(data_cluster_one_hot, 2, mean, na.rm = TRUE)
# data_cluster_one_hot_sds <- apply(data_cluster_one_hot, 2, sd, na.rm = TRUE)
# data_cluster_normalized <- scale(data_cluster_one_hot, 
#                                  center = data_cluster_one_hot_means,
#                                  scale = data_cluster_one_hot_sds)
# 
# 
# # get rid of NA's
# data_cluster_normalized <- na.omit(data_cluster_normalized)
# 
# 
# # calculate distances
# data_cluster_distances <- dist(data_cluster_normalized)
# 
# 
# # hierarchical clustering
# data_cluster_clusters <- hclust(data_cluster_distances)
# 
# # visualize results
# plot(data_cluster_clusters)
# ## too many clusters, maybe change approach here
# 
# 
# ggdendrogram(data_cluster_clusters)
# 
# 
# 
# # check relationships of the data
# 
# # chi-squared
# data_categorical <- data_cluster[,c("gender", "Country_Name", "education", 
#                                     "occupation", "av_income_factorized")]
# 
# 
# # create function to get values for the chi-square test statistic
# create_chi_sqaured_matrix <- function(dataframe){
#   column_names <- colnames(dataframe)
#   n <- length(column_names)
#   chi_squared_matrix <- matrix(0, nrow = n, ncol = n)
#   p_matrix <- matrix(0, nrow = n, ncol = n)
#   rownames(chi_squared_matrix) <- column_names
#   colnames(chi_squared_matrix) <- column_names
#   rownames(p_matrix) <- column_names
#   rownames(p_matrix) <- column_names
#   
#   for (i in 1:n){
#     for (j in i:n){
#       if(i == j){
#         chi_squared_matrix[i, j] <- NA
#         p_matrix[i, j] <- NA
#       } else {
#         contingency_table <- table(dataframe[[i]],
#                                    dataframe[[j]])
#         test <- chisq.test(contingency_table)
#         chi_squared_matrix[i, j] <- test$statistic
#         chi_squared_matrix[j, i] <- test$statistic
#         p_matrix[i, j] <- test$p.value
#         p_matrix[j, i] <- test$p.value
#       }
#     }
#   }
#   
#   list(chi_squared_matrix = chi_squared_matrix,
#        p_matrix = p_matrix)
# }
# 
# 
# # get results
# chi_squared_1 <- create_chi_sqaured_matrix(data_categorical)
# 
# # check results
# print(chi_squared_1$chi_squared_matrix)
# print(chi_squared_1$p_matrix)
# 
# 
# 
# # mutual information and heatmap
# 
# # function to calculate mutual informatio matrix of a dataframe
# calculate_mutual_info_matrix <- function(dataframe){
#   column_names <- colnames(dataframe)
#   n <- length(column_names)
#   mi_matrix <- matrix(0, nrow = n, ncol = n)
#   rownames(mi_matrix) <- column_names
#   colnames(mi_matrix) <- column_names
#   
#   for (i in 1:n){
#     for (j in i:n){
#       if (i == j){
#         mi_matrix[i, j] <- 1
#       } else {
#         mi <- mutinformation(dataframe[[i]], dataframe[[j]])
#         mi_matrix[i, j] <- mi
#         mi_matrix[j, i] <- mi
#       }
#     }
#   }
#   
#   return(mi_matrix)
# }
# 
# # calculate information
# mi_1 <- calculate_mutual_info_matrix(data_categorical)
# print(mi_1)
# 
# # plot heatmap of mutual information values
# heatmap(mi_1)
# 
# 
# 
# # Multiple Correspondence Analysis 
# # (PCA-equivalent for categorical variables)
# 
# data_mca <- data_recoded[c("gender", "education",
#                            "occupation", 
#                            "environment")]
# 
# 
# categories <- apply(data_mca, 2, function(x) nlevels(as.factor(x)))
# categories
# 
# 
# mca_1 <- MCA(na.omit(data_mca))
# 
# 
# mca_1$eig
# 
# 
# mca_1_vars_data <- data.frame(mca_1$var$coord, Variable = rep(names(categories), categories))
# mca_1_obs_data <- data.frame(mca_1$ind$coord)
# 
# ggplot(data = mca_1_vars_data, aes(x = Dim.1, y = Dim.2, label = rownames(mca_1_vars_data))) +
#   geom_hline(yintercept = 0, colour = "violetred1") +
#   geom_vline(xintercept = 0, colour = "violetred1") +
#   geom_text(aes(colour = Variable)) +
#   ggtitle("MCA plot") +
#   theme_minimal()
# 
# 
# # plot categories and observations
# ggplot(data = mca_1_obs_data, aes(x = Dim.1, y = Dim.2)) +
#   geom_hline(yintercept = 0, colour = "violetred1") +
#   geom_vline(xintercept = 0, colour = "violetred1") +
#   geom_point(colour = "cornflowerblue", alpha = 0.8) +
#   geom_density_2d(colour = "cornflowerblue") +
#   geom_text(data = mca_1_vars_data,
#             aes(x = Dim.1, y = Dim.2, label = rownames(mca_1_vars_data), colour = Variable)) +
#   ggtitle("MCA plot") +
#   scale_colour_discrete(name = "Variable") +
#   theme_minimal()
# 
# 
# 
# # factor analysis of mixed data
# data_fa <- data_recoded[c("gender", "education",
#                           "occupation", 
#                           "environment", "Country_Name", "age")]
# 
# 
# fa_md <- FAMD(na.omit(data_fa), graph = FALSE)
# print(fa_md)
# 
# 
# fa_md_eigenvalues <- get_eigenvalue(fa_md)
# head(fa_md_eigenvalues)
# 
# 
# fviz_screeplot(fa_md)
# 
# 
# fa_md_var <- get_famd_var(fa_md)
# head(fa_md_var$coord)
# head(fa_md_var$cos2)
# head(fa_md_var$contrib)
# 
# 
# fviz_famd_var(fa_md, repel = TRUE)
# fviz_contrib(fa_md, "var", axes = 1)
# fviz_contrib(fa_md, "var", axes = 2)



# Section 5: Data preparation - Part 5------------------------------------------

# transform data to long format for panel data logit regression

# transform data to long format
data_lf <- data_recoded %>% gather(GB_S1, GB_S2, GB_S3, GB_S4, GB_S5, GB_S6,
                                   GB_S7, GB_S8, GB_S9, PA_S1, PA_S2, PA_S3,
                                   PA_S4, PA_S5, PA_S6, PA_S7, PA_S8, PA_S9,
                                   CHO_S1, CHO_S2, CHO_S3, CHO_S4, CHO_S5,
                                   CHO_S6, CHO_S7, CHO_S8, CHO_S9,
                                   key = "choice", value = "decision")


# check results
print(paste0("Expected number of rows: ", 
             length(unique(data_recoded$lfdn)) * (3*9)))
print(paste0("Actual number of rows: ", nrow(data_lf)))



# factorize variables that should be factorized but are not factorized yet
data_lf$decision <- as.factor(data_lf$decision)
data_lf$country <- as.factor(data_lf$country)
print_empty_line()
print("Datatypes:")
print(sapply(data_lf, class))



# deal with missing values

# visualize frequency of missing values
missmap(data_lf)


# function to get NA's of a dataframe for each variable
na_perc <- function(dataset){
  na_s <- data.frame(colnames(dataset))
  na_s["NA_freq"] <- rep(NA, dim(dataset)[2])
  for (i in 1:dim(dataset)[2]){
    na_s[i, 2] <- round(sum(is.na(dataset[,i])) / dim(dataset)[1], 4) * 100 
  }
  na_s <- na_s[order(na_s$NA_freq, decreasing = TRUE),]
  colnames(na_s) <- c("variable", "na_percantage")
  return(na_s)
}


# na-frequencies long-format dataset
na_perc(data_lf)

# final_comment has a lot of NA's as expected, which is fine in this case, as the
# variable will not be used for the model comparison.
# decision also has a lot of NA's which is not surprising as it includes all 
# decision that were by construction not made by certain participants (e.g.
# someone expressing that they do not buy chocolate will have automatically NA's
# for all choices regarding chocolate). In this case the best way to deal with
# these NA's is to throw them out either way, as they were no meant to be 
# observations in the first place. Imputation is not a good strategy here, as
# someone who does no buy chocolate could be very different from individuals who 
# do, hence, using information from other individuals for an educated guess might
# be biased.
# Other variables have little amoutn of NA's.


# naive estimate of how many of NA's overall could be traced back to decision
print(paste0("NA percantage for decision: ", 
             na_perc(data_lf %>% subset(select = decision))[2], "%"))
print(paste0("NA percantage in any row: ", 
             round(sum(!complete.cases(data_lf %>% 
                                         subset(select = -c(final_comment)))) / 
                     nrow(data_lf) * 100, 2), "%"))


# remove rest of NA's

# number of rows before removing NA's
print(paste0("number of rows before cleaning: ", nrow(data_lf)))
#data_lf <- data_lf %>% subset(select = -c(final_comment))
data_lf_test <- data_lf %>% subset(!is.na(decision))
print(paste0("number of rows, when only deleting rows that have NA's for the 
choice variable: ", nrow(data_lf_test)))
data_lf <- data_lf[complete.cases(data_lf %>% 
                                    subset(select = -c(final_comment))),]
print(paste0("number of rows, after cleaning: ", nrow(data_lf)))

na_perc(data_lf)

# Only 18 rows have NA's left when deleting the rows that have NA's for the
# decision variable, hence these rows will also be removed in this case.



# adding prices and product type as variables

# create new variables for price
data_lf$price_cn <- rep(0, nrow(data_lf))
data_lf$price_no_cn <- rep(0, nrow(data_lf))
data_lf$product_type <- rep(0, nrow(data_lf))


# add prices
# ground beef carbon neutral
data_lf$price_cn <- ifelse(data_lf$choice %in% 
                             c("GB_S1", "GB_S4", "GB_S5"), 
                           3.49,
                           ifelse(data_lf$choice %in% 
                                    c("GB_S2", "GB_S6", "GB_S8"), 
                                  4.99, 
                                  ifelse(data_lf$choice %in% 
                                           c("GB_S3", "GB_S7", "GB_S9"), 
                                         6.59,
                                         data_lf$price_cn)))
# pasta carbon neutral
data_lf$price_cn <- ifelse(data_lf$choice %in% 
                             c("PA_S1", "PA_S4", "PA_S5"), 
                           1.09,
                           ifelse(data_lf$choice %in% 
                                    c("PA_S2", "PA_S6", "PA_S8"), 
                                  1.79, 
                                  ifelse(data_lf$choice %in%
                                           c("PA_S3", "PA_S7", "PA_S9"), 
                                         2.49,
                                         data_lf$price_cn)))
# chocolate carbon neutral
data_lf$price_cn <- ifelse(data_lf$choice %in% 
                             c("CHO_S1", "CHO_S4", "CHO_S5"), 
                           0.79,
                           ifelse(data_lf$choice %in% 
                                    c("CHO_S2", "CHO_S6", "CHO_S8"), 1.29, 
                                  ifelse(data_lf$choice %in%
                                           c("CHO_S3", "CHO_S7", "CHO_S9"), 
                                         1.79,
                                         data_lf$price_cn)))

# ground beef not carbon neutral
data_lf$price_no_cn <- ifelse(data_lf$choice %in% 
                                c("GB_S1", "GB_S2", "GB_S3"), 
                              3.49,
                              ifelse(data_lf$choice %in% 
                                       c("GB_S4", "GB_S6", "GB_S7"), 
                                     4.99, 
                                     ifelse(data_lf$choice %in%
                                              c("GB_S5", "GB_S8", "GB_S9"),
                                            6.59,
                                            data_lf$price_no_cn)))
# pasta not carbon neutral
data_lf$price_no_cn <- ifelse(data_lf$choice %in% 
                                c("PA_S1", "PA_S2", "PA_S3"), 
                              1.09,
                              ifelse(data_lf$choice %in% 
                                       c("PA_S4", "PA_S6", "PA_S7"), 
                                     1.79, 
                                     ifelse(data_lf$choice %in%
                                              c("PA_S5", "PA_S8", "PA_S9"),
                                            2.49,
                                            data_lf$price_no_cn)))
# chocolate not carbon_neutral
data_lf$price_no_cn <- ifelse(data_lf$choice %in% 
                                c("CHO_S1", "CHO_S2", "CHO_S3"), 
                              0.79,
                              ifelse(data_lf$choice %in% 
                                       c("CHO_S4", "CHO_S6", "CHO_S7"), 
                                     1.29, 
                                     ifelse(data_lf$choice %in%
                                              c("CHO_S5", "CHO_S8", "CHO_S9"),
                                            1.79,
                                            data_lf$price_no_cn)))


# add product category
data_lf$product_type <- ifelse(data_lf$choice %>% 
                                 str_detect("GB") == TRUE, 
                               "Ground beef",
                               ifelse(data_lf$choice %>% 
                                        str_detect("PA") == TRUE, 
                                      "Pasta", "Chocolate"))



# only keep active decisions

# check outcome-distribution
print("outcome distribution:")
print(table(data_lf$decision, useNA = "always"))
print_empty_line()

# remove observations with outcome "neither" since there are few observations
# and multinomial logistic regression with random effects is not reliably
# available in R
print(paste0("number of observations before removing 'neither': ", 
             dim(data_lf)))
data_model <- data_lf[data_lf$decision != 3,]
print(paste0("number of observations after removing 'neither': ",
             dim(data_model)))

# recode decision-variable to 0-1
data_model$decision <- as.factor(ifelse(data_model$decision == 2, 0, 1))
print(table(data_model$decision))



# check that every combination of outcome and expression of regressors is in 
# the data
var_list <- c("gender", "education", "occupation", "av_income_factorized", 
              "price_cn", "price_no_cn", "product_type", "environment", 
              "country", "age")

for (variable in var_list){
  formula_temp <- paste0("~ decision + ", variable)
  #print(formula_temp)
  print("variable: ")
  print(xtabs(formula_temp, data = data_model))
  print_empty_line()
}
# most regressors are fine but some expressions of the country-variable are
# have one-sided outcomes, some models (e.g. logistic regression) will not
# be able to estimate probabilities for these variable-expressions
# age also does also have one-sided distributions for some ages, this should
# be fine however, since age is continuous 


# write function to get expressions with one-sided outcome-distributions
target_acquired <- function(dataset, variable){
  expression_list = c()
  formula_temp <- paste0("~decision + ", variable)
  xtab_temp <- xtabs(formula_temp, data = dataset)
  colnames_temp <- colnames(xtab_temp)
  for (name in colnames_temp){
    expression_temp <- xtab_temp[, name]
    if (0 %in% expression_temp){
      expression_list = c(expression_list, name)
    }
  }
  return(expression_list)
}

# countries
countries_one_sided <- target_acquired(data_model, "country")
print(countries_one_sided)

# ages
ages_one_sided <- target_acquired(data_model, "age")
print(ages_one_sided)


# subset data based on retrieved expressions
print(dim(data_model))
data_model <- data_model %>% subset(!(country %in% countries_one_sided))
print(dim(data_model))
# check result
nrow(data_lf) == (nrow(data_model) + 
                    nrow(data_lf[data_lf$country %in% countries_one_sided,]))
# worked

# let's also remove ages with one-sided outcome-distributions just to be sure
data_model <- data_model %>% subset(!(age %in% ages_one_sided))
print(dim(data_model))
nrow(data_lf) == (nrow(data_model) + 
                    nrow(data_lf[data_lf$country %in% countries_one_sided,]) + 
                    nrow(data_lf[data_lf$age %in% ages_one_sided,]))
# worked



# check how variables are coded (which categories are reference categories)
# helper-function
factor_check <- function(dataset){
  data_temp <- dataset[, sapply(dataset, is.factor)]
  return(data_temp)
}


check_coding <- function(dataset, variable_list){
  data_temp <- dataset[, variable_list]
  data_temp <- factor_check(data_temp)
  
  for(var in colnames(data_temp)){
    print(paste0(var, ":"))
    print(contrasts(data_temp[, var][[1]]))
    print_empty_line()
  }
}

print("Coding of variables: ")
check_coding(data_model, var_list)
# all factor-variables are properly dummy-coded


# check relationship of age and outcome-variable
# plotting distribution of outcomes over age
cdplot(decision ~ age, 
       data = data_model,
       col = c("green", "blue"),
       ylab = "Decision for or against the product with the carbon-neutral label",
       xlab = "Age",
       main = "Conditional density plot")
# outcome-distributions seems approximately equal over age



# normalize numeric variables

# function that standardizes columns of a dataet based on a subset of variables
# given by a list
standardize_data <- function(dataset, variable_list){
  var_list_temp <- variable_list
  numeric_cols_temp <- dataset[, var_list_temp] %>% 
    select_if(is.numeric) %>% colnames()
  print("Standardized variables:")
  print(numeric_cols_temp)
  #dataset_num_temp <- dataset[, numeric_cols_temp]
  #min_max_scaler <- preProcess(dataset_num_temp, method = "range")
  #dataset[, numeric_cols_temp] <- predict(min_max_scaler, 
  #                                       dataset[, numeric_cols_temp])
  dataset[, numeric_cols_temp] <- scale(dataset[, numeric_cols_temp])
  return(dataset)
}

data_standardized <- standardize_data(data_model, var_list)



# Section 6: Get training-, validation-, and test datasets----------------------

# train test split

# subset dataframe for variables that are relevant for further analyses
data_model <- data_standardized %>% 
  subset(select = c(lfdn, purchase_CHO, purchase_PA, purchase_GB, rank_GB, 
                    rank_CHO, rank_CHEESE, rank_WHEAT, rank_TOM, age, gender, 
                    country, education, occupation, av_income, environment, 
                    Country_Name, Generation, av_income_factorized, choice, 
                    decision, price_cn, price_no_cn, product_type, 
                    final_comment))


# split data into training and test set by including certain individuals in the
# test set only to get a better estimate of the generalization error
individuals <- unique(data_model$lfdn)

# determine fractions for train and test set, I decided to include about 15%
# of individuals in the test set
train_size <- floor(0.70 * length(individuals))
valid_test_size <- (length(individuals) - train_size)

# sample individuals for training and test set
# make sure the results can be repeated the same way
set.seed(100)

train_idx <- sample(individuals, train_size, replace = FALSE)
valid_test_idx <- individuals[which(!individuals %in% train_idx)]

valid_size = floor(0.5 * length(valid_test_idx))
valid_idx <- sample(valid_test_idx, valid_size, replace = FALSE)
test_idx = valid_test_idx[which(!valid_test_idx %in% valid_idx)]


# test whether it worked correctly
count_dub = 0
for (i in train_idx){
  if (i %in% test_idx){
    count_dub = count_dub + 1
  }
}
print(paste0("number of dublicates: ", count_dub))
print(paste0("unique idx train set: ", length(unique(train_idx))))
print(paste0("unique idx valid set: ", length(unique(valid_idx))))
print(paste0("unique idx test set: ", length(unique(test_idx))))
# worked

# split data into train and test set
train <- data_model[data_model$lfdn %in% train_idx, ]
valid <- data_model[data_model$lfdn %in% valid_idx, ]
test <- data_model[data_model$lfdn %in% test_idx, ]
# train might not contain exactly 85% of observations since the split was made
# based on individuals, hence, train has 85% of individuals, not necessary 85%
# of observations

# check if outcome distributions are approximately similar
print("outcome-distributions for training and test set (relative frequency:")
print(round(table(train$decision) / nrow(train), 2))
print(round(table(valid$decision) / nrow(valid), 2))
print(round(table(test$decision) / nrow(test), 2))
# distributions are similar



# check distribution of outcome-variable
print("Distribution of outcome variable in training set:")
print(table(train$decision))

# create more balanced training sets
# option 1: upsample training set

# create dataframe with upsampled data
set.seed(100)  # next command samples
train_upsampled <- upSample(x = train %>% subset(select = -c(decision)),
                            y = as.factor(train$decision),
                            yname = "decision")

# check results
print("Distribution of outcome variable in upsampled training set:")
table(train_upsampled$decision)


# option 2: downsample training set
set.seed(100)  # just to make sure
train_downsampled <- downSample(x = train %>% subset(select = -c(decision)),
                                y = as.factor(train$decision),
                                yname = "decision")

# check results
print("Distribution of outcome variable in downsampled training set:")
table(train_downsampled$decision)


# option 3: ROSE

# preparation for ROSE
train_2 <- train
train_2$choice <- as.factor(train_2$choice)
train_2$product_type <- as.factor(train_2$product_type)



set.seed(100)
train_rose <- ROSE(as.factor(decision) ~ age + as.numeric(gender) + 
                     as.numeric(Country_Name) + as.numeric(education) + 
                     as.numeric(occupation) + as.numeric(environment) + 
                     as.numeric(av_income_factorized) + as.numeric(choice) + 
                     price_cn + price_no_cn + as.numeric(product_type) +
                     purchase_CHO + purchase_PA + purchase_GB, 
                   data = train_2)$data  
# variables only transformed in the formula for the computation, hence the 
# resulting dataframe keeps the assigned labels
# despite the name, no roses were trained in the process 
print("Distribution of outcome variable in ROSE training set:")
table(train_rose$decision)
# ROSE generates observations based on neighbours by using a smoothed 
# bootstrapping approach, it selects an observation of one of the classes and 
# generates new examples in its neighbourhood 
# kernel-methods usually only work with continous data, but ROSE also works
# with non-continous data, since it uses over- and undersampling
# (e.g. a synthetic clone of x_i will have the same component for the j-th
# variable as x_i)


# option 4: SMOTE
# SMOTE only works with continuous data and is hence not applicable here



# export data
# does not need to be repeated every time the script runs
write.csv(data_model, "data/data_model.csv")
# export train and test sets
write.csv(train, "data/train_model.csv")
write.csv(valid, "data/valid_model.csv")
write.csv(test, "data/test_model.csv")
write.csv(train_upsampled, "data/train_model_upsampled.csv")
write.csv(train_downsampled, "data/train_model_downsampled.csv")
write.csv(train_rose, "data/train_model_rose.csv")



# Section 7: Base Logistic Regression-------------------------------------------

# models for explanation

# null model
# models are fit on whole data set, since this part is about inference,
# predictions are done in Python later
fit_0 <- glmer(decision ~ 1 + (1 | lfdn),
               data = train, family = "binomial"(link = "logit"))

summary(fit_0)


# full model
fit_1 <- glmer(decision ~ gender + education + occupation + 
                 av_income_factorized + price_cn + price_no_cn + product_type + 
                 environment + Country_Name + age + purchase_CHO + 
                 purchase_PA + purchase_GB + (1 | lfdn),
               data = train, family = "binomial"(link = "logit"))

summary(fit_1)


# trying a model with a squared age-term
fit_2 <- glmer(decision ~ gender + education + occupation + 
                 av_income_factorized + price_cn + price_no_cn + 
                 product_type + environment + age + I(age^2) + Country_Name + 
                 purchase_CHO + purchase_PA + purchase_GB + (1 | lfdn),
               data = train, family = "binomial")

summary(fit_2)
# does not do much for the significance of age, also AIC slightly higher,
# this ain't it
# BIC is also lower for the second model (BIC penalizes more variables harder,
# hence, adding all the variables seems to be sensible)



# anova to compare null model and full model
print(anova(fit_0, fit_1))
# the full model shows an improved fit compared to the null model based on 
# AIC, maximized log likelihood, and deviance left

# check inidvidual variables in model
print(anova(fit_1))
# age and gender seem to not explain much of the model variation, trying 
# another model
fit_3 <- glmer(decision ~ gender + price_cn + price_no_cn + education +
                 product_type + environment + age + I(age^2) + Country_Name + 
                 purchase_CHO + purchase_PA + purchase_GB + (1 | lfdn),
               data = train, family = "binomial")

summary(fit_3)
anova(fit_2, fit_3)
# the newest model has a slightly lower AIC and BIC (which makes sense, since
# contains two variables less, but it explains slightly less of the deviance,
# the difference between the models is not significant, hence, the first model
# will be kept)



# model for prediction without random effects, so a fixed effects model
fit_predict <- glm(decision ~ gender + education + occupation + 
                     av_income_factorized + price_cn + price_no_cn + 
                     product_type + environment + age + Country_Name + 
                     purchase_CHO + purchase_PA + purchase_GB,
                   data = train, family = "binomial"(link = "logit"))

summary(fit_predict)

# compare fixed-effects model and mixed-effects model
anova(fit_1, fit_predict)
# all measures indicate that the random effects model works better for fitting
# the data
# for prediction purposes the fixed effects model will be used since the test
# data does not include the same individuals as the training data



# other training sets
# ROSE
fit_rose_predict <- glm(decision ~ gender + education + occupation + 
                          av_income_factorized + price_cn + price_no_cn + 
                          product_type + environment + age + Country_Name + 
                          purchase_CHO + purchase_PA + purchase_GB,
                        data = train_rose, family = "binomial"(link = "logit"))

# downsampled
fit_ds_predict <- glm(decision ~ gender + education + occupation + 
                        av_income_factorized + price_cn + price_no_cn + 
                        product_type + environment + age + Country_Name + 
                        purchase_CHO + purchase_PA + purchase_GB,
                      data = train_downsampled, family = "binomial"(link = "logit"))


fit_us_predict <- glm(decision ~ gender + education + occupation + 
                        av_income_factorized + price_cn + price_no_cn + 
                        product_type + environment + age + Country_Name + 
                        purchase_CHO + purchase_PA + purchase_GB,
                      data = train_upsampled, family = "binomial"(link = "logit"))



# Section 8: prediction and export of results-----------------------------------

# predictions
predictions_proba <- predict(fit_predict, newdata = test, type = "response")
predictions <- round(predictions_proba)
# type = "response" returns probabilties instead of log odds

# rose
rose_predictions_proba <- predict(fit_rose_predict, newdata = test, 
                                  type = "response")
rose_predictions <- round(rose_predictions_proba)


# downsampled
ds_predictions_proba <- predict(fit_ds_predict, newdata = test, 
                                type = "response")
ds_predictions <- round(ds_predictions_proba)


# upsampled
us_predictions_proba <- predict(fit_us_predict, newdata = test, 
                                type = "response")
us_predictions <- round(us_predictions_proba)



# export latex-code of outputs
sink(file = "outputs/stargazer_fit_predict_output.txt")
stargazer(fit_predict, single.row = TRUE)
sink(file = NULL)


sink(file = "outputs/stargazer_whole_output.txt")
stargazer(fit_0, fit_1, fit_3, fit_predict, single.row = TRUE)
sink(file = NULL)


sink(file = "outputs/stargazer_inference_output.txt")
stargazer(fit_0, fit_1, fit_2, fit_3, single.row = TRUE)
sink(file = NULL)


sink(file = "outputs/stargazer_inference_compact_output.txt")
stargazer(fit_0, fit_1, fit_3, single.row = TRUE)
sink(file = NULL)


sink(file = "outputs/stargazer_inference_prediction.txt")
stargazer(fit_0, fit_1, fit_predict, single.row = TRUE)
sink(file = NULL)


sink(file = "outputs/stargazer_inference_compacter.txt")
stargazer(fit_0, fit_1, single.row = TRUE)
sink(file = NULL)


# other training sets
sink(file = "outputs/stargazer_predict_rose.txt")
stargazer(fit_rose_predict, single.row = TRUE)
sink(file = NULL)

sink(file = "outputs/stargazer_predict_ds.txt")
stargazer(fit_ds_predict, single.row = TRUE)
sink(file = NULL)

sink(file = "outputs/stargazer_predict_us.txt")
stargazer(fit_us_predict, single.row = TRUE)
sink(file = NULL)



# export predictions to import them in Python for prediction comparisons
write.csv(predictions_proba, "outputs/base_log_reg_predictions_proba.csv")
write.csv(predictions, "outputs/base_log_reg_predictions.csv")

# rose
write.csv(rose_predictions_proba, "outputs/base_log_reg_rose_predictions_proba.csv")
write.csv(rose_predictions, "outputs/base_log_reg_rose_predictions.csv")

# downsampled
write.csv(ds_predictions_proba, "outputs/base_log_reg_ds_predictions_proba.csv")
write.csv(ds_predictions, "outputs/base_log_reg_ds_predictions.csv")

# upsampled
write.csv(us_predictions_proba, "outputs/base_log_reg_us_predictions_proba.csv")
write.csv(us_predictions, "outputs/base_log_reg_us_predictions.csv")



# Outro-------------------------------------------------------------------------
"
To continue the analysis, please open the python-code-file.
"



