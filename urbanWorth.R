###############################
# HOUSE PRICE PREDICTION MODEL
# Ames 2000 Dataset
###############################

### --- 1. Downloading Appropriate Libraries --- ###
set.seed(123)
library(dplyr)
library(ggplot2)
library(corrplot)
library(readr)
options(max.print = 100)  # Display up to 100 columns


### --- 2. Reading the Data --- ###
data <- read.csv("ames2000_NAfix.csv")
nrow(data)
ncol(data)


### --- 3. Converting Categorical Variables into Numeric Factors --- ###
categorical <- c("MS.SubClass", "MS.Zoning", "Street", "Alley", "Lot.Shape",
                 "Land.Contour", "Utilities", "Lot.Config", "Land.Slope", "Neighborhood",
                 "Condition.1", "Condition.2", "Bldg.Type", "House.Style", "Overall.Qual",
                 "Overall.Cond", "Roof.Style", "Roof.Matl", "Exterior.1st", "Exterior.2nd",
                 "Mas.Vnr.Type", "Exter.Qual", "Exter.Cond", "Foundation", "Bsmt.Qual", "Bsmt.Cond",
                 "Bsmt.Exposure", "BsmtFin.Type.1", "BsmtFin.Type.2", "Heating", "Heating.QC",
                 "Central.Air", "Electrical", "Kitchen.Qual", "Functional", "Fireplace.Qu",
                 "Garage.Type", "Garage.Yr.Blt", "Garage.Finish", "Garage.Qual", "Garage.Cond",
                 "Paved.Drive", "Pool.QC", "Fence", "Misc.Feature", "Sale.Type", "Sale.Condition")

for (i in 1:length(categorical)) {
  data[, categorical[i]] <- as.numeric(factor(data[, categorical[i]]))
}


### --- 4. Dropping All Columns Containing NAs --- ###
drop_na_columns <- function(data) {
  data <- data %>%
    select_if(~ !any(is.na(.)))
  return(data)
}

data <- drop_na_columns(data)


### --- 5. Shuffling & Splitting the Data --- ###
data <- sample(data, size = length(data), replace = FALSE)
training_data <- sample_frac(data, size = 0.5)
testing_data <- sample_frac(data, size = 0.5)


### --- 6. Calculating Correlation of Each Variable with SalePrice --- ###
correlations <- data.frame(Column = character(0), Correlation = numeric(0))

for (col_name in colnames(data)) {
  correlation <- cor(as.numeric(data[[col_name]]), data$SalePrice)
  correlations <- rbind(correlations,
                        data.frame(Column = col_name, Correlation = correlation))
}
print(correlations)

# I will use every variable with a correlation coefficient >= 0.5 as a predictor variable.


### --- 7. Plotting Correlation Matrix of Predictor Variables --- ###
var <- c("Overall.Qual", "Year.Built", "Year.Remod.Add", "X1st.Flr.SF", "Gr.Liv.Area",
         "TotRms.AbvGrd", "Full.Bath", "Kitchen.Qual", "Exter.Qual", "Bsmt.Qual")

corr.matrix <- cor(data[var])
corrplot(corr.matrix, method = "circle")

# Observations:
# - High collinearity between:
#   1. "TotRms.AbvGrd" & "Gr.Liv.Area"
#   2. "Overall.Qual" & "Exter.Qual"
#   3. "Year.Built" & "Bsmt.Qual"


### --- 8. Spread of Predictor Variables --- ###
# Boxplot of "Year.Built"
boxplot(data$Year.Built, data = data, main = "Boxplot of Year.Built")
1880

# Boxplot of "Full.Bath"
boxplot(data$Full.Bath, data = data, main = "Boxplot of Full.Bath")
5

# Violin plot of "X1st.Flr.SF"
ggplot(data, aes(x = factor(1), y = X1st.Flr.SF)) +
  geom_violin(trim = FALSE, fill = "maroon") +
  geom_boxplot(width = 0.15, fill = "white", color = "black", outlier.shape = NA) +
  labs(x = "", y = "X1st.Flr.SF") +
  ggtitle("Violin plot of X1st.Flr.SF")

# Violin plot of "Gr.Liv.Area"
ggplot(data, aes(x = factor(1), y = Gr.Liv.Area)) +
  geom_violin(trim = FALSE, fill = "maroon") +
  geom_boxplot(width = 0.15, fill = "white", color = "black", outlier.shape = NA) +
  labs(x = "", y = "Gr.Liv.Area") +
  ggtitle("Violin plot of Gr.Liv.Area")


### --- 9. Linear Regression: Checking for Interactions --- ###
par(mfrow = c(2, 2))
require(regclass)
library(tidyr)

selected_cols <- c("Overall.Qual", "Year.Built", "Year.Remod.Add", "X1st.Flr.SF",
                   "Gr.Liv.Area", "Full.Bath", "TotRms.AbvGrd", "Kitchen.Qual",
                   "Exter.Qual", "Bsmt.Qual", "SalePrice")

data_selected <- training_data[selected_cols]
data_long <- gather(data_selected, Predictor, Value, -SalePrice)

ggplot(data_long, aes(x = Value, y = SalePrice, color = Predictor, group = Predictor)) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Predictor Value", y = "Sale Price", title = "Interaction Plot") +
  theme_minimal()

# Observations:
# - Interaction between "Year.Built" & "Year.Remod.Add"
# - Three-way interaction: "Year.Built" * "Year.Remod.Add" * "Gr.Liv.Area"


### --- 10. Original Model --- ###
original.model <- lm(SalePrice ~ Overall.Qual + Year.Built + + Year.Remod.Add + X1st.Flr.SF +
                       Gr.Liv.Area + Full.Bath + TotRms.AbvGrd + Kitchen.Qual +
                       Exter.Qual + Bsmt.Qual +
                       Year.Built * Year.Remod.Add +
                       Year.Built * Year.Remod.Add * Gr.Liv.Area,
                     data = training_data)
summary(original.model)

orig.VIF <- VIF(original.model)
print(orig.VIF)

# VIF > 5 for "Gr.Liv.Area" & "TotRms.AbvGrd" → Collinearity detected.
# Dropping "TotRms.AbvGrd" to improve accuracy.


### --- 11. Updated Original Model --- ###
new.original.model <- lm(SalePrice ~ Overall.Qual + Year.Built + Year.Remod.Add + X1st.Flr.SF +
                           Gr.Liv.Area + Full.Bath + Kitchen.Qual + Exter.Qual + Bsmt.Qual +
                           Year.Built * Year.Remod.Add +
                           Year.Built * Year.Remod.Add * Gr.Liv.Area,
                         data = training_data)
summary(new.original.model)

plot(new.original.model, lwd = 2, main = "Updated Original Model")


### --- 12. Transformations of SalePrice --- ###
# (Square Root Transformation)
square_root <- lm(sqrt(SalePrice) ~ Overall.Qual + Year.Built + + Year.Remod.Add + X1st.Flr.SF +
                    Gr.Liv.Area + Full.Bath + Kitchen.Qual + Exter.Qual + Bsmt.Qual +
                    Year.Built * Year.Remod.Add +
                    Year.Built * Year.Remod.Add * Gr.Liv.Area,
                  data = training_data)
summary(square_root)
plot(square_root, lwd = 2, main = "Square Root of SalePrice")

# (Log Transformation)
par(mfrow = c(2, 2))
log <- lm(log(SalePrice) ~ Overall.Qual + Year.Built + + Year.Remod.Add + X1st.Flr.SF +
            Gr.Liv.Area + Full.Bath + Kitchen.Qual + Exter.Qual + Bsmt.Qual +
            Year.Built * Year.Remod.Add +
            Year.Built * Year.Remod.Add * Gr.Liv.Area,
          data = training_data)
summary(log)
plot(log, lwd = 2, main = "Log of SalePrice")

# (Squaring)
square <- lm((SalePrice)^2 ~ Overall.Qual + Year.Built + + Year.Remod.Add + X1st.Flr.SF +
               Gr.Liv.Area + Full.Bath + Kitchen.Qual + Exter.Qual + Bsmt.Qual +
               Year.Built * Year.Remod.Add +
               Year.Built * Year.Remod.Add * Gr.Liv.Area,
             data = training_data)
summary(square)
plot(square, lwd = 2, main = "Squaring SalePrice")


### --- 13. Random Forest --- ###
set.seed(123)
library(randomForest)

m1 <- randomForest(SalePrice ~ Overall.Qual + Year.Built + + Year.Remod.Add + X1st.Flr.SF +
                     Gr.Liv.Area + Full.Bath + Kitchen.Qual + Exter.Qual + Bsmt.Qual +
                     Year.Built * Year.Remod.Add +
                     Year.Built * Year.Remod.Add * Gr.Liv.Area,
                   data = training_data, mtry = sqrt(79), ntree = 100)
m1

rf_formula <- SalePrice ~ Overall.Qual + Year.Built + Year.Remod.Add + X1st.Flr.SF +
  Gr.Liv.Area + Full.Bath + Kitchen.Qual + Exter.Qual + Bsmt.Qual +
  Year.Built * Year.Remod.Add +
  Year.Built * Year.Remod.Add * Gr.Liv.Area

m2 <- randomForest(rf_formula, data = training_data, mtry = sqrt(79), ntree = 50)
m3 <- randomForest(rf_formula, data = training_data, mtry = 79 / 2, ntree = 50)
m4 <- randomForest(rf_formula, data = training_data, mtry = 79, ntree = 50)
m5 <- randomForest(rf_formula, data = training_data, mtry = sqrt(79), ntree = 500)
m5

### --- 14. Model Selection & Confidence Intervals --- ###
predicted_value <- predict(log, testing_data, type = "response")

print(confint(log))
print(confint(square))
print(confint(square_root))
print(confint(original.model))

# Training Data Confidence Interval
log.training <- lm(log(SalePrice) ~ Overall.Qual + Year.Built + + Year.Remod.Add + X1st.Flr.SF +
                     Gr.Liv.Area + Full.Bath + TotRms.AbvGrd + Kitchen.Qual +
                     Exter.Qual + Bsmt.Qual +
                     Year.Built * Year.Remod.Add +
                     Year.Built * Year.Remod.Add * Gr.Liv.Area,
                   data = training_data)
print(confint(log.training))

# Observations:
# Confidence intervals are not significantly different between training & testing sets
# due to random sampling.


### --- 15. Prediction Accuracy (MSE) --- ###
actual.value <- log(testing_data$SalePrice)
mse <- mean((actual.value - predicted_value)^2)
print(mse)


### --- 16. Scatterplot: Actual vs Predicted House Prices --- ###
plot(predicted_value,
     actual.value,
     xlab = "Value Predicted by Model",
     ylab = "Actual Value of the House",
     main = "Actual Value vs Predicted Value using Linear Regression",
     pch = 16)

# Plotting y = x line
abline(a = 0, b = 1, col = "blue", lwd = 2)

