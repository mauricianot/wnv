
# Load required packages (Plot and Selected variable)
library(ggplot2)    # Visualization
library(dplyr)      # Data manipulation
library(glmnet)     # LASSO and Ridge regression
library(caret)      # Cross-validation
library(pROC)       # ROC curve for binary models
library(MuMIn)      # AIC for model selection
library(corrplot)   # Correlation

# Load required packages (model NLME and spatial data)
library(nlme)       # For Linear Mixed Models (LMM)
library(MASS)       # For stepAIC (AIC-based model selection)
library(sp)         # For spatial data handling
library(gstat)      # For variogram analysis

# set path folder
setwd(paste0("/srv/shiny-server/projet-zoonotic/"))

# ---- 1. Data for analysis environnemental and demographic (sex and age) ----
data.model.expo <- read.csv("./WNV_data.csv")
data.model.expo$sex <- ifelse(data.model.expo$sex == "masculin", 1, 0)
data.model.expo$animal_poultry <- ifelse(!is.na(data.model.expo$animal_poultry),
                                         data.model.expo$animal_poultry,0)

# ---- 2. Correlation between all variables (environemental, socioeconomic and demographic----
# Compute Spearman correlation matrix excluding specific columns
data.model.expo.corr1 <- data.model.expo[, !colnames(data.model.expo) %in% 
                                           c("village_id", "cluster_id", 
                                             "WN.ENV.Alog_blk",
                                             "WN_ENVandNS1_Sero_3sd_corr")]
cor_matrix_all <- cor(data.model.expo.corr1, 
                      method = "spearman", 
                      use = "pairwise.complete.obs")

# Define variable names
var_names_all <- c(
  "Average elevation (m)", 
  "Percentage of Forest around household (%)",
  "Percentage of Residential area around household (%)",
  "Percentage of Rice field around household (%)",
  "Percentage of Savannah around household (%)", 
  "Percentage of Waterbodies around household (%)",
  "Annual Maximum Percentage of Forest loss over 5 year (%)",
  "Population density (buildings/km²)",
  "Average Maximum flooded rice fields - primary agricultural season (%)",
  "Average Maximum flooded rice fields - secondary agricultural season (%)",
  "Annual Maximum Daily Relative Humidity (RH) (%)",
  "Annual Minimum Land Surface Temperature (LST) (°C)",
  "Precipitation of Wettest Month (mm)",
  "Normalized Difference Vegetation Index (NDVI)",
  "Topographic Wetness Index (TWI)",
  "Distance to secondary paved road (km)",
  "Distance to tertiary road (km)",
  "Distance to unclassified road (km)",
  "Distance to nearest PHC (km)",
  "Distance to nearest RP road (km)",
  "Rudimentary Housing", "Economic level", "Secondary education or higher", 
  "Cropland ownership", "Household cropland area", "Outdoor occupation", 
  "Livestock ownership","Total number of mammalian livestock", 
  "Total number of poultry livestock","Household Poultry Ownership",
  "Using mosquito nets", "Household Mosquito Net Ownership", 
  "Individual Mosquito Net Use", "Indoor residual spraying", 
  "Unprotected water supply source", "Open defecation",
  "Male Sex","Age"
)

# Assign names to correlation matrix
rownames(cor_matrix_all) <- var_names_all
colnames(cor_matrix_all) <- var_names_all

# # Save to PDF with large page
pdf("/srv/shiny-server/projet-zoonotic/FIGURES/corrWithAllVaribales.pdf", 
    width = 20, height = 20)  # big canvas

# Create the correlation plot
corrplot(cor_matrix_all,
         method = "circle",      # Use circles to represent correlations
         type = "lower",         # Show full matrix (not just upper or lower triangle)
         addCoef.col = "black",  # Add correlation coefficients in black
         number.cex = 0.5,       # Size of correlation coefficient numbers
         tl.cex = 1.25,          # Text label size
         tl.pos = "l")           # only top labels
dev.off()

# variables selected by context and correlation between variables without sex and age
data.model.expo.corr2 <- data.model.expo[, !colnames(data.model.expo) %in% 
                                           c("village_id", "cluster_id", 
                                             "WN.ENV.Alog_blk", 
                                             "WN_ENVandNS1_Sero_3sd_corr",
                                             "elev_mean",
                                             "dist_RP_km","animal_poultry_risk"
                                           )]
cor_matrix_final <- cor(data.model.expo.corr2, 
                        method = "spearman", 
                        use = "pairwise.complete.obs")

# Define variable names
var_names_final <- c(
  "Percentage of Forest around household (%)",
  "Percentage of Residential area around household (%)",
  "Percentage of Rice field around household (%)",
  "Percentage of Savannah around household (%)", 
  "Percentage of Waterbodies around household (%)",
  "Annual Maximum Percentage of Forest loss over 5 year (%)",
  "Population density (buildings/km²)",
  "Average Maximum flooded rice fields - primary agricultural season (%)",
  "Average Maximum flooded rice fields - secondary agricultural season (%)",
  "Annual Maximum Daily Relative Humidity (RH) (%)",
  "Annual Minimum Land Surface Temperature (LST) (°C)",
  "Precipitation of Wettest Month (mm)",
  "Normalized Difference Vegetation Index (NDVI)",
  "Topographic Wetness Index (TWI)",
  "Distance to secondary paved road (km)",
  "Distance to tertiary road (km)",
  "Distance to unclassified road (km)",
  "Distance to nearest PHC (km)",
  "Rudimentary Housing", "Economic level", "Secondary education or higher", 
  "Cropland ownership", "Household cropland area", "Outdoor occupation", 
  "Livestock ownership","Total number of mammalian livestock", 
  "Total number of poultry livestock",
  "Using mosquito nets", "Household Mosquito Net Ownership", 
  "Individual Mosquito Net Use", "Indoor residual spraying", 
  "Unprotected water supply source", "Open defecation",
  "Male Sex","Age"
)

# Assign names to correlation matrix
rownames(cor_matrix_final) <- var_names_final
colnames(cor_matrix_final) <- var_names_final

# # Save to PDF with large page
pdf("/srv/shiny-server/projet-zoonotic/FIGURES/corrWithFinalVaribales.pdf", width = 20, height = 20)  # big canvas

# Create the correlation plot
corrplot(cor_matrix_final,
         method = "circle",      # Use circles to represent correlations
         type = "lower",         # Show full matrix (not just upper or lower triangle)
         addCoef.col = "black",  # Add correlation coefficients in black
         number.cex = 0.5,       # Size of correlation coefficient numbers
         tl.cex = 1.25,          # Text label size
         tl.pos = "l")           # only top labels
dev.off()

# ---- 3. Model GLMM with variable continuous ----
# Create spatial data frame
data.model.expo.modMfi <- data.model.expo[, !colnames(data.model.expo) %in% 
                                            c("elev_mean",#"Highway_unclassified",
                                              "dist_RP_km","animal_poultry_risk"
                                            )] # variable data model GLMM

data.model.expo.modMfi <- data.model.expo.modMfi  |>
  mutate(
    sex = ifelse(sex == 1, "Male", "Female"),
    sex = factor(sex, levels = c("Female", "Male"))) |>
  # to factor all catogory variables
  mutate(across(c(village_id, cluster_id,
                  house_risk, educ_chief_mother_risk,
                  cropland, occup_princip_risk, animal,
                  mosq_net_use, mosq_net_house_risk, mosq_net_use_risk,
                  insecticides_walls, imp_water, imp_latrine), as.factor)) |>
  # to scale all numeric variables
  mutate(across(c(land_Forest, land_Residential.area, 
                  land_Rice.field, land_Savannah, land_Water.bodies, 
                  max_perct_loss5year, dp_km2, 
                  max_flood__primary, max_flood__secondary,
                  max_RH_year, min_LST_year, bio_PWM, mean_NDVI, twi_mean, 
                  Highway_secondary, Highway_tertiary, dist_PHC_km, 
                  wealth_score, cropland_ha, animal_poultry, 
                  animal_risk_sum, age), scale))
  

# Model GLMMTMB
# ---- 4.1 Fit linear and generalized linear mixed models (GLMM) ----
# exclude variables : "village_id", "cluster_id", 
# "WN.ENV.Alog", "WN.NS1.Alog", "WN_ENVandNS1_Sero_3sd", "WN_ENVandNS1_Sero_3sd_corr"
varNamesCols_modMfi <- colnames(data.model.expo.modMfi)[11:(ncol(data.model.expo.modMfi))]
formula_glmmMfi <- paste("WN.ENV.Alog_blk ~ ", 
                          paste(varNamesCols_modMfi, collapse = " + "), 
                          paste(" + ","(1 | cluster_id:village_id)"))
print(formula_glmmMfi)

# Fit full model
library(glmmTMB)
library(buildmer)

modSelect_Mfi <- buildglmmTMB(
  formula = as.formula(formula_glmmMfi),
  data = data.model.expo.modMfi,
  family = gaussian(),  # or binomial(link = "logit"), gaussian(), etc.
  buildmerControl(direction = c("backward"), 
                  include =  ~ (1 | cluster_id:village_id),
                  crit = 'AIC', elim = 'AIC')
)

bestModMfi <- modSelect_Mfi@model
summary(bestModMfi) # best model

# model GLMM final seroprevalence
modMfi_final <- glmmTMB(WN.ENV.Alog_blk ~ 1 + 
                          land_Forest + land_Residential.area + land_Savannah +
                          min_LST_year + bio_PWM + mean_NDVI + twi_mean + 
                          Highway_secondary + educ_chief_mother_risk + 
                          cropland_ha + occup_princip_risk + 
                          mosq_net_use + #mosq_net_use_risk + 
                          age + 
                          (1 | cluster_id:village_id),
                         family = gaussian(),
                         data = data.model.expo.modMfi)

summary(modMfi_final) # model final

# plot coeff model GLMM seroprevalence
library(broom.mixed)
coef_modMfi <- tidy(modMfi_final, conf.int = TRUE)
coef_modMfi <- coef_modMfi[coef_modMfi$effect=="fixed" & 
                             coef_modMfi$term!="(Intercept)",]

# Add significance codes based on p-value thresholds
coef_modMfi$signif_code <- with(coef_modMfi, ifelse(p.value < 0.001, "***",
                                                      ifelse(p.value < 0.01, "**",
                                                             ifelse(p.value < 0.05, "*",
                                                                    ifelse(p.value < 0.1, ".", " ")))))
# Convert beta coefficients to Odds Ratios (OR)
coef_modMfi$OR <- exp(coef_modMfi$estimate)

# Convert 95% confidence intervals to OR scale
coef_modMfi$OR_low <- exp(coef_modMfi$conf.low)
coef_modMfi$OR_high <- exp(coef_modMfi$conf.high)
coef_modMfi$p.value.txt <- paste0(round(coef_modMfi$OR,2), coef_modMfi$signif_code)

# Define your desired order (replace with your actual terms)
var_order_modMfi <- c("land_Forest", "land_Residential.area", "land_Savannah",
                       "min_LST_year", "bio_PWM", "mean_NDVI", "twi_mean", 
                       "Highway_secondary","educ_chief_mother_risk1",
                       "cropland_ha", "occup_princip_risk1",
                       "mosq_net_use1", #"mosq_net_use_risk1", 
                      "age")

var_order_modMfi_labs <- c( 
                         "Percentage of Forest around household", 
                         "Percentage of Residential area around household",
                         "Percentage of Savannah around household",
                         "Annual Minimum Land Surface Temperature (LST)",
                         "Precipitation of Wettest Month (bioclimatic)",
                         "Normalized Difference Vegetation Index (NDVI)",
                         "Average Topographic Wetness Index (TWI)", 
                         "Distance between household and the nearest\nMajor Paved road",
                         "Secondary education or higher",
                         "Household cropland area",
                         "Outdoor occupation",
                         "Household Mosquito Net Ownership (all types)", "Age")

# Convert 'term' to a factor with specified levels
coef_modMfi$term <- factor(coef_modMfi$term, levels = var_order_modMfi,
                            labels = var_order_modMfi_labs)
coef_modMfi$term <- factor(coef_modMfi$term, levels = rev(var_order_modMfi_labs))

# plot coef model final
#plot_coef_modMfi <- 
ggplot(coef_modMfi, aes(x = OR, y = term, xmin = OR_low, xmax = OR_high)) +
  geom_pointrange(aes(color = OR > 1), linewidth = 1.5) +  # Thicker lines for xmin-xmax
  scale_color_manual(values = c("red", "blue")) +
  geom_vline(xintercept = 1, linetype = "dashed") +  # Thicker reference line
  theme_classic(base_size = 18) +  # Increase base font size for all elements
  scale_y_discrete(position = "right")+  # Move y-axis to right
  theme(
    axis.text = element_text(size = 16),  # Increase axis label size
    axis.title.x = element_text(size = 22, face = "bold"),
    axis.title = element_text(size = 18, face = "bold"),  # Make axis titles larger & bold
    legend.position = "none"
  ) +
  labs(x = expression("Odds ratio (" * e^beta * ")"), y = NULL) +
  geom_text(aes(label = paste0(p.value.txt)), hjust = 0.5, vjust = -1, size =5.0)

ggsave(filename = "./FIGURES/coef_modMfi_600dpi.png",
       plot_coef_modMfi,
       width = 14, height = 9, dpi = 600,
       units = "in", device='png', bg = "white")


# Model evaluation
# Residual Diagnostics with DHARMa
library(DHARMa)
sim_res_modMfi_final <- simulateResiduals(modMfi_final)
plot(sim_res_modMfi_final)

# Extract residuals
res_scale_modMfi <- sim_res_modMfi_final$scaledResiduals
res_fitted_modMfi <- unname(sim_res_modMfi_final$fittedResiduals)
all.equal(res_scale_modMfi, res_fitted_modMfi)  # Should be TRUE

# Plot histogram
hist(res_fitted_modMfi, main = "DHARMa Scaled Residuals")

# Model Fit (Pseudo R²)
library(performance)
r2_tjur(modMfi_final) # Coefficient of determination (D)
pscl::pR2(modMfi_final) # Calculate R² in R for a Binomial Model (McFadden)
rmse(modMfi_final) # Check RMSE

# ANOVA-style Tests
library(car)
Anova(modMfi_final, type = "III")

# Effect Visualization : evaluate the significance of fixed effects
library(effects)
plot(allEffects(modMfi_final))
