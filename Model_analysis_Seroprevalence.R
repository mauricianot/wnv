
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

# ---- 1. Data for analysis environnemental and demographic (sex and age) ----
data.model.expo <- read.csv("./WNV_data.csv")
data.model.expo$sex <- ifelse(data.model.expo$sex == "masculin", 1, 0)
data.model.expo$animal_poultry <- ifelse(!is.na(data.model.expo$animal_poultry),
                                         data.model.expo$animal_poultry,0)

# ---- 2. Correlation between all variables (environemental, socioeconomic and demographic----
# Compute Spearman correlation matrix excluding specific columns
data.model.expo.corr1 <- data.model.expo[, !colnames(data.model.expo) %in% 
                                          c("village_id", "cluster_id", 
                                            "WN.ENV.Alog_blk", "WN_ENVandNS1_Sero_3sd_corr")]
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
pdf("./FIGURES/corrWithAllVaribales.pdf", width = 20, height = 20)  # big canvas

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
# removed "elev_mean", "dist_RP_km", "land_Water.bodies", "bio_PS", "bio_PDM", 
data.model.expo.corr2 <- data.model.expo[, !colnames(data.model.expo) %in% 
                                           c("village_id", "cluster_id", 
                                             "WN.ENV.Alog_blk", 
                                             "WN_ENVandNS1_Sero_3sd_corr",
                                             "elev_mean", "dist_RP_km",
                                             "animal_poultry_risk"
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
pdf("./FIGURES/corrWithFinalVaribales.pdf", width = 20, height = 20)  # big canvas

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
data.model.expo.modSero <- data.model.expo[, !colnames(data.model.expo) %in% 
                                             c("elev_mean",
                                               "dist_RP_km",
                                               "animal_poultry_risk"
                                             )] # variable data model GLMM

data.model.expo.modSero <- data.model.expo.modSero  |>
  mutate(
    sex = ifelse(sex == 1, "Male", "Female"), 
    sex = factor(sex, levels = c("Female", "Male"))) |>
  # to factor all catogory variables
  mutate(across(c(village_id, cluster_id, 
                  WN_ENVandNS1_Sero_3sd_corr,
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
                  Highway_secondary, Highway_tertiary, Highway_unclassified, dist_PHC_km, 
                  wealth_score, cropland_ha, animal_poultry, 
                  animal_risk_sum, age), scale))
  

# Model GLMMTMB
# ---- 4.1 Fit linear and generalized linear mixed models (GLMM) ----
varNamesCols_modSero <- colnames(data.model.expo.modSero)[11:(ncol(data.model.expo.modSero))]
formula_glmmSero <- paste("WN_ENVandNS1_Sero_3sd_corr ~ ", 
                          paste(varNamesCols_modSero, collapse = " + "), 
                          paste(" + ","(1 | cluster_id:village_id)"))
print(formula_glmmSero)

# Fit full model
library(glmmTMB)
library(buildmer)

modSelect_Sero <- buildglmmTMB(
  formula = as.formula(formula_glmmSero),
  data = data.model.expo.modSero,
  family = binomial(link = "logit"),  # or binomial(), gaussian(), etc.
  buildmerControl(direction = c("backward"), 
                  include =  ~ (1 | cluster_id:village_id),
                  crit = 'AIC', elim = 'AIC')
)

#bestModSero <- modSelect_Sero@model
summary(modSelect_Sero@model)
summary(bestModSero) # best model

# model GLMM final seroprevalence
modSero_final <- glmmTMB(WN_ENVandNS1_Sero_3sd_corr ~ 1 + 
                           land_Forest + land_Residential.area + land_Savannah +
                           bio_PWM + mean_NDVI + twi_mean + 
                           Highway_secondary + dist_PHC_km + 
                           cropland_ha + occup_princip_risk + #animal_risk_sum + 
                           mosq_net_use + #mosq_net_use_risk + 
                           age + 
                           (1 | cluster_id:village_id),
                         family = binomial(link = "logit"),
                         data = data.model.expo.modSero)

summary(modSero_final) # model final

# plot coeff model GLMM seroprevalence
library(broom.mixed)
coef_modSero <- tidy(modSero_final, conf.int = TRUE)
coef_modSero <- coef_modSero[coef_modSero$effect=="fixed" &
                               coef_modSero$term!="(Intercept)",]

# Add significance codes based on p-value thresholds
coef_modSero$signif_code <- with(coef_modSero, ifelse(p.value < 0.001, "***",
                                                      ifelse(p.value < 0.01, "**",
                                                             ifelse(p.value < 0.05, "*",
                                                                    ifelse(p.value < 0.1, ".", " ")))))
# Convert beta coefficients to Odds Ratios (OR)
coef_modSero$OR <- exp(coef_modSero$estimate)

# Convert 95% confidence intervals to OR scale
coef_modSero$OR_low <- exp(coef_modSero$conf.low)
coef_modSero$OR_high <- exp(coef_modSero$conf.high)
coef_modSero$p.value.txt <- paste0(round(coef_modSero$OR,2), coef_modSero$signif_code)

# Define your desired order (replace with your actual terms)
var_order_modSero <- c("land_Forest", "land_Residential.area", 
                       "land_Savannah",
                       "bio_PWM", "mean_NDVI", "twi_mean", 
                       "Highway_secondary","dist_PHC_km",
                       "cropland_ha", "occup_princip_risk1", 
                       "mosq_net_use1", 
                       "age")

var_order_modSero_labs <- c( 
                         "Percentage of Forest around household", 
                         "Percentage of Residential area around household",
                         "Percentage of Savannah around household",
                         "Precipitation of Wettest Month (bioclimatic)", 
                         "Normalized Difference Vegetation Index (NDVI)",
                         "Average Topographic Wetness Index (TWI)", 
                         "Distance between household and the nearest\nMajor Paved road",
                         "Distance between household and the nearest\nPrimary Health Centers (PHC)",
                         "Household cropland area",
                         "Outdoor occupation",
                         "Household Mosquito Net Ownership (all types)",
                         "Age")

# Convert 'term' to a factor with specified levels
coef_modSero$term <- factor(coef_modSero$term, levels = var_order_modSero,
                            labels = var_order_modSero_labs)
coef_modSero$term <- factor(coef_modSero$term, levels = rev(var_order_modSero_labs))

# plot coef model final
#plot_coef_modSero <- 
ggplot(coef_modSero, aes(x = OR, y = term, xmin = OR_low, xmax = OR_high)) +
  geom_pointrange(aes(color = OR > 1), linewidth = 1.5) +  # Thicker lines for xmin-xmax
  scale_color_manual(values = c("red", "blue")) +
  geom_vline(xintercept = 1, linetype = "dashed") +  # Thicker reference line
  theme_classic(base_size = 18) +  # Increase base font size for all elements
  scale_y_discrete(position = "right")+  # Move y-axis to right
  theme(
    axis.text = element_text(size = 18),  # Increase axis label size
    axis.title.x = element_text(size = 22, face = "bold"),
    axis.title = element_text(size = 18, face = "bold"),  # Make axis titles larger & bold
    legend.position = "none"
  ) +
  labs(x = expression("Odds ratio (" * e^beta * ")"), y = NULL) +
  geom_text(aes(label = paste0(p.value.txt)), hjust = 0.5, vjust = -1, size =5.5)

ggsave(filename = "./FIGURES/coef_modSero_600dpi.png",
       plot_coef_modSero,
       width = 14, height = 9, dpi = 600,
       units = "in", device='png', bg = "white")

# Model evaluation
# Residual Diagnostics with DHARMa
library(DHARMa)
sim_res_modSero_final <- simulateResiduals(modSero_final)
plot(sim_res_modSero_final)

# Extract residuals
res_scale_modSero <- sim_res_modSero_final$scaledResiduals
res_fitted_modSero <- unname(sim_res_modSero_final$fittedResiduals)
all.equal(res_scale_modSero, res_fitted_modSero)  # Should be TRUE

# Plot histogram
hist(res_fitted_modSero, main = "DHARMa Scaled Residuals")

# Model Fit (Pseudo R²)
library(performance)
r2_tjur(modSero_final) # Coefficient of determination (D)
pscl::pR2(modSero_final) # Calculate R² in R for a Binomial Model (McFadden)
rmse(modSero_final) # Check RMSE

# ANOVA-style Tests
library(car)
Anova(modSero_final, type = "III")

# Effect Visualization : evaluate the significance of fixed effects
library(effects)
plot(allEffects(modSero_final))

###### Age-prevalence #################
df_plot_age.prev <- data.model.expo[, colnames(data.model.expo) %in% 
                                      c("cluster_id",
                                        "WN_ENVandNS1_Sero_3sd_corr", "age")]

# spatial grouby cluster
df_plot_age.prev.clust <- df_plot_age.prev |>
  dplyr::select(cluster_id, WN_ENVandNS1_Sero_3sd_corr) |> 
  group_by(cluster_id) |>
  dplyr::summarise_all(mean) |> arrange(cluster_id)

## classed by level risk form 80 cluster
quantiles.grapp_corr <- quantile(df_plot_age.prev.clust$WN_ENVandNS1_Sero_3sd_corr, 
                                 probs = c(0.25, 0.75))
df_plot_age.prev.clust <- df_plot_age.prev.clust %>%
  mutate(clsuter_type_corr = case_when(
    (WN_ENVandNS1_Sero_3sd_corr <= quantiles.grapp_corr[1]) ~ "Low",
    ((WN_ENVandNS1_Sero_3sd_corr > quantiles.grapp_corr[1]) & (WN_ENVandNS1_Sero_3sd_corr <= quantiles.grapp_corr[2])) ~ "Medium",
    (WN_ENVandNS1_Sero_3sd_corr > quantiles.grapp_corr[2]) ~ "High"
  ))

# merge group by cluster to three classes (low, medium, and high) join to all data
df_plot_age.prev <- merge(df_plot_age.prev, 
                          df_plot_age.prev.clust[,c('cluster_id', 'clsuter_type_corr')], 
                          by="cluster_id")
df_plot_age.prev <- df_plot_age.prev |>
                          mutate(age_group = cut(
                            age,
                            breaks = c(0, 1, 5, 10, 20, 40, Inf),   # your custom breakpoints
                            include.lowest = TRUE,       # include lowest value in first interval
                            right = TRUE,                # intervals are (a,b]
                            labels = c("[0–1]", "(1–5]", "(5–10]", "(10–20]", "(20–40]" , "(40+]")
                          ))

# Check the categories
#table(df_plot_age.prev$clsuter_type)
table(df_plot_age.prev$clsuter_type_corr)
table(df_plot_age.prev$age_group)

library(binom)
data.mapAgeP.all <- df_plot_age.prev |> 
  dplyr::select(age_group, WN_ENVandNS1_Sero_3sd_corr) |> #WN_ENVandNS1_Sero_3sd,
  group_by(age_group) |>
  summarise(
    n = n(),
    pos_corr = sum(WN_ENVandNS1_Sero_3sd_corr),
    .groups = "drop"
  ) |>
  #filter(n >= 5) %>% # optional: exclude very small villages
  mutate(
    ci_corr = binom.confint(pos_corr, n, methods = "wilson")
  ) |>
  mutate(
    seroprev_corr = ci_corr$mean * 100,
    lower_CI_corr = ci_corr$lower * 100,
    upper_CI_corr = ci_corr$upper * 100
  ) |> arrange(age_group)

data.mapAgeP_corr <- df_plot_age.prev |>
  dplyr::select(clsuter_type_corr, age_group, WN_ENVandNS1_Sero_3sd_corr) %>%
  group_by(clsuter_type_corr, age_group) |>
  summarise(
    n = n(),
    pos_corr = sum(WN_ENVandNS1_Sero_3sd_corr),
    .groups = "drop"
  ) |>
  mutate(
    ci_corr = binom.confint(pos_corr, n, methods = "wilson")
  ) |>
  mutate(
    seroprev_corr = ci_corr$mean * 100,
    lower_CI_corr = ci_corr$lower * 100,
    upper_CI_corr = ci_corr$upper * 100
  ) |> arrange(clsuter_type_corr, age_group)

# Line plot
data.mapAgeP_corr$clsuter_type_corr <- factor(
  data.mapAgeP_corr$clsuter_type_corr,
  levels = c("High", "Medium", "Low")  # specify desired order
)

age_prevalence_district <-
ggplot(data.mapAgeP.all, aes(x = age_group, y = seroprev_corr, group = 1)) +
  geom_line(color = "#669bbc", size = 1.2) +
  geom_point(color = "#c1121f", size = 3) +
  geom_errorbar(
    aes(ymin = lower_CI_corr, ymax = upper_CI_corr),
    width = 0.15,
    color = "#c1121f",
    size = 1
  ) +
  #facet_wrap(~ clsuter_type, ncol = 1, scales = "free_y") +  # ⬅ free Y, stacked vertically
  labs(x = NULL, y = NULL) +
  theme_classic(base_size = 22) +
  theme(
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 20),
    panel.border = element_rect(color = "black", fill = NA)
  )

age_prevalence_classifed <-
ggplot(data.mapAgeP_corr, aes(x = age_group, y = seroprev_corr, group = 1)) +
  geom_line(color = "#669bbc", size = 1.2) +
  geom_point(color = "#c1121f", size = 3) +
  geom_errorbar(
    aes(ymin = lower_CI_corr, ymax = upper_CI_corr),
    width = 0.15,
    color = "#c1121f",
    size = 1
  ) +
  facet_wrap(~ clsuter_type_corr, ncol = 1, scales = "free_y") +  # ⬅ free Y, stacked vertically
  labs(x = NULL, y = NULL) + 
  theme_classic(base_size = 22) +
  theme(
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 20),
    panel.border = element_rect(color = "black", fill = NA)
  )

library(cowplot)

# Combine plots side by side
combined_plot <- plot_grid(
  age_prevalence_district,
  age_prevalence_classifed,
  labels = c("A", "B"),     # panel labels
  label_size = 22,
  ncol = 2,                 # number of columns
  align = "hv",             # align horizontally and vertically
  axis = "lrbt"             # align all axes (left, right, bottom, top)
)

# Add global axis labels
x_lab <- ggdraw() + 
  draw_label("Age Category (years)", size = 22, x = 0.5, hjust = 0.5)

y_lab <- ggdraw() + 
  draw_label("Seroprevalence (%)", angle = 90, size = 22, x = 0.5, hjust = 0.5)

# Assemble everything (title + plots + axis labels)
final_figure <- plot_grid(
  plot_grid(y_lab, combined_plot, ncol = 2, rel_widths = c(0.025, 1)),
  x_lab,
  ncol = 1,
  rel_heights = c(1, 0.05)
)

# Display the final figure
print(final_figure)
ggsave(
  filename = "./FIGURES/age_seroprevalence.png",
  plot = final_figure,
  bg = "white",
  width = 16,      # in inches
  height = 9,      # in inches
  dpi = 600        # resolution (suitable for publication)
)

# plot occupation 
data.mapOccP <- df_plot_age.prev %>% dplyr::select(clsuter_type, occup_princip_risk, WN_ENVandNS1_Sero_3sd) %>%
  group_by(clsuter_type, occup_princip_risk) %>%
  summarise(
    n = n(),
    pos = sum(WN_ENVandNS1_Sero_3sd),
    .groups = "drop"
  ) %>%
  mutate(
    ci = binom.confint(pos, n, methods = "wilson")
  ) %>%
  mutate(
    seroprev = ci$mean * 100,
    lower_CI = ci$lower * 100,
    upper_CI = ci$upper * 100
  ) %>% arrange(clsuter_type, occup_princip_risk)

library(ggplot2)
data.mapOccP$clsuter_type <- factor(
  data.mapOccP$clsuter_type,
  levels = c("High", "Medium", "Low")  # specify desired order
)

#occ_prevalence_classifed <-
ggplot(data.mapOccP, aes(x = as.factor(occup_princip_risk), y = seroprev, group = 1)) +
  geom_line(color = "#669bbc", size = 1.2) +
  geom_point(color = "#c1121f", size = 3) +
  geom_errorbar(
    aes(ymin = lower_CI, ymax = upper_CI),
    width = 0.15,
    color = "#c1121f",
    size = 1
  ) +
  facet_wrap(~ clsuter_type, ncol = 1, scales = "free_y") +  # ⬅ free Y, stacked vertically
  labs(x = NULL, y = NULL) + 
  theme_classic(base_size = 22) +
  theme(
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold", size = 20),
    panel.border = element_rect(color = "black", fill = NA)
  )

# Plot bar Occupation 
occ.more14years <- data.model_cutoffs[data.model_cutoffs$age <= 14 ,] #& !is.na(data.model_cutoffs$occup_princip)
occ.more14years$occup_princip <- as.factor(occ.more14years$occup_princip)

library(forcats)  # for fct_infreq
ggplot(occ.more14years, aes(x = fct_infreq(occup_princip))) +
  geom_bar(fill = "#74add1", color = "black", width = 0.7) +
  labs(
    title = "Number of individuals (14 years and older) by main occupation",
    x = "Main Occupation Category",
    y = "Count"
  ) +
  theme_classic(base_size = 22)+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)  # tilt x-axis labels
  )
