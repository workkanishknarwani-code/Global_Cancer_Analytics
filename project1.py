import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sklearn; print(sklearn.__version__)
from scipy.stats import linregress, pearsonr, spearmanr
from scipy.stats import shapiro, f_oneway
from scipy.stats import kruskal

warnings.filterwarnings("ignore")


data = pd.read_csv(r"D:\python\global_cancer_patients_2015_2024 (1).csv")

#DESCRIPTIVE ANALYSIS


#plotting fo age coloumn

print(data["Age"].describe())

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.kdeplot(data["Age"], fill=True, color="lightgreen")
plt.title("KDE plot for Age")

plt.subplot(1,2,2)
sns.histplot(data["Age"], bins=30, kde=False, color="cyan")
plt.title("Histogram plot for Age")

plt.tight_layout()
plt.show()

#Figure_1

#INFERENCE


#This represents a broad representation of both young and elderly 
#patients in the data set, which suggests age based comparative analysis

#plotting fo gender coloumn

print(data["Gender"].value_counts())


sns.barplot(
    x=data["Gender"].value_counts().index,
    y=data["Gender"].value_counts().values,
    palette=["blue", "pink", "green"]
)

for i, v in enumerate(data["Gender"].value_counts()): 
	plt.text(i, v, str(v), ha="center", va="bottom")

plt.title("Gender count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

#Figure_2 Updated
#INFERENCE

#This dataset contains three gender catgories(male, female and others), with the most common being male(16976).
#Gender distribution is sufficient for evaluating gender specific survival trends and severity outcomes


#Countries coloumn

country_counts = data["Country_Region"].value_counts()

print(country_counts)

plt.figure(figsize=(5,5))
plt.pie(
    country_counts.values,
    labels=country_counts.index,
    autopct='%1.1f%%'
)
plt.title("Country/Region Distribution")
plt.show()

#Figure_3
# INFERENCE

# Patients come from 10 different countries/Regions, with Australia being the most represented (5092 patients). Number of 
# data points from each country is almost same. This dicersity enables cross-country comparison and treatment economic.



# For Cancer type

print(data["Cancer_Type"].value_counts())
sns.barplot(
		x=data["Cancer_Type"].value_counts().index,
		y=data["Cancer_Type"].value_counts().values
)
for i, v in enumerate(data["Cancer_Type"].value_counts()):
 	plt.text(i, v, str(v))

plt.title("Cancer type count")
plt.xlabel("Cancer type")
plt.ylabel("Count")
plt.show()

#Figure_4

#INFERENCE
#We have in total 8 types of cancer, with each cancer having aprrox. same no. of data points under the label, 
#most common cancers are colon cancer followed by prostate cancer.



#For cancer stage

print(data["Cancer_Stage"].value_counts())
sns.barplot(
		x=data["Cancer_Stage"].value_counts().index,
		y=data["Cancer_Stage"].value_counts().values
)
for i, v in enumerate(data["Cancer_Stage"].value_counts()):
 	plt.text(i, v, str(v))

plt.title("Cancer stage count")
plt.xlabel("Cancer_Stage")
plt.ylabel("Count")
plt.show()

#Figure_5

#INFERENCE

#Cancer stage have five stages with most values ranging from 0 to 4, with stage 2 the most common one, and 
#each stage have same number of data points under it's label. 



#For Treatment cost(in USD)

print(data["Treatment_Cost_USD"].describe())

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.kdeplot(data["Treatment_Cost_USD"], fill=True, color="lightgreen")
plt.title("KDE plot for Treatment_Cost_USD")

plt.subplot(1,2,2)
sns.histplot(data["Treatment_Cost_USD"], bins=30, kde=False, color="cyan")
plt.title("Histogram plot for Treatment_Cost_USD")

plt.tight_layout()
plt.show()

print(data["Treatment_Cost_USD"].describe())
#Figure 6

#INFERENCE

#Treatment cost have no skewness ans there are almost same no. of data points 
#under each bin as observed by histogram.



#Analyzing the risk factors
print(data.columns)
columns_of_interest = ['Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']

summary = print(data[columns_of_interest].agg(["mean", "std", "max", "min"]))
summary

#INFERENCE

#These variables have nearly identical means and std deviation, indicating they were likely designed on the same 
#standardized scale. They are essential in studying interaction effects (eg: genetic risks * smoking) on survival

 
#Determine the relationship between risk factors and cancer sensitivity

Risk_Factors = ['Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
Titles = ['Genetic Risk', 'Air Pollution', 'Alcohol Use', 'Smoking', 'Obesity Level']
colors = ["blue", "green", "orange", "red", "purple"]

plt.figure(figsize=(20,12))
for i , (factor, title, color) in enumerate(zip(Risk_Factors, Titles, colors), 1):
	plt.subplot(2,3,i)

	x=data[factor]
	y=data["Target_Severity_Score"]
	slope, intercept, r_value, p_value, std_err= linregress(x,y)
	r_squared = r_value**2

	sns.lineplot(x=factor, y="Target_Severity_Score", data=data, color=color)
	plt.plot(x, x*slope + intercept, color="black", linewidth=2, label=	"Regression Line")
	plt.title(f"{title} vs Severiy Score\n R2 = {r_squared}, slope= {slope}")
	plt.xlabel(factor)
	plt.ylabel("Target_Severity_Score")
	plt.legend()
plt.tight_layout()
plt.show()

#Figure_7

#INFERENCE
#To understand the contribution of various risk factors to cancer severity, line plots were generated for five primary variables: Genetic risk,
#Air Pollution, Alcohol Use, Smoking, Obesity Level plotted against the target severity score.

#Genetic risk vs Target severity score:

#R^2 =  0.23; A weak linear relationship. only 23% of variability in Target_Severity_Score is explained by Genetic risk. This suggests that other factors
#likely play a larger role in influencing the severity score.

#Slope = 0.20; A positive slope indicates that as Genetic Risk increases the Target_Severity_Score also tends to increase. For each unit increase in Genetic risk,
#Target_Severity_Score increases by 0.20 units. However the R^2 is relatively low, this trend is not very consistent with across the data



#A the Proportion of early-stage diagnoses by cancer type
 
print(data["Cancer_Type"].unique())

#For Lung;
stage_count = data[data["Cancer_Type"]=="Lung"]["Cancer_Stage"].value_counts()

early_stage_sum = stage_count.get("Stage 0", 0) + stage_count.get("Stage I", 0)
total_sum = stage_count.sum()

proportion = (early_stage_sum / total_sum) * 100

print(f"Proportion of lung cancer diagnosed at stage 0 and stage 1: {proportion:.2f}%")

#For Skin

print(data["Cancer_Type"].unique())

stage_count = data[data["Cancer_Type"]=="Skin"]["Cancer_Stage"].value_counts()

early_stage_sum = stage_count.get("Stage 0", 0) + stage_count.get("Stage I", 0)
total_sum = stage_count.sum()

proportion = (early_stage_sum / total_sum) * 100

print(f"Proportion of lung cancer diagnosed at stage 0 and stage 1: {proportion:.2f}%")

#INFERENCE 
#The analysis demonstrates that early_stage diagnosis for various cancer types is relatively widespread, with most cancers having an early diagnosis rate bw
#38.43% and 40.41%. Liver cancer shows the highest proportion, while lung cancer shows the lowest. These findings suggest that while screening and diagnostic 
#methods are effective, improvements can still be made, particularly in lung cancer detection.

#Further research into screening strategies, early intervention, and the use of advanced diagnostic technologies could help increase the proportion of   
#early_stage diagnosis, ultimately lending to better survival rates and outcomes for cancer patients. The relatively small variations across the cancer types 
#indiate that, in general, healthcare systems may need to focus on enhancing early detection uniformly, with targeted efforts to address specific gaps in 
#detection, particularly, for cancers like lung cancer. 

#Identify Key Predictors of Cancer Severity and Survivaal Years

Features = ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
Target = ["Survival_Years", "Target_Severity_Score"] 


#Calculating Correlations
pearson_corr = data[Features + Target].corr(method="pearson")
print(pearson_corr)

spearman_corr = data[Features + Target].corr(method="spearman")
print(spearman_corr)



#Slicing out the relationship with only target variables
x = pearson_corr[Target]
print(x)

y = spearman_corr[Target]
print(y)


#Combining both 
correlation_df = pd.concat(
    [x.add_prefix("Pearson_"), y.add_prefix("Spearman_")],
    axis=1
)
print(correlation_df)

#INFERENCE

#For Severiy_Score -> Since value of Pearson's and Spearman's coefficient for Smoking is 0.484 and 0.478 it shows strong positive correlation. Higher smoking levels are associated with higher
#severity score.

#For Survival_Years -> Since value of Pearson's and Spearman's coefficient for almost 0.00 that means negligible or negative correlation. These variables do not show any significant linear
#or monotonic association with survival duration. 

#WE WILL DO SAME ANALYSIS USING MACHINE LEARNING ALGORITHMS (RandomForest)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import r2_score


#FOR TARGET_SEVERITY_SCORE

#Converting categorical coloumns to numerical coloumns

Categorical_Cols = ["Gender", "Country_Region", "Cancer_Type", "Cancer_Stage"]
for col in Categorical_Cols:
	le=LabelEncoder()
	data[col]=le.fit_transform(data[col])

#Preparing features and input 

X = data.drop(columns=[
    "Patient_ID",
    "Survival_Years",
    "Target_Severity_Score",
    "Treatment_Cost_USD"
])

Y_Severity = data["Target_Severity_Score"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_Severity, test_size=0.2, random_state=42
)

#Train test split 
X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(X,Y_Severity, test_size=0.2, random_state=42)

#Train the Model 

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train_s, Y_train_s)

#Evaluating the Model
print("train_r2_Severity:", r2_score(Y_train_s, model.predict(X_train_s)))
print("test_r2_Severity:", r2_score(Y_test_s, model.predict(X_test_s)))

#train_r2_Severity: 0.9689886894803543
#test_r2_Severity: 0.7683892517466494

feature_importance_severity = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=True)

#Plotting for important features 
plt.figure(figsize=(10,6))
feature_importance_severity.plot(kind="bar", color="skyblue")
plt.title("Feature Importance for Target Severity Score (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Impotance Score")
plt.xticks(rotation= 45)
plt.tight_layout()
plt.show()

#Figure_8

#INFERENCE 
#This graph shows variation caused by a particular feature while it is combined with other features.
#smoking 0.2336 Most imp predictor of severity score. The more a patient smokes, the higher severity tends to be.
#genetic risk 0.2286 strong genetic predisposition is nearly as important as smoking.
#Treatment cost usd 0.2133 Higher treatment costs are associated with more severe conditions.
#Alchohol use 0.1291 Alchohol plays a significant role 
#Air pollution 0.1271 Enviromental factor_patients in more polluted areas have worse severity scores.
#Obesity level 0.0573 has an effect, but much smaller.
#Age to Gender <0.01 very low importance, these don't explain much variation in severity score 



#FOR SURVIVAL_YEARS

Categorical_Cols = ["Gender", "Country_Region", "Cancer_Type", "Cancer_Stage"]
for col in Categorical_Cols:
	le=LabelEncoder()
	data[col]=le.fit_transform(data[col])

#Preparing features and input 

X = data.drop(columns=[
    "Patient_ID",
    "Survival_Years",
    "Target_Severity_Score",
    "Treatment_Cost_USD"
])


y = data["Target_Severity_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=40
)

param_grid = {
	'n_estimators': [100,200],
	'max_depth': [5, 10, None],
	'min_samples_split': [2, 5],
	'min_samples_leaf': [1, 2] 
}

#Train the Model 

base_model = RandomForestRegressor(random_state=40)

GSC = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

GSC.fit(X_train, y_train)

best_rf_severity = GSC.best_estimator_

train_r2 = r2_score(
    y_train,
    best_rf_severity.predict(X_train)
)

test_r2 = r2_score(
    y_test,
    best_rf_severity.predict(X_test)
)

print("Train R2:", train_r2)
print("Test  R2:", test_r2)

sns.histplot(data["Survival_Years"], kde=True)
plt.title("Distribution of Survival Years")
plt.show()

#Figure_9

print(data.corr(numeric_only=True)["Survival_Years"].sort_values(ascending=True))

#INFERENCE
#The information in your coloumns does not help the model to figure out how long a person will survive 



#Explore the economic burden of cancer treatment across different demographics and countries 

print(data.groupby(["Country_Region","Gender"])["Treatment_Cost_USD"].mean().reset_index())

data["Age_Group"]=pd.cut(data["Age"],bins=[0,30,45,60,75,100],labels=["0-30","31-45","46-60","61-75","76+"])

#pd.cut converts numeric coloumns into intervals

Country_Age_Cost = data.groupby(["Country_Region", "Age_Group", "Gender"])["Treatment_Cost_USD"].mean().reset_index()
plt.figure(figsize=(10,6))
sns.barplot(
    data=Country_Age_Cost,
    x="Country_Region",
    y="Treatment_Cost_USD",
    hue="Gender"
)
plt.title("Average Treatment Cost by Country, Age Group and gender")
plt.show()

#Figure_10

country_age_cost = data.groupby(["Country_Region", "Age_Group"])["Treatment_Cost_USD"].mean().reset_index()
heatmap_data = country_age_cost.pivot(index="Age_Group", columns="Country_Region", values="Treatment_Cost_USD")
plt.figure(figsize=(10,6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f")
plt.title("Average treatment cost by age group and country")
plt.show()

#Figure_11

#INFERENCE

#Geographic Disparities in Economic Burden:
#Cancer treatment costs are significantly higher in developed nations such as the USA, Australia, and China, revealing the heavy financial load in advanced healthcare systems.
#Meanwhile, countires like India and Pakistan exhibit comparatively lower costs, likely due to lower healthcare pricing structures or limited access to advanced treatment.
#This highlights a clear global inequality in healthcare affordability that can intensify financial strain depending on a patient's country or residence

#Gender Based Cost Patterns are Uniform:
#Across all countries, gender-based differences in average treatment costs are minimal, suggesting no major gender bias in pricing or access to cancer care. This uniformity in  
#may reflect standardization in treatment protocols or equitable healthcare policies, but it also points to the fact that the financial impact of cancer is universal across genders.

#Age-Related Escalation in Treatment Costs:
#Treatment costs tend to rise with age, particulary for those aged 61 above. This trend is espacially evident in countries like Australia and the USA, where older age groups face
#sharply higher costs. The increased financial burden in these groups could be due to more intensive care needs, multiple comorbidities, or prolonged treatments. This pattern underlines
#the vulnerability of elderly populations and the pressing need for targeted support for senior citizens..

#Role of Healthcare Systems in cost variation:
#Countries with robust public healthcare systems such as Canada, Germany, and the UK; show relatively stable treatment costs across age groups, reflecting the benefits
#of healthcare subsidies or coverage. This consistency reinforces the importance of Government intervention and Universal healthcare in mitigating financial disparities in
#cancer treatment. 



#ASSESING WHETHER HIGHER TREATMENT COST IS ASSOCIATED WITH LONGER SURVIVAL

#Null Hypothesis: There is no correlation between treatment cost and survival years
#Alternate Hypothesis: There is correlation(positive or negative) between treatment cost and survival years.

x = data["Treatment_Cost_USD"]
y = data["Survival_Years"]

#Performing pearson correlation test

pearson_corr, pearson_p = pearsonr(x,y)
print(f"Pearson Correlation Cofficient: {pearson_corr}")
print(f"Pearson P-Value: {pearson_p}")

#Performing spearman correlation test

spearman_corr, spearman_p = spearmanr(x,y)
print(f"spearman Correlation Cofficient: {spearman_corr}")
print(f"spearman P-Value: {spearman_p}")

alpha = 0.05

def interpret_corr(corr, p, method):
    if p<alpha:
        print(f"{method}, we reject the null hypothesis")
    else:
        print(f"{method}, we failed to reject the null hypothesis")

interpret_corr(pearson_corr, pearson_p, "Pearson")
interpret_corr(spearman_corr, spearman_p, "Spearman")

sns.regplot(x=x, y=y, line_kws={"color":"red"})
plt.show()

#Figure_12


#INFERENCE

#According the coefficient we can observe that there is no correlation between treatment cost and suvival years. 
#From Hypothesis testing also we can conclude that there is no correlation between treatment cost and suvival years.


#EVALUATING IF HIGHER CANCER STAGES LEADS TO GREATER TREATMENT COSTS AND REDUCED SURVIVAL YEARS

df = data

Stage_Order = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
Group_Stats = df.groupby("Cancer_Stage")[["Treatment_Cost_USD","Survival_Years"]].mean().reset_index() 
#print(Group_Stats)

#There is no change in treatment cost as well as average survival years is stage increases

#Let's do the same viz Hypothesis testing 

#Treatment cost vs cancer stage
#Null Hypothesis: The average treatment cost is same across all cancer stages i.e normally distributed.
#Alternate Hypothesis: Atleast one stage has a differen average cost i.e not normally distrbuted.



#Survival years vs cancer stage
#Null Hypothesis: The average survival years are same across all cancer stages i.e normally distributed.
#Alternate Hypothesis: Atleast one stage has a different survival duration i.e not normally distrbuted.

grouped_costs = []
grouped_survival = []

for stage in Stage_Order:
    stage_data = df[df["Cancer_Stage"]==stage]
    cost = stage_data["Treatment_Cost_USD"]
    Survival = stage_data["Survival_Years"]
    grouped_costs.append(cost)
    grouped_survival.append(Survival)

#Checking whether the groups are norally distrubuted or not 

normal_survival = 0
normal_cost = 0

for i in range (len(Stage_Order)):
    cost_p= shapiro(grouped_costs[i]).pvalue
    surv_p= shapiro(grouped_survival[i]).pvalue
    print(f"cost{cost_p} for group [i]")
    print(f"survival{surv_p} for group [i]")
    if cost_p < 0.05:
        normal_cost+=1
    if cost_p < 0.05:
        normal_survival+=1
print(normal_survival)
print(normal_cost)

#Since p value is approx. 0 i.e less than 0.05 we can reject null hypothesis.

#Let's check this from kruskal wallis test

kruskal_cost = kruskal(*grouped_costs)
kruskal_Survival = kruskal(*grouped_survival)

p_cost = kruskal_cost.pvalue
p_Survival = kruskal_Survival.pvalue

print(p_cost)
print(p_Survival)

#INFERENCE

# p - value = 0.42; No significant difference in treatment cost among cancer stages.

# p - value = 0.60; No significant difference in survival years among cancer stages



#EXAMINING WHETHER HIGHES GENETIC RISK AMPLIFIES THE NEGATIVE EFFECTS OF SMOKING ON CANCER SEVERITY

#Null Hypothesis: The interaction effect between genetic risk and smoking on Cancer Severity is not significant.
#               (Genetic risk does not amplify or alter the effect of smoking)

#Alternate Hypothesis: The interaction effect between genetic risk and smoking on Cancer Severity is significant.
#                   (Genetic risk does amplify or alter the effect of smoking)                                              

import statsmodels.formula.api as smf

model = smf.ols("Target_Severity_Score ~ Genetic_Risk*Smoking", data=data).fit()
print(model.summary2().tables[1].loc["Genetic_Risk:Smoking"])

#INFERENCE

#Test used: MLR with interaction term
#The interaction coefficient is very small (-ve)
#Since p value = 0.62 i.e greater than 0.05, we do not reject null hypothesis.
#The interaction effect between genetic risk and smoking on target severity score is not statistically significant ( p value > 0.05)
#This means that based on your data, there is no evidence that genetic risk amplifies or reduces the effect of smoking on the target severity score
#In other words, smoking and genetic risk may each have independent effects(or none), but they do not interact in a way that significantly change the outcome. 






















