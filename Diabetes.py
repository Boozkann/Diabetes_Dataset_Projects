import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
import seaborn as sns

from scipy.optimize import anderson
from skimage.feature import shape_index
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def diabetes_load():
    data = pd.read_csv("/Users/ozkan/OneDrive/Desktop/Feature_Engineering/Diabetes.csv")
    return data


df = diabetes_load()
df.head()

##########################################################################################
# 1. Outliers
##########################################################################################

##########################################################################################
# Aykırı Değerleri Yakalama : Doğrusal modellerde kullanılabilecek etkili bir yöntemdir.
# 4 Şekilde yakalanabilir
##########################################################################################

# 1. Domain Knowledge : Sektör bilgisiyle hızlıca çözülebilir.

# 2. Standard Deviation : %5 std dev belirlersek 100'lük bir veri 95-105 tahmin aralığında değer alır.

# 3. Z Score : Bir çan eğrisinde 0 ortalama yönetmi ile aykırı değerleri buluruz.

# 4. Boxplot : (Interquantile range) En çok kullanılan yöntem olarak; min ve max dışındaki değerler aykırı değerdir.

############################################################################################
# Grafik Teknikle Outlier Analizi
############################################################################################

sns.boxplot(x=df["Glucose"])
plt.show()


# 50 civarı q1, 200 de q3, bu sınırların dışı ise aykırı değerlerdir.

sns.boxplot(x=df["Insulin"])
plt.show()

# 4 çeyrek arasında 3 sınır bulunur. dolayısıyla 1-3 dışında kalanlar aykırı değerlerdir.

q1 = df["Insulin"].quantile(0.25)
q3 = df["Insulin"].quantile(0.75)

# Domain knowledga göre aykırı değerler 0.25-0.75 yerine farklı sınırlar belirlenebilir.

iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Insulin"] < low) | (df["Insulin"] > up)].head()     # | veya

df[(df["Insulin"] < low)]

df[(df["Insulin"] > up)]

df[(df["Insulin"] < low) | (df["Insulin"] > up)].index


##############################################################################
# Aykırı Değer Var mı Yok mu?
##############################################################################

df[(df["Insulin"] < low) | (df["Insulin"] > up)].any(axis=None)

df[(df["Insulin"] < low)].any(axis=None)

# Fonksiyon oluşturursak :

def thresholds_outliers(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return float(low_limit), float(up_limit)

thresholds_outliers(df, "Insulin")
thresholds_outliers(df, "Glucose")


low, up = thresholds_outliers(df, "Insulin")

low, up

def outlier_check(dataframe, col_name):
    low_limit, up_limit = thresholds_outliers(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

outlier_check(df, "Glucose")
outlier_check(df, "Insulin")


###############################################################################
# col_names_grab      cat, num, cat_but_car vs.
###############################################################################

def col_names_grab(dataframe, cat_th=10, car_th=20):

# cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
# num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = col_names_grab(df)

cat_cols
num_cols
cat_but_car

df.head()

num_cols = [col for col in num_cols if col not in "Outcome"]

## Outcome sütunu etiket gibi olduğu için numerik sütunlarımız arasından kaldırdık.

num_cols


outlier_check(df, "Insulin")  ## böyle tek tek yapmak yerine for döngüsü kullanırız

for col in num_cols:
    print(col, outlier_check(df, col))


###################################################
# Aykırı Değerlerin Kendilerine Erişmek
###################################################

def outliers_grab(dataframe, col_name, index=False):
    low, up = thresholds_outliers(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


outliers_grab(df, "Insulin")
outliers_grab(df, "Insulin", index=True)
age_index = outliers_grab(df, "Age", index=True)


thresholds_outliers(df, "Insulin")
outlier_check(df, "Insulin")
outliers_grab(df, "Insulin", True)


##################################################
# Aykırı Değer Problemini Çözme
##################################################

##################################################
# Silme
###################################################

low, up = thresholds_outliers(df, "Insulin")
df.shape

df[~((df["Insulin"] < low) | (df["Insulin"] > up))].shape

## Bunu fonksiyona dönüştürürsek;

def outlier_remove(dataframe, col_name):
    low_limit, up_limit = thresholds_outliers(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = col_names_grab(df)

num_cols = [col for col in num_cols if col not in "Outcome"]

df.shape

for col in num_cols:
    new_df = outlier_remove(df, col)

new_df.shape

df.shape

df.shape[0] - new_df.shape[0]

#######################################################
# Baskılama Yöntemi (re-assignment with thresholds)
#######################################################

low, up = thresholds_outliers(df, "Insulin")


df[((df["Insulin"] < low) | (df["Insulin"] > up))]
df[((df["Insulin"] < low) | (df["Insulin"] > up))]["Insulin"]

df.loc[((df["Insulin"] < low) | (df["Insulin"] > up)), "Insulin"]


df["Insulin"] = df["Insulin"].astype(float)
df.loc[(df["Insulin"] > up), "Insulin"] = up
df.loc[(df["Insulin"] < low), "Insulin"] = low


def replace_as_thresholds(dataframe, variable):
    low_limit, up_limit = thresholds_outliers(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_check(df, "Insulin")  ## False geldi, outlier kalmadı

for col in num_cols:
    print(col, outlier_check(df, col))

for col in num_cols:
    print(col, replace_as_thresholds(df, col))

for col in num_cols:
    print(col, outlier_check(df, col))

 ##Burada for döngüsüyle bütün veri setine baskılama uyguladık ve outlier'lar limitle replace edildi.


############################################################
# Recap Yontemi
############################################################

df = diabetes_load()
thresholds_outliers(df, "Insulin")
outlier_check(df, "Insulin")
outliers_grab(df, "Insulin", True)

outlier_remove(df, "Insulin").shape
replace_as_thresholds(df, "Insulin")
outlier_check(df, "Insulin")


############################################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
############################################################

df = diabetes_load()
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()

df.head()
df.shape

############################################################
# Threshold Belirleme
############################################################

low, up = thresholds_outliers(df, "Insulin")

df[((df["Insulin"] < low) | (df["Insulin"] > up))].shape


low, up = thresholds_outliers(df, "Glucose")

df[((df["Glucose"] < low) | (df["Glucose"] > up))].shape

##### LOF #################################################

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_   ### Uzaklık değerlerine bakalım ###
df_scores[0:5]
## -df_scores  başına (-) koyarak negatif gelen mesafeleri pozitif çıkarırız.

np.sort(df_scores)[0:5]

### Eşik Değer Seçme ###

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th =np.sort(df_scores)[5]

df[df_scores < th]

df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis = 0, labels= df[df_scores < th].index)

##########################################################
# Eksik Değer Analizi
##########################################################

df = diabetes_load()
df.head()
df.isnull().values.any()

df.isnull().sum()

df.notnull().sum()

df.isnull().sum().sum()

#Visualization

for col in num_cols:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    sns.histplot(df[col], kde=True, ax=axs[0], color="skyblue")
    axs[0].set_title(f"{col} - Histogram")

    sns.boxplot(x=df[col], ax=axs[1], color="lightgreen")
    axs[1].set_title(f"{col} - Boxplot")

    plt.tight_layout()
    plt.show()


####################################

counts = df['Outcome'].value_counts(normalize=True).reset_index()
counts.columns = ['Outcome', 'Percent']
counts['Count'] = df['Outcome'].value_counts().values
counts['Outcome'] = counts['Outcome'].map({0: 'Negative', 1: 'Positive'})

sns.barplot(data=counts, x='Outcome', y='Count', palette='Set2')

for i, row in counts.iterrows():
    plt.text(i, row['Count'] + 5, f"{row['Percent']*100:.1f}%",
             ha='center', fontsize=10)

plt.title('Distribution of Outcome Variable')
plt.tight_layout()
plt.show()


####################################
# Statistics #######################
####################################

summary_stats =  df.groupby("Outcome").agg(["mean", "count"]).reset_index()
summary_stats

mean_values = summary_stats.xs('mean', axis=1, level=1).T

plt.figure(figsize=(14, 9))
mean_values.plot(kind='bar', width=0.75, cmap='Set2')
plt.title('Statistics Outcome')
plt.ylabel('Mean')
plt.xticks(rotation=45, fontsize=9)
plt.legend(title='Outcome', labels=['0', '1'])
plt.tight_layout()
plt.show()


## Correlation ##########################

plt.figure(figsize=(12, 5))
plt.title('General Correlation', fontsize = 10, fontweight='bold')
sns.heatmap(df[num_cols].corr(), cmap='Set3', annot=True)
plt.xticks(fontsize=9, rotation=15)
plt.yticks(fontsize=9, rotation=15)
plt.tight_layout()
plt.show()


## Missing Values and Imputation#########

null_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in null_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

null_columns


df.isnull().sum().sort_values(ascending=False)

msno.bar(df, fontsize=8, figsize=(5,2), color="salmon");
plt.show()

msno.matrix(df[null_columns], fontsize=8, figsize=(6,3), color=(0.7, 0.4, 0.1));
plt.tight_layout()
plt.show(block=True)


df.isnull().sum().sort_values(ascending=False)


msno.matrix(df, fontsize=8, figsize=(6,3), color=(0.7, 0.4, 0.1))
plt.tight_layout()
plt.show(block=True)

msno.heatmap(df, fontsize=8, figsize=(6,3), cmap='Set2')
plt.tight_layout()
plt.show()

sns.histplot(data=df, x="Glucose", bins=30, kde=True, color="salmon", edgecolor='grey')
plt.show()

df["Glucose"] = df.groupby("Outcome")["Glucose"].transform(lambda x : x.fillna(x.mean()))
df["Glucose"].isnull().sum()



sns.histplot(data=df, x="BMI", bins=30, kde=True, color="salmon", edgecolor='grey')
plt.show()


df["BMI"] = df.groupby("Outcome")["BMI"].transform(lambda x : x.fillna(x.median()))
df["Glucose"].isnull().sum()


# 1. encoder yapma
# 2. standartlastirma yapmak
# 3. makina ogrenmesi

df = diabetes_load()

cols_for_knn = ["BloodPressure", "BMI", "Age", "Glucose", "Insulin", "SkinThickness"]
df_knn = df[cols_for_knn].copy()


# değişkenlerin standartlatırılması
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_knn), columns=cols_for_knn)



# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=cols_for_knn)
df_imputed.head()



# scaler yapildiktan sonra eski haline getirme islemi
df_inverse = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=cols_for_knn)


df["BloodPressure"] = df_inverse["BloodPressure"]
df["BMI"] = df_inverse["BMI"]
df["Age"] = df_inverse["Age"]
df["Insulin"] = df_inverse["SkinThickness"]
df["SkinThickness"] = df_inverse["SkinThickness"]

df[["BloodPressure", "BMI", "Age", "Glucose", "Insulin", "SkinThickness"]].isnull().sum()


df[num_cols].describe()








