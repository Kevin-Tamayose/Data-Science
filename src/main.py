import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

FRAC_AMOSTRA = 0.4
RANDOM_SEED = 42
paises_americas = ['ARG', 'BRA', 'CAN', 'CHL', 'COL', 'CRI', 'DOM', 'SLV', 'GTM', 'MEX', 'PAN', 'PRY', 'PER', 'USA', 'URY']

# 1. Lendo os dados do ALUNO (Student Questionnaire)
cols_student = ['CNT', 'CNTSCHID', 'CNTSTUID', 'W_FSTUWT']
df_student = pd.read_parquet('data/parquet/CY08MSP_STU_QQQ.parquet', columns=cols_student)
print(f"Sucesso: {df_student.shape[0]} linhas carregadas.")

# 2. Lendo os dados da ESCOLA (School Questionnaire)
cols_school = ['CNT', 'CNTSCHID', 'SC001Q01TA', 'STRATUM']
df_school = pd.read_parquet('data/parquet/CY08MSP_SCH_QQQ.parquet', columns=cols_school)
print(f"Sucesso: {df_school.shape[0]} linhas carregadas.")

# 3. Lendo os dados do PROFESSOR (Teacher Questionnaire)
# cols_teacher = ['CNT', 'CNTSCHID', 'TC001Q01NA', 'TC002Q01NA']
# df_teacher = pd.read_parquet('data/parquet/CY08MSP_TCH_QQQ.parquet', columns=cols_teacher)
# print(f"Sucesso: {df_teacher.shape[0]} linhas carregadas.")

# Isso vai forçar o Pandas a mostrar se há LINHAS de fato
print(f"Linhas no Student: {len(df_student)}")
print(f"Linhas no School: {len(df_school)}")


# Subamostragem
print(df_student.info())
print(df_student.describe())

rng = np.random.RandomState(42)

def weighted_sample_robust(group, frac):
    n = int(len(group) * frac)
    if n == 0: return group.head(0)
    ui = rng.rand(len(group))
    ki = ui ** (1 / group['W_FSTUWT'])
    
    return group.loc[ki.nlargest(n).index]

df_amostrado = df_student.groupby('CNT', group_keys=False).apply(
    lambda x: weighted_sample_robust(x, FRAC_AMOSTRA)
)
print(f"Linhas no Student Original: {len(df_student)}")
print(f"Linhas no Student Amostrado: {len(df_amostrado)}")