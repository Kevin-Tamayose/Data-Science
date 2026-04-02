import pandas as pd 
import pyreadstat

def transform_sav_to_parquet(path, encoding='LATIN1', columns:list = []):
    if len(columns) > 0:
        df, meta = pyreadstat.read_sav(path, encoding=encoding, usecols=columns)
    else:
        df, meta = pyreadstat.read_sav(path, encoding=encoding)

    path = path.replace('/sav/', '/parquet/')
    df.to_parquet(path.replace('.SAV', '.parquet'), engine='pyarrow', compression='snappy')
    return df, meta

# 1. Lendo os dados do ALUNO (Student Questionnaire)
df_student, meta_student = transform_sav_to_parquet('data/sav/CY08MSP_STU_QQQ.SAV')

# 2. Lendo os dados da ESCOLA (School Questionnaire)
df_school, meta_school = transform_sav_to_parquet('data/sav/CY08MSP_SCH_QQQ.SAV')

# 3. Lendo os dados do PROFESSOR (Teacher Questionnaire)
# df_teacher, meta_teacher = transform_sav_to_parquet('data/sav/CY08MSP_TCH_QQQ.SAV')
