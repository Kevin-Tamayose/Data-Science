import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

FRAC_AMOSTRA = 0.2
RANDOM_SEED = 42
paises_americas = ['ARG', 'BRA', 'CAN', 'CHL', 'COL', 'CRI', 'DOM', 'SLV', 'GTM', 'MEX', 'PAN', 'PRY', 'PER', 'USA', 'URY']

# 1. Lendo os dados do ALUNO (Student Questionnaire)
# sexo 1 feminino, 2 masculino

# Colocar no documento que algumas perguntas vem da interpretação do estudante e portanto podem ser enviesadas, como por exemplo a percepção socioeconômica e a relação com professores e escola.
cols_student = ['CNT', 'CNTSCHID', 'CNTSTUID', 'W_FSTUWT',
                # --- IDENTIFICAÇÃO ---
                # sexo 1 feminino, 2 masculino
                'ST004Q01TA',
                # Ano escolar
                'ST001Q01TA',

                # --- PERGUNTAS DE SIM OU NÃO (1: Sim, 2: Não) ---  
                # Quarto para si mesmo
                'ST250Q01JA',
                # Computador para uso escolar
                'ST250Q02JA',
                # Software educacional
                'ST250Q03JA',
                # Celular com internet
                'ST250Q04JA',
                # Internet
                'ST250Q05JA',
                # ST250Q06JA e ST250Q07JA são perguntas que mudam ou aparecem de acordo com o país
                # --- QUANTIDADE DE BENS (1: Nenhum, 2: Um, 3: Dois, 4: Três ou mais) ---
                # Carros, vans ou caminhões
                'ST251Q01JA',
                # Motos ou ciclomotores
                'ST251Q02JA',
                # Banheiros com chuveiro ou banheira
                'ST251Q03JA',
                # Instrumentos musicais
                'ST251Q06JA',
                # Obras de arte
                'ST251Q07JA',

                # --- DISPOSITIVOS COM TELA (Escala de 1 a 8) ---
                # Quantidade total de dispositivos com tela
                'ST253Q01JA',
                
                # --- DETALHAMENTO DE TELAS (1: Nenhum, 2: 1 ou 2, 3: 3 a 5, 4: Mais de 5) ---
                # Televisões
                'ST254Q01JA',
                # Computadores de mesa (Desktop)
                'ST254Q02JA',
                # Laptops ou notebooks
                'ST254Q03JA',
                # Tablets
                'ST254Q04JA',
                # Leitores de e-book (Kindle, etc)
                'ST254Q05JA',
                # Smartphones
                'ST254Q06JA',

                # --- LIVROS E FAMÍLIA ---
                # Quantidade de livros em casa (Escala 1 a 7)
                'ST255Q01JA',
                # Quantidade de irmãos (1: Nenhum a 4: Três ou mais)
                'ST230Q01JA',

                # --- ESCOLARIDADE ---
                # Nível de escolaridade da mãe (Escala 1 a 5)
                'ST005Q01JA',
                # Nível de escolaridade do pai (Escala 1 a 5)
                'ST007Q01JA',
                
                # --- PERCEPÇÃO SOCIOECONÔMICA (Escala de 1 a 10) ---
                # Onde a família está agora na escala social
                'ST259Q01JA',
                # Onde o aluno acha que estará aos 30 anos
                'ST259Q02JA',

                # --- HISTÓRICO E ASSIDUIDADE ---
                # Tempo de permanência na escola atual (Escala 1 a 5)
                'ST226Q01JA',
                # Repetição de ano (01: Nunca, 02: Uma vez, 03: Duas ou mais)
                'ST127Q01TA', # No ensino fundamental (ISCED 1)
                'ST127Q02TA', # No ensino médio inferior (ISCED 2)
                'ST127Q03TA', # No ensino médio superior (ISCED 3)
                # Faltou à escola por mais de 3 meses seguidos?
                'ST260Q01JA', # No ensino fundamental
                'ST260Q02JA', # No ensino médio inferior
                'ST260Q03JA', # No ensino médio superior
                'ST261Q11JA ', # Faltou à escola por mais de 3 meses seguidos por causa de desastres naturais
                # Faltas e atrasos nas últimas 2 semanas (1: Nunca a 4: 5 ou mais vezes)
                'ST062Q01TA', # Faltou um dia inteiro
                'ST062Q02TA', # Matou algumas aulas
                'ST062Q03TA', # Chegou atrasado

                # --- RELAÇÃO COM PROFESSORES E ESCOLA (1: Discordo totalmente a 4: Concordo totalmente) ---
                # Professores me respeitam
                'ST267Q01JA',
                # Professores se preocupariam se eu estivesse triste
                'ST267Q02JA',
                # Professores ficariam felizes em me ver no futuro
                'ST267Q03JA',
                # Sinto medo/intimidado pelos professores
                'ST267Q04JA',
                # Professores realmente se interessam por como estou
                'ST267Q05JA',
                # Professores são amigáveis
                'ST267Q06JA',
                # Professores se interessam pelo bem-estar dos alunos
                'ST267Q07JA',
                # Professores são rudes/malvados comigo
                'ST267Q08JA',

                # --- SENSO DE PERTENCIMENTO (1: Concordo totalmente a 4: Discordo totalmente) ---
                # Sinto-me como um estranho na escola
                'ST034Q01TA',
                # Faço amigos facilmente
                'ST034Q02TA',
                # Sinto que pertenço à escola
                'ST034Q03TA',
                # Sinto-me deslocado
                'ST034Q04TA',
                # Outros alunos parecem gostar de mim
                'ST034Q05TA',
                # Sinto-me sozinho na escola
                'ST034Q06TA',

                # --- SEGURANÇA (1: Concordo totalmente a 4: Discordo totalmente) ---
                'ST265Q01JA', # Seguro no caminho para a escola
                'ST265Q02JA', # Seguro no caminho para casa
                'ST265Q03JA', # Seguro dentro da sala de aula
                'ST265Q04JA', # Seguro em outros locais (corredor, banheiro, etc)

                # --- ROTINA SEMANAL (0 a 5 ou mais dias) ---
                # Intenção de juntar colunas para identificar rotina diaria
                # Atividades ANTES da escola:
                'ST294Q01JA', # Toma café da manhã
                'ST294Q02JA', # Estuda ou faz lição
                'ST294Q03JA', # Trabalha em casa / cuida da família
                'ST294Q04JA', # Trabalha por dinheiro
                'ST294Q05JA', # Pratica exercícios/esportes
                # Atividades DEPOIS da escola:
                'ST295Q01JA', # Janta
                'ST295Q02JA', # Estuda ou faz lição
                'ST295Q03JA', # Trabalha em casa / cuida da família
                'ST295Q04JA', # Trabalha por dinheiro
                'ST295Q05JA', # Pratica exercícios/esportes

                # --- USO DE DISPOSITIVOS DIGITAIS (Horas por dia) ---
                # Podem ser juntadas em horas semanais ou descartadas posteriormente
                # None / Up to 1 hour / More than 1 hour and up to 2 hours /
                # More than 2 hours and up to 3 hours / More than 3 hours and up to 4 hours / More than 4 hours and up
                # to 5 hours / More than 5 hours and up to 6 hours / More than 6 hours and up to 7 hours / More than 7
                # Para atividades de aprendizagem:
                'ST326Q01JA', # Na escola
                'ST326Q02JA', # Fora da escola (dias úteis)
                'ST326Q03JA', # Nos fins de semana
                # Para lazer:
                'ST326Q04JA', # Na escola
                'ST326Q05JA', # Fora da escola (dias úteis)
                'ST326Q06JA', # Nos fins de semana

                # --- MENTALIDADE E CRENÇAS (1: Discordo totalmente a 4: Concordo totalmente) ---
                # Inteligência é algo que você não pode mudar muito
                'ST263Q02JA',
                # Algumas pessoas não são boas em matemática, não importa o quanto estudem
                'ST263Q04JA',
                # Algumas pessoas não são boas em [Língua Materna], não importa o quanto estudem
                'ST263Q06JA',
                # Criatividade é algo que você não pode mudar muito
                'ST263Q08JA',

                # --- BEM-ESTAR GERAL ---
                # Satisfação com a vida (Escala de 0 a 10)
                'ST016Q01NA',

                # --- CARGA HORÁRIA (Número de aulas por semana) ---
                # Número de aulas de Matemática por semana
                'ST059Q01TA',
                # Número total de aulas por semana (todas as matérias)
                'ST059Q02JA',

                # --- TEMPO DE LIÇÃO DE CASA (Escala de 1 a 6) ---
                # Tempo gasto com lição de Matemática
                'ST296Q01JA',
                # Tempo gasto com lição de [Língua Materna]
                'ST296Q02JA',
                # Tempo gasto com lição de [Ciências]
                'ST296Q03JA',
                # Tempo total gasto com lição de todas as matérias
                'ST296Q04JA',
                
                # --- QUALIDADE DA AULA DE MATEMÁTICA (Escala de 1 a 10) ---
                'ST272Q01JA',
                
                # Confiança em aprendizado por conta propria caso a escola feche, escala de 1 a 4, com 4 sendo muito confiante
                'ST356Q01JA',

                # ST022Q01TA tras questões sobre a lingua mais falada em casa, pode ser usada para ver se a lingua mais falada é a nativa , porem precisa de tratamento para tirar a informação
]
materias = ['MATH', 'READ', 'SCIE']
cols_notas = [f'PV{i}{mat}' for mat in materias for i in range(1, 11)]

cols_student_df = cols_student + cols_notas

df_student = pd.read_parquet('data/parquet/CY08MSP_STU_QQQ.parquet', columns=cols_student_df)
df_student = df_student.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Sucesso: {df_student.shape[0]} linhas carregadas.")

# 1. Definimos os grupos de colunas para cada matéria
cols_math = [f'PV{i}MATH' for i in range(1, 11)]
cols_read = [f'PV{i}READ' for i in range(1, 11)]
cols_scie = [f'PV{i}SCIE' for i in range(1, 11)]

# 2. Criamos as médias individuais para cada grande área
df_student['MEAN_MATH'] = df_student[cols_math].mean(axis=1)
df_student['MEAN_READ'] = df_student[cols_read].mean(axis=1)
df_student['MEAN_SCIE'] = df_student[cols_scie].mean(axis=1)

# 3. Calculamos a Média Global das 3 matérias
df_student['MEDIA_GLOBAL_PISA'] = df_student[['MEAN_MATH', 'MEAN_READ', 'MEAN_SCIE']].mean(axis=1)
df_student = df_student.drop(columns= cols_notas)

# 2. Lendo os dados da ESCOLA (School Questionnaire)
cols_school = ['CNT', 'CNTSCHID', 'SC001Q01TA', 'STRATUM']
df_school = pd.read_parquet('data/parquet/CY08MSP_SCH_QQQ.parquet', columns=cols_school)
df_school = df_school.sample(frac=1, random_state=42).reset_index(drop=True)
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