import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Thread
from PIL import Image
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



# def gamma_params(mean, std):
#     """uncomment if need gamma param"""
#     shape = round((mean / std) ** 2, 4)
#     scale = round((std ** 2) / mean, 4)
#     return (shape, scale)

def my_norm_confidence(df, column, alpha=0.9):
    """Создание доверительных интервалов"""
    interval = stats.norm.interval(alpha, loc=df[column].mean(), scale=df[column].std())
    return interval


def full_discrable(df, column, alpha=0.9):
    """Общий анализ дданных по выбранной колонке датафрейма"""
    if pd.api.types.is_numeric_dtype(df[column]):
        with st.beta_expander("График распределения с доверительными интервалами и 99% процентилем: "):
            """Возможно есть смысл перейти на plotly"""
            confidence = my_norm_confidence(df, column, alpha=0.9)
            fig, ax = plt.subplots()
            fig.set_size_inches(4,3)
            fig = sns.displot(ax=ax, data=df[column])
            plt.axvline(x=confidence[1], color="g", linestyle="-", label='max: {}'.format(confidence[1]))
            plt.axvline(x=confidence[0], color="r", linestyle="-", label='min: {}'.format(confidence[0]))
            plt.axvline(x=np.percentile(df[column], 99), color="b", linestyle=":",
                        label='99%: {}'.format(np.percentile(df[column], 99)))
            plt.title('Интервал: от {} до {}'.format(confidence[0], confidence[1]))
            plt.legend()
            st.pyplot(fig)
            # Явный признак?
            if df[column].min() > confidence[0]:
                st.write("Распределение смещено!")

        st.write('50% процентиль: ', np.percentile(df[column], 50))
        st.write('75% процентиль: ', np.percentile(df[column], 75))
        st.write('90% процентиль: ', np.percentile(df[column], 90))
        st.write('99% процентиль: ', np.percentile(df[column], 99))
        st.write("Эксцесс: ", kurtosis(df[column])) # добавить определения для смещенной выборки: см. Notion
        st.write("Ассиметрия: ", skew(df[column])) # добавить определения для смещенной выборки: см. Notion
        # shape, scale = gamma_params(df[column].mean(), df[column].std())
        # st.write("Гамма параметры: ")
        # st.write("k: ", shape)
        # st.write("theta: ", scale)
    else:
        st.text('Тип выбранной колонки - текст, дальнейшие расчеты невозможны.')

def correlation_in_data(df):
    """Поиск корреляций в датафрейме"""
    df_types = pd.DataFrame(df.dtypes, columns=['Тип данных'])
    numerical_cols = df_types[~df_types['Тип данных'].isin(['object',
                                                            'bool'])].index.values
    st.title("Корреляция в данных")
    columns_ = st.multiselect('Выберите колонки для построения матрицы корреляций', numerical_cols)
    if len(columns_) != 0:
        dff = df[columns_]
        corr_ = dff.corr()
        st.write(corr_)
    else:
        st.stop()
    with st.beta_expander("Визуализация матрицы корреляций: "):
        """Возможно есть смысл перейти на plotly"""
        fig, ax = plt.subplots()
        # fig.set_size_inches(4, 3)
        sns.heatmap(corr_, ax=ax, center=0, annot=True, cmap='Spectral')
        st.pyplot(fig)

def explore(df):
    """Общий статистический анализ данных"""
    # DATA
    st.write('Данные:')
    st.write(df)
    # SUMMARY
    try:
        df_types = pd.DataFrame(df.dtypes, columns=['Тип данных'])
        numerical_cols = df_types[~df_types['Тип данных'].isin(['object',
                       'bool'])].index.values
        df_types['Количество'] = df.count()
        df_types['Уникальных значений'] = df.nunique()
        df_types['Минимум'] = df[numerical_cols].min()
        df_types['Максимум'] = df[numerical_cols].max()
        df_types['Среднее значение'] = df[numerical_cols].mean()
        df_types['Медианное значение'] = df[numerical_cols].median()
        df_types['Среднеквадратическое отклонение'] = df[numerical_cols].std()
        st.write('Общие сведения о данных :')
        st.write(df_types)
    except AttributeError as err:
        # st.warning(err)
        st.stop()


def metod_range_territory(df, column):
    """Добавление методов под конкретные задачи:
        1. Метод ранжирования территории"""
    if pd.api.types.is_numeric_dtype(df[column]):
        mean = df[column].mean()
        std = df[column].std()
        p = st.number_input('Введите показатель заболеваемости P (по умолчанию 88,3): ',value=88.3,  step=0.0001)
        st.write('Среднее значение: ', mean)
        st.write('Среднеквадратическое отклонение: ', std)
        k = 1.0
        st.write('Коэффициент K: ', k)
        with st.beta_expander("Ввести значения вручную: "):
            mean = st.number_input('Среднее значение: ', value=mean, step=0.0001)
            std = st.number_input('Среднеквадратическое отклонение: ', value=std, step=0.0001)
            k = st.number_input('Коэффициент K: ', value=k, step=0.0001)
        if st.button('Сбросить значения'):
            mean = df[column].mean()
            std = df[column].std()
            k = 1.0
        st.title('Результат: ')
        if p>=(mean+1.5*k*std):
            st.write('Высокий уровень заболеваемости, P({}) >= {}'.format(p, (mean+1.5*k*std)))
        if (mean+0.5*k*std) < p < (mean+1.5*k*std):
            st.write('Выше среднего уровня заболеваемости, {} < P({}) < {}'.format((mean+0.5*k*std), p, (mean + 1.5 * k * std)))
        if (mean-0.5*k*std) <= p <= (mean+0.5*k*std):
            st.write('Cредний уровень заболеваемости, {} =< P({}) =< {}'.format((mean-0.5*k*std), p, (mean+0.5*k*std)))
        if (mean - 1.5 * k * std) < p < (mean - 0.5 * k * std):
            st.write('Ниже среднего уровня заболеваемости, {} < P({}) < {}'.format((mean - 1.5 * k * std), p,
                                                                                (mean - 0.5 * k * std)))
        if p <= (mean - 1.5 * k * std):
            st.write('Низкий уровень заболеваемости, P({}) <= {}'.format(p, (mean - 1.5 * k * std)))
    else:
        st.text('Тип выбранной колонки - текст, дальнейшие расчеты невозможны.')
        pass


def get_df(file):
    """Выбор метода чтения файла с ограничениями по расширению файла"""
    try:
        extension = file.name.split('.')[1]
        if extension.upper() == 'CSV':
            df = pd.read_csv(file, encoding='cp1251')
        elif extension.upper() == 'XLSX':
            df = pd.read_excel(file,  engine='openpyxl')
        return df
    except OSError as err:
        st.warning(err)
        return None

def load_data():
    """Загрузка данных"""
    st.write("Выберите файл с расширением .csv или .xlsx для дальнейшего анализа:")
    file = st.file_uploader("", type=['csv', 'xlsx'])
    if not file:
        st.stop()
        # st.write("Выбранный файл не имеет расширения .csv или .xlsx")
        # return
    df = get_df(file)
    return df







# Настройка общего вида приложения
# image = Image.open(r"logo_.png")
# favicon = Image.open(r"logo.png")
st.set_page_config(page_title = 'data-analysis-app', layout="wide", initial_sidebar_state="expanded") #, page_icon = (favicon)
#st.sidebar.image(image, width=5, use_column_width='auto')
st.sidebar.title('Модуль анализа данных')
selected = st.sidebar.selectbox('Выберите инструмент', ['', "Первичный анализ данных","Углубленный анализ данных", "Методика ранжирования территорий по уровням заболеваемости"])




if selected == "Методика ранжирования территорий по уровням заболеваемости":
    st.title('Методика ранжирования территорий по уровням заболеваемости')
    df = load_data()
    st.write(df)
    # explore(df)
    st.title('Выберите колонку для дальнейших расчетов: ')
    column = st.selectbox('', df.columns)
    metod_range_territory(df, column)

if selected == "Углубленный анализ данных":
    st.title('Углубленный анализ данных, сводный отчет по данным: ')
    from streamlit_pandas_profiling import st_profile_report
    df = load_data()
    pr = ProfileReport(df)
    st_profile_report(pr) #streamlit-pandas-profiling 0.1.2

if selected == "Первичный анализ данных":
    st.title('Первичный анализ данных: ')
    df = load_data()
    explore(df)
    column = st.selectbox('Выберите колонку для дальнейшего анализа: ', df.columns)
    full_discrable(df, column, alpha=0.9)
    correlation_in_data(df)

