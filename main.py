import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium", app_title="Statistical normality testing")


@app.cell
def _(
    ad_pvalue,
    ad_stat,
    dp_pvalue,
    dp_stat,
    ll_pvalue,
    ll_stat,
    mo,
    sw_pvalue,
    sw_stat,
):
    def res(p, alpha):
        if isinstance(p, str):
            return "<span style='color:yellow;'>не применим</span>"

        if isinstance(p, float):
            if p > alpha:
                return "<span style='color:lightgreen;'>не отвергается</span>"
            else: 
                return "<span style='color:red;'>отвергается</span>"

    def fmt(val):
        if isinstance(val, str):
            return f"<span style='color:yellow;'>{val}</span>"
        elif isinstance(val, float):
            return f'{val:.3f}'

    summary = mo.md(f"""
    |  Критерий        | Значение статистики |   $p$-значение  |  $H_0, \\alpha=0.05$  |  $H_0, \\alpha=0.01$ |
    |: -------------- :|: ----------------- :|: ------------- :|: ------------------- :| :----: |
    | Lillifors        |    {ll_stat:.3f}    | {ll_pvalue:.3f} | {res(ll_pvalue,0.05)} | {res(ll_pvalue,0.01)}|
    |Anderson-Darling  |    {ad_stat:.3f}    | {ad_pvalue:.3f} | {res(ad_pvalue,0.05)} | {res(ad_pvalue,0.01)}|
    |Shapiro-Wilk      |    {fmt(sw_stat)}   | {fmt(sw_pvalue)}| {res(sw_pvalue,0.05)} | {res(sw_pvalue,0.01)}|
    |D'Agostino-Pearson|    {fmt(dp_stat)}   | {fmt(dp_pvalue)}| {res(dp_pvalue,0.05)} | {res(dp_pvalue,0.01)}|
    """)
    return fmt, res, summary


@app.cell
def _(
    ad_pvalue,
    ad_stat,
    dp_pvalue,
    dp_stat,
    fmt,
    ll_pvalue,
    ll_stat,
    mo,
    res,
    sw_pvalue,
    sw_stat,
):
    ## Сводная таблица     
    # table-layout: fixed;
    # width: 80%

    summary_html = mo.center(mo.md(f"""
    <style>
    .summary_table table {{
        border-collapse: collapse;
        border-spacing: 1px;
        text-align: center;
    }}

    .summary_table thead, tbody, th, td {{
        border: 1px solid #4e4e4f;;
        padding: 5px;
        text-align: center;
        font-family: Times New Roman;
        font-size: 12px;
    }}
    </style>

    <div class="summary_table" role="region" tabindex="0">
    <table class="summary_table">
        <thead>
            <tr>
                <th style='text-align:center;border: 1px solid #4e4e4f;' colspan=2>Название критерия</th>
                <th style='text-align:center;border: 1px solid #4e4e4f;' rowspan=2>Значение<br/>статистики</th>
                <th style='text-align:center;border: 1px solid #4e4e4f;' rowspan=2><em>p</em>-значение</th>
                <th style='text-align:center;border: 1px solid #4e4e4f;' colspan=2><em>H</em><sub>0</sub></th>
                <th style='text-align:center;border: 1px solid #4e4e4f;' rowspan=2>Примечание</th>
            </tr>
            <tr>
                <th style='text-align:center;border: 1px solid #4e4e4f;'>ru</th><th style='text-align:center;border: 1px solid #4e4e4f;'>en</th>
                <th style='text-align:center;border: 1px solid #4e4e4f;'><i>α</i> = 0.05</th>
                <th style='text-align:center;border: 1px solid #4e4e4f;'><i>α</i> = 0.01</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style='text-align:center;'>Лиллифорс</td>
                <td style='text-align:center;'>Lillifors</td>
                <td style='text-align:center;'>{fmt(ll_stat)}</td>
                <td style='text-align:center;'>{fmt(ll_pvalue)}</td>
                <td style='text-align:center;'>{res(ll_pvalue,0.05)}</td>
                <td style='text-align:center;'>{res(ll_pvalue,0.01)}</td>
                <td style='text-align:center;'>Тест Колмогорова-Смирнова адаптированный для нормального распределения</td>
            </tr>
            <tr>
                <td style='text-align:center;'>Андерсона-Дарлинга</td>
                <td style='text-align:center;'>Anderson-Darling</td>
                <td style='text-align:center;'>{fmt(ad_stat)}</td>
                <td style='text-align:center;'>{fmt(ad_pvalue)}</td>
                <td style='text-align:center;'>{res(ad_pvalue,0.05)}</td>
                <td style='text-align:center;'>{res(ad_pvalue,0.01)}</td>
                <td style='text-align:center;'></td>
            </tr>
            <tr>
                <td style='text-align:center;'>Шапиро-Уилки</td>
                <td style='text-align:center;'>Shapiro-Wilk</td>
                <td style='text-align:center;'>{fmt(sw_stat)}</td>
                <td style='text-align:center;'>{fmt(sw_pvalue)}</td>
                <td style='text-align:center;'>{res(sw_pvalue,0.05)}</td>
                <td style='text-align:center;'>{res(sw_pvalue,0.01)}</td>
                <td style='text-align:center;'></td>
            </tr>
            <tr>
                <td style='text-align:center;'>Д'Агостино-Пирсона</td>
                <td style='text-align:center;'>D'Agostino-Pearson</td>
                <td style='text-align:center;'>{fmt(dp_stat)}</td>
                <td style='text-align:center;'>{fmt(dp_pvalue)}</td>
                <td style='text-align:center;'>{res(dp_pvalue,0.05)}</td>
                <td style='text-align:center;'>{res(dp_pvalue,0.01)}</td>
                <td style='text-align:center;'></td>
            </tr>
        </tbody>
    </table>
    </div>
    """))

    #summary_html
    return (summary_html,)


@app.cell
def _():
    import marimo as mo
    import math
    import altair as alt
    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm
    import numpy as np
    return alt, math, mo, np, pd, sm, stats


@app.cell
def _(alt, df, mo, pd):
    chart1 = (alt.Chart(df, width=200, height=200, title='Histogram')
        .mark_bar()
        .encode(alt.X('x:Q', bin=True), y='count()')
    )

    chart2 = (alt.Chart(df, width=200, height=200, title='Q-Q plot')
        .transform_quantile('x', step=0.1, as_=['p', 'v'])
        .transform_calculate(normal='quantileNormal(datum.p)')
        .mark_circle()      
        .encode(alt.Y('p:Q'), alt.X('normal:Q'))
    )

    chart3 = (alt.Chart(pd.DataFrame({'x': [-2,2], 'y': [0,1]}), width=200, height=200)
        .mark_line()
        .encode(alt.X('x:Q'), alt.Y('y:Q'))
    )

    chart4 = (alt.Chart(df, width=200, height=200)
        .mark_tick()      
        .encode(
            alt.X('x:Q', title=None),
            alt.Size('count()', legend=None, color='red')
        )
    )

    visual = mo.md(f'''
    {mo.center(mo.as_html((chart1 + chart4) | (chart2 + chart3)))}
    ''')
    #visual
    return chart1, chart2, chart3, chart4, visual


@app.cell
def _(data, math, pd):
    df = pd.DataFrame({'x': data})
    n = len(data)

    coln = [i for i in range(1,len(data)+1)]
    cold = data
    xbar = sum(cold)/len(cold)

    col1 = [d - xbar for d in cold]
    col2 = [d**2 for d in col1]
    col3 = [d**3 for d in col1]
    col4 = [d**4 for d in col1]

    s = math.sqrt(sum(col2)/(len(data)-1))
    A = sum(col3)/len(cold)/s**3
    E = sum(col4)/len(cold)/s**4-3

    # Показатели по Пустыльнику
    A_кр = 3*math.sqrt(6*(n-1)/(n+1)/(n-3))
    E_кр = 5*math.sqrt(24*n*(n-2)*(n-3)/(n+1)**2/(n+3)/(n+5))

    # Показатели по Плохинскому
    m_A = math.sqrt(6/n); t_A = abs(A)/m_A
    m_E = 2*m_A;          t_E = abs(E)/m_E
    return (
        A,
        A_кр,
        E,
        E_кр,
        col1,
        col2,
        col3,
        col4,
        cold,
        coln,
        df,
        m_A,
        m_E,
        n,
        s,
        t_A,
        t_E,
        xbar,
    )


@app.cell
def _(col1, col2, col3, col4, coln, pd):
    def get_md_table(df,
        fmts=None,             
        index=False,
        headers=None,
        add_sum=False,
        float_format="%.3f"
    ):
        # Get headers
        if headers is None:
            headers = []
            if index:
                headers.append(df.index.name if df.index.name else "Index")
            headers += list(df.columns)

        # Create markdown rows
        rows = []
        rows.append("| "+" | ".join(str(h) for h in headers)+" |") # Header row
        rows.append("| "+" :|: ".join([" --- "]*len(headers))+" :|")  # Separator row
        for i, row in df.iterrows():     # Data rows
            row_data = []
            if index:
                row_data.append(str(i))
            row_data += [(f'{x:.3f}' if isinstance(x, float) else str(x)) for x in row.values ]
            rows.append("| " + " | ".join(row_data) + " |")

        if add_sum:
            # sum_cols = [f'{df[col].sum():.3f}' for col in df.columns[1:]]
            sum_cols = [f'{sum(float(item) for item in df[col]):.3f}' for col in df.columns[1:]]
            rows.append("| $\sum$ | " + " | ".join(sum_cols) + " |")

        return "\n".join(rows)

    in_df = pd.DataFrame({
        '№': coln,
        'x-x': col1,
        '(x-x)**2': col2,
        '(x-x)**3': col3,
        '(x-x)**4': col4,
    })

    md_table = get_md_table(in_df,
        headers=['$№$',r'$x-x_i$',r'$(x-x_i)^2$',r'$(x-x_i)^3$',r'$(x-x_i)^4$'],
        add_sum=True,          
    )
    return get_md_table, in_df, md_table


@app.cell
def _(data, np, sm, stats):
    # Lillifors test http://en.wikipedia.org/wiki/Lilliefors_test
    ll_stat, ll_pvalue = sm.stats.lilliefors(data)

    # Shapiro-Wilk
    if len(data) >= 8:
        sw_stat, sw_pvalue = stats.shapiro(data)
    else:
        sw_stat, sw_pvalue = 'n<8', 'n<8'

    #Anderson-Darling
    ad_stat, ad_pvalue = sm.stats.normal_ad(np.array(data))

    # D’Agostino-Pearson
    if len(data) >= 20:
        dp_stat, dp_pvalue = stats.normaltest(data)
    else:
        dp_stat, dp_pvalue = 'n<20', 'n<20'
    return (
        ad_pvalue,
        ad_stat,
        dp_pvalue,
        dp_stat,
        ll_pvalue,
        ll_stat,
        sw_pvalue,
        sw_stat,
    )


@app.cell
def _(mo):
    intro_why = mo.md('''
    Широкий круг статистических методов требуют для корректной работы проверки нормальности распределения:

    * Дисперсионный анализ относится к группе параметрических методов и поэтому его следует применять только тогда, когда известно или можно принять, что распределение признака является нормальным.

    **Список источников**

    * Суходольский Г.В., 1972;
    * Шеффе Г., 1980 и др.)
    * Плохинский Н.А.,1964, с.34-36;
    * Плохинский Н.А., 1970, с.71-72
    * Пустыльник     
    ''')

    intro_when = mo.md('''
    Центральная предельная теорема (ЦПТ) утверждает, что распределение суммы конечного числа случайных величин с конечными матожиданиями и
    дисперсиями стремится к нормальному.

    В частности, поскольку среднее серии измерений можно рассматривать как сумму одинаково распределенных случайных величин, то ожидается, что выборочное среднее для достаточно больших выборок - нормально распределено.

        **Вопрос:** что считать достаточно большой выборкой?

    ''')

    intro_which = mo.md('''
    Статистические методы проверки называются еще **критериями**.

    ошибка первого рода: α = вероятность не заметить эффект или различие, когда они есть

    ошибка второго рода: β = вероятность ошибочно определить эффект или различие, когда их нет

    Мощность критерия = 1-β - вероятность заметить эффект или различие, когда они есть 

    При одинаковом заданном уровне значимости α, различные критерии могут иметь различные мощности.
    Понятно, что надежнее всего определит эффект или различие критерий с наибольшей мощностью. 

    Тем не менее, бывают ситуации, когда может потребоваться использовать менее мощные критерии, например, если в работу или отчет нужно включить расчеты для внешней проверки, а нормальность надежно определяется.

    Критерии Плохинского и Пустыльника легко проверяются, в том числе и ручным расчетом.
    ''')
    return intro_when, intro_which, intro_why


@app.cell
def _(A, A_кр, E, E_кр, cold, m_A, m_E, md_table, mo, n, s, t_A, t_E):
    _ins1 = rf'$\displaystyle\frac{{|{A:.3f}|}}{{|{m_A:.3f}|}}$'

    Пустыльник = mo.md(rf'''

    Имеем: _n_ = {len(cold)} - количество испытаний, 

    {md_table}

           _s_ = {s:.3f} - выборочное стандартное отклонение 

    Тогда критические показатели для значений А и Е

    &nbsp; $\displaystyle A_{{кр}}=3\cdot\sqrt{{\frac{{6\cdot(n-1)}}{{(n+1)\cdot(n+3)}}}}$
    = {A_кр:.3f}

    &nbsp; $\displaystyle E_{{кр}}=5\cdot\sqrt{{\frac{{24\cdot(n-1)}}{{(n+1)\cdot(n+3)}}}}$
    = {E_кр:.3f}


    Показатели асимметрии и эксцесса свидетельствуют о достоверном отличии эмпирических распределений от нормального в том случае, если они превышают по абсолютной величине свою ошибку репрезентативности 
    в 3 и более раз:

    В данном случае:

    $\displaystyle t_A=\frac{{|A|}}{{m_A}}$={_ins1}
    = {t_A:.3f} - {'<span style="color:lightgreen;">не превышает 3</span>' if  t_A <= 3 else '<span style="color:red;">превышает 3</span>'}
    ,&nbsp;&nbsp;&nbsp;&nbsp;
    $\displaystyle t_E=\frac{{|E|}}{{m_E}}$
    = {t_E:.3f} - {'<span style="color:lightgreen;">не превышает 3</span>' if  t_E <= 3 else '<span style="color:red;">превышает 3</span>'}

    Гипотеза о нормальности 
    {'<span style="color:lightgreen;">не отвергается</span>' if  t_A <= 3 and t_E <= 3 else '<span style="color:red;">отвергается</span>'}
    ''')

    _ins2 = rf'$\displaystyle\frac{{|{A:.3f}|}}{{|{m_A:.3f}|}}$'

    Плохинский = mo.md(rf'''

    Имеем: _n_ = {n} - количество испытаний, 
           _s_ = {s:.3f} - стандартное отклонение 

    Тогда

    Ассимметрия:&nbsp; $\displaystyle A_{{эмп}}=\frac1n\sum_{{i}}^n\left(\frac{{x_i-\bar{{x}}}}\sigma\right)^3$
    = {A:.3f}

    Ошибка репрезентативности ассимметрии:&nbsp; $\displaystyle m_{{A}}=\sqrt{{\frac6n}}$
    = {m_A:.3f}

    Эксцесс:&nbsp; $\displaystyle E_{{эмп}}=\frac1n\sum_{{i}}^n\left(\frac{{x_i-\bar{{x}}}}\sigma\right)^4-3$
    = {E:.3f}

    Ошибка репрезентативности эксцесса:&nbsp; $\displaystyle m_{{E}}=2\sqrt{{\frac6n}}$
    = {m_E:.3f}

    Показатели асимметрии и эксцесса свидетельствуют о достоверном отличии эмпирических распределений от нормального в том случае, если они превышают по абсолютной величине свою ошибку репрезентативности 
    в 3 и более раз:

    В данном случае:

    $\displaystyle t_A=\frac{{|A|}}{{m_A}}$={_ins2}
    = {t_A:.3f} - {'<span style="color:lightgreen;">не превышает</span>' if  t_A <= 3 else '<span style="color:red;">превышает</span>'}
    ,&nbsp;&nbsp;&nbsp;&nbsp;
    $\displaystyle t_E=\frac{{|E|}}{{m_E}}$
    = {t_E:.3f} - {'<span style="color:lightgreen;">не превышает</span>' if  t_E <= 3 else '<span style="color:red;">превышает</span>'}

    Гипотеза о нормальности 
    {'<span style="color:lightgreen;">не отвергается</span>' if  t_A <= 3 and t_E <= 3 else '<span style="color:red;">отвергается</span>'}
    ''')
    return Плохинский, Пустыльник


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Проверка нормальности распределения""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Введение""")
    return


@app.cell(hide_code=True)
def _(intro_when, intro_which, intro_why, mo):
    def orange(text):
        return f'<span style="color:orange;">{text}</span>'

    mo.accordion({
      orange('Зачем проверять нормальность распределения?')              : intro_why,
      orange('Когда может не сработать центральная предельная теорема?') : intro_when,
      orange('Какой метод проверки выбрать?')                            : intro_which,
    }, multiple=False)
    return (orange,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Ввести данные""")
    return


@app.cell
def _(mo):
    _choose = mo.ui.dropdown([
        'нормального распределения'     ,
        'логнормального распределения'  ,
        'суммы нормальных распределений',
        'равномерного распределения'    ,
        'тестовые данные'               ,
    ], 
      label=' из',
    )

    _volume = mo.ui.number(
        5, 5000,
        value=20,
        label='объемом '    
    )

    _new_data = mo.ui.text_area(
        value='',
        placeholder='Введите в этом поле, разделяя запятыми без пробелов,' + 
          ' собственные данные, используя десятичную точку, например, так:\n1.23,3.27,4.3367',
        rows = 5,
    )

    what_data = mo.md("""
      {new_data}

      или выбрать тестовую выборку 
  
      {volume} {choose}
    """).batch(new_data=_new_data, volume=_volume, choose=_choose).form(submit_button_label="Выбрать")

    what_data
    return (what_data,)


@app.cell
def _(mo, np, what_data):
    mo.stop(what_data.value is None, None)
    #print(what_data.value)
    own_data = what_data.value['new_data']
    vol = what_data.value['volume']
    chosen = what_data.value['choose']
    if own_data == '':
        match chosen:
            case 'нормального распределения':
                data = np.random.normal(3, 2, vol)
            case 'логнормального распределения':
                data = np.random.lognormal(0, 1, vol)
            case 'суммы нормальных распределений':
                data = np.random.normal(3, 2, vol) + np.random.normal(-1, 3, vol)
            case 'равномерного распределения': 
                data = np.random.uniform(2,4, vol)
            case 'тестовые данные':
                data = [11, 13, 12, 9, 10, 11, 8, 10, 15, 14, 8, 7, 10, 10, 5, 8]

    return chosen, data, own_data, vol


@app.cell(hide_code=True)
def _():
    ## Методы проверки нормальности распределения
    return


@app.cell(hide_code=True)
def _(mo, orange, summary_html, visual, Плохинский, Пустыльник):
    mo.accordion({
      orange('Визуальный анализ: ГОСТ Р ИСО 5479-2002')             : visual,
      orange('Критерий Шапиро-Уилка (8≤n≤50): ГОСТ Р ИСО 5479-2002'): 'Work in progress',
      orange('Проверка нормальности по Пустыльнику')                : Пустыльник,
      orange('Проверка нормальности по Плохинскому')                : Плохинский,
      orange('Сводная таблица')                                     : summary_html,
      #orange('Сводная md-таблица'                                  : summary,
    }, multiple=False, lazy=True)
    return


if __name__ == "__main__":
    app.run()
