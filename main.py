

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium", app_title="Statistical normality testing")


@app.cell
def _(ad_pv, ad_st, dp_pv, dp_st, ll_pv, ll_st, mo, sw_pv, sw_st):
    def res(p, alpha):
        """ Расцвечиваем выводы в зависимости от результата """
        if isinstance(p, str):
            return "<span style='color:yellow;'>не применим</span>"

        if isinstance(p, float):
            if p > alpha:
                return "<span style='color:lightgreen;'>не отвергается</span>"
            else: 
                return "<span style='color:red;'>отвергается</span>"

    def fmt(val):
        """ Расцвечиваем числовой вывод в зависимости от результата """
        if isinstance(val, str):
            return f"<span style='color:yellow;'>{val}</span>"
        elif isinstance(val, float):
            return f'{val:.3f}'

    summary_2_alphas = mo.md(f"""
    |  Критерий        |Значение статистики|$p$-значение|$H_0,\\alpha=0.05$ | $H_0, \\alpha=0.01$ |
    |: -------------- :|: --------------- :|: -------- :|: --------------- :| :-----------------: |
    | Lillifors        |    {ll_st:.3f}    |{ll_pv:.3f} | {res(ll_pv,0.05)} | {res(ll_pv,0.01)}   |
    |Anderson-Darling  |    {ad_st:.3f}    |{ad_pv:.3f} | {res(ad_pv,0.05)} | {res(ad_pv,0.01)}   |
    |Shapiro-Wilk      |    {fmt(sw_st)}   |{fmt(sw_pv)}| {res(sw_pv,0.05)} | {res(sw_pv,0.01)}   |
    |D'Agostino-Pearson|    {fmt(dp_st)}   |{fmt(dp_pv)}| {res(dp_pv,0.05)} | {res(dp_pv,0.01)}   |
    """)

    summary = mo.md(f"""
    |  Критерий        | Значение статистики | $p$-значение | $H_0,\\alpha=0.05$ | 
    |: -------------- :|: ----------------- :|: ---------- :|: ---------------- :| 
    | Lillifors        |    {ll_st:.3f}      | {ll_pv:.3f}  |  {res(ll_pv,0.05)} | 
    |Anderson-Darling  |    {ad_st:.3f}      | {ad_pv:.3f}  |  {res(ad_pv,0.05)} | 
    |Shapiro-Wilk      |    {fmt(sw_st)}     | {fmt(sw_pv)} |  {res(sw_pv,0.05)} | 
    |D'Agostino-Pearson|    {fmt(dp_st)}     | {fmt(dp_pv)} |  {res(dp_pv,0.05)} | 
    """)
    return fmt, res


@app.cell
def _(ad_pv, as_st, dp_pv, dp_st, fmt, ll_pv, ll_st, mo, res, sw_pv, sw_st):
    ## Сводная таблица     
    # table-layout: fixed;
    # width: 80%

    th_cell = "style='text-align: center; border: 1px solid #4e4e4f; font-size:14px';"
    tb_cell = "style='text-align: center;'"


    summary_html_full = mo.md(f"""
    <style>
    .summary_table table {{
        border-collapse: collapse;
        border-spacing: 1px;
        text-align: center;
    }}

    .summary_table thead, tbody, th, td {{
        border: 1px solid #4e4e4f;
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
                <th {th_cell} colspan=2>Название критерия</th>
                <th {th_cell} rowspan=2>Значение<br/>статистики</th>
                <th {th_cell} rowspan=2><em>p</em>-значение</th>
                <th {th_cell} colspan=2><em>H</em><sub>0</sub></th>
                <th {th_cell} rowspan=2>Примечание</th>
            </tr>
            <tr>
                <th {th_cell}>ru</th><th {th_cell}>en</th>
                <th {th_cell}><i>α</i> = 0.05</th>
                <th {th_cell}><i>α</i> = 0.01</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td {tb_cell}>Лиллифорс</td>
                <td {tb_cell}>Lillifors</td>
                <td {tb_cell}>{fmt(ll_st)}</td>
                <td {tb_cell}>{fmt(ll_pv)}</td>
                <td {tb_cell}>{res(ll_pv,0.05)}</td>
                <td {tb_cell}>{res(ll_pv,0.01)}</td>
                <td {tb_cell}>Тест Колмогорова-Смирнова адаптированный для нормального распределения</td>
            </tr>
            <tr>
                <td {tb_cell}>Андерсона-Дарлинга</td>
                <td {tb_cell}>Anderson-Darling</td>
                <td {tb_cell}>{fmt(as_st)}</td>
                <td {tb_cell}>{fmt(ad_pv)}</td>
                <td {tb_cell}>{res(ad_pv,0.05)}</td>
                <td {tb_cell}>{res(ad_pv,0.01)}</td>
                <td {tb_cell}></td>
            </tr>
            <tr>
                <td {tb_cell}>Шапиро-Уилки</td>
                <td {tb_cell}>Shapiro-Wilk</td>
                <td {tb_cell}>{fmt(sw_st)}</td>
                <td {tb_cell}>{fmt(sw_pv)}</td>
                <td {tb_cell}>{res(sw_pv,0.05)}</td>
                <td {tb_cell}>{res(sw_pv,0.01)}</td>
                <td {tb_cell}></td>
            </tr>
            <tr>
                <td {tb_cell}>Д'Агостино-Пирсона</td>
                <td {tb_cell}>D'Agostino-Pearson</td>
                <td {tb_cell}>{fmt(dp_st)}</td>
                <td {tb_cell}>{fmt(dp_pv)}</td>
                <td {tb_cell}>{res(dp_pv,0.05)}</td>
                <td {tb_cell}>{res(dp_pv,0.01)}</td>
                <td {tb_cell}></td>
            </tr>
        </tbody>
    </table>
    </div>
    """).center()


    summary_html = mo.md(f"""
    <style>
    .summary_table table {{
        border-collapse: collapse;
        border-spacing: 1px;
        text-align: center;
    }}

    .summary_table thead, tbody, th, td {{
        border: 1px solid #4e4e4f;
        padding: 5px;
        text-align: center;
        font-family: Times New Roman;
        font-size: 14px;
    }}
    </style>

    <div class="summary_table" role="region" tabindex="0">
    <table class="summary_table">
        <thead>
            <tr>
                <th {th_cell}>Название критерия<br/>ru/en</th>
                <th {th_cell}>Значение<br/>статистики</th>
                <th {th_cell}><em>p</em>-значение</th>
                <th {th_cell}><em>H</em><sub>0</sub><br/><i>α</i> = 0.05</th>
                <th {th_cell}>Примечание</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td {tb_cell}>Лиллифорс<br/>Lillifors</td>
                <td {tb_cell}>{fmt(ll_st)}</td>
                <td {tb_cell}>{fmt(ll_pv)}</td>
                <td {tb_cell}>{res(ll_pv,0.05)}</td>
                <td {tb_cell}>Тест Колмогорова-Смирнова адаптированный для нормального распределения</td>
            </tr>
            <tr>
                <td {tb_cell}>Андерсона-Дарлинга<br/>Anderson-Darling</td>
                <td {tb_cell}>{fmt(as_st)}</td>
                <td {tb_cell}>{fmt(ad_pv)}</td>
                <td {tb_cell}>{res(ad_pv,0.05)}</td>
                <td {tb_cell}></td>
            </tr>
            <tr>
                <td {tb_cell}>Шапиро-Уилки<br/>Shapiro-Wilk</td>
                <td {tb_cell}>{fmt(sw_st)}</td>
                <td {tb_cell}>{fmt(sw_pv)}</td>
                <td {tb_cell}>{res(sw_pv,0.05)}</td>
                <td {tb_cell}></td>
            </tr>
        </tbody>
    </table>
    </div>
    """).center()

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
    return (visual,)


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
            rows.append("| $\\sum$ | " + " | ".join(sum_cols) + " |")

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
    return (md_table,)


@app.cell
def _(data, np, sm, stats):
    # Lillifors test http://en.wikipedia.org/wiki/Lilliefors_test
    ll_st, ll_pv = sm.stats.lilliefors(data)

    # Shapiro-Wilk
    if len(data) >= 8:
        sw_st, sw_pv = stats.shapiro(data)
    else:
        sw_st, sw_pv = 'n<8', 'n<8'

    #Anderson-Darling
    as_st, ad_pv = sm.stats.normal_ad(np.array(data))

    # D’Agostino-Pearson
    if len(data) >= 20:
        dp_st, dp_pv = stats.normaltest(data)
    else:
        dp_st, dp_pv = 'n<20', 'n<20'
    return ad_pv, as_st, dp_pv, dp_st, ll_pv, ll_st, sw_pv, sw_st


@app.cell
def _(mo):
    intro_why = mo.md('''
    Широкий круг статистических методов требуют для корректной работы проверки нормальности распределения:

    * Дисперсионный анализ относится к группе параметрических методов и поэтому его следует применять только тогда, когда известно или можно принять, что распределение признака является нормальным.

    **Список источников**

    * Суходольский, Г.В. Основы математической статистики для психологов. - Л.: Издателство Ленинградского ун-та. 1972. - 432 с.
    * Шеффе Г. Дисперсионный анализ. - М.: Наука. 1980. - 512 с.
    * Плохинский, Н.А. Наследуемость. Ред.-изд. отдел СО РАН СССР. 1964. - 196 с. (с.34-36)
    * Плохинский, Н.А. Биометрия: учебное пособие. - 2 изд. - М.: Изд-во Московского ун-та. 1970. - 268 с. (с. 71-72)
    * Пустыльник, Е.И. Статистические методы анализа и обработки наблюдений / Е.И. Пустыльник. – М.: Наука, 1968. – 288 с.: ил.
    * D’Agostino, R. B. (1971). An omnibus test of normality for moderate and large sample size. Biometrika, 58, 341-348
    * D’Agostino, R. and Pearson, E. S. (1973). Tests for departure from normality. Biometrika, 60, 613-622.
    * Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). Biometrika, 52(3/4), 591-611
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
    mo.md(f"""
    # Проверка нормальности распределения

    {mo.md("## [bit.ly/normality-test](https://bit.ly/normality-test)").center()}
    """).center()
    return


@app.cell
def _(mo):
    qrcode = mo.md('''
    ![](data:image/png;base64,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ZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJW/UgvgAAAbBJREFUIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJZGGPGVVAk0ASSAJJ4OQEsrBPnn56TwJJIAkkgTEJ/H9yssEppTuFDgAAAABJRU5ErkJggg==)
    ''')
    return (qrcode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Введение""")
    return


@app.cell(hide_code=True)
def _(intro_when, intro_which, intro_why, mo, qrcode):
    def orange(text):
        return f'<span style="color:orange;">{text}</span>'

    mo.accordion({
      orange('Зачем проверять нормальность распределения?')              : intro_why,
      orange('Когда может не сработать центральная предельная теорема?') : intro_when,
      orange('Какой метод проверки выбрать?')                            : intro_which,
      orange('QR-code')                                                  : qrcode,
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
                data = np.random.normal(-3, 1, vol) + np.random.normal(6, 3, vol)
            case 'равномерного распределения': 
                data = np.random.uniform(2,4, vol)
            case 'тестовые данные':
                data = [11, 13, 12, 9, 10, 11, 8, 10, 15, 14, 8, 7, 10, 10, 5, 8]

    return (data,)


@app.cell
def _(mo):
    выводы = mo.md("""
    ## Перспективы marimo для учебного процесса: :rocket:

    | :thumbup: Pros  |
    |: --- |
    | Возможность статического хостинга - не стоит ни копейки (сейчас: GitHub, ?GitVerse) |
    | Возможность реализации сложных алгоритмов, посредством Python-библиотек numpy, scipy, statsmodels и др. |
    | Нативная работа с математическими формулами, через Латех синтаксис |
    | Легкость создания интерфейсов, даже по сравнению со Streamlit и ShinyPy |
    | Современно выглядящие интерфейсные элементы |
    | Возможность использования графических библиотек matplotlib, seaborn, altair, plotly|

    | :thumbdown: Contras |
    |: --- |
    | Необходимо отдельное решение для обратной связи и синхронизации с обучающимися |
    | Трудность создания сложных интерфейсных взаимодействий |
    | Трудности с анимациями (например: пошаговый вывод формул, анимации процессов) |
    """).center()
    return (выводы,)


@app.cell(hide_code=True)
def _():
    ## Методы проверки нормальности распределения
    return


@app.cell(hide_code=True)
def _(mo, orange, summary_html, visual, Плохинский, Пустыльник, выводы):
    mo.accordion({
      orange('Визуальный анализ: ГОСТ Р ИСО 5479-2002')             : visual,
    #  orange('Критерий Шапиро-Уилка (8≤n≤50): ГОСТ Р ИСО 5479-2002'): 'Work in progress',
      orange('Проверка нормальности по Пустыльнику')                : Пустыльник,
      orange('Проверка нормальности по Плохинскому')                : Плохинский,
      orange('Сводная таблица')                                     : summary_html,
      orange('Выводы')                                              : выводы,
      #orange('Сводная md-таблица'                                  : summary,
    }, multiple=False, lazy=True)
    return


if __name__ == "__main__":
    app.run()
