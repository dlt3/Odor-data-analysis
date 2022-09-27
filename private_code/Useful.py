"""여러개의 데이터 프레임 가로 방향으로 출력"""
### multi_table([df1, df2])
def multi_table(table_list):
    from IPython.core.display import HTML
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )




"""데이터 프레임 특정 조건(1개) 셀 에 색칠하기"""
def color_at_df(df, condition, color) : 
    color = 'background-color: ' + color
    new_df = df.style.apply(lambda L: [ color if (i in condition) or (i == condition) else "" for i in L ], axis=0)
    return new_df
