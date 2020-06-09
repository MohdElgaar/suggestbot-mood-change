import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import date, time, datetime

app = dash.Dash(__name__)
app.title = 'Mood Swingers'
server = app.server

placeholder_data = pd.DataFrame({'action': ['Facebook (App)', 'Call',
    'App2', 'App3', 'Youtube (App)', 'Netflix (App)', 'Extra'],
    'type': ['Neg', 'Neg', 'Neg', 'Pos', 'Pos', 'Pos', 'Pos'],
    'time': [time(13, 00), time(15, 0), time(12, 0), time(11, 0),
        time(19, 0), time(17, 0), None],
    'icon': ['fb.png', 'phone.png', 'weather.png', 'workout.png',
        'yt.png', 'netflix.png', None],
    'effects': [4, 2, 1, 3, 5, 1, 2]
        })

placeholder_data_line = pd.DataFrame({
    'time': [time(i) for i in range(9, 22)],
    'valence': [ 3, 5, 6, 7, 4, 2, -1, -4, -2, 0, 2, 3, 3],
    'UV': [0.5, 0.5, 1, 5, 4, 5, 0.5, 0.5, 0.5, 3, 0, 0, 0]})

min_t = placeholder_data_line['time'].min().hour
max_t = placeholder_data_line['time'].max().hour

def fill_table(df, timegiven):
    df = df.loc[df['time'] <= time(timegiven[1])]
    df = df.loc[df['time'] >= time(timegiven[0])]
    pos = df.loc[df['type'] == 'Pos']
    neg = df.loc[df['type'] == 'Neg']

    maxlen = min(len(pos), len(neg))

    rows = []
    if maxlen > 0:
        for p, n in zip(pos['action'], neg['action']):
            rows.append(html.Tr([html.Td([p]), html.Td([n])]))
    if len(pos) > maxlen:
        for p in pos[maxlen:]['action']:
            rows.append(html.Tr([html.Td([p]), ""]))
    elif len(neg) > maxlen:
        for n in neg[maxlen:]['action']:
            rows.append(html.Tr(["", html.Td([n])]))

    return rows

def line_plot(df_line, df_data, timegiven, UV = False):
    df_line = df_line.loc[df_line['time'] <= time(timegiven[1])]
    df_line = df_line.loc[df_line['time'] >= time(timegiven[0])]
    df_data = df_data.loc[df_data['time'] <= time(timegiven[1])]
    df_data = df_data.loc[df_data['time'] >= time(timegiven[0])]

    valence = go.Scatter(x = df_line['time'], y = df_line['valence'],
            name = "Valence")

    df_data.dropna(subset=['time'], inplace=True)
    max_time = timegiven[1]
    min_time = timegiven[0]
    max_valence = max(df_line['valence'])
    min_valence = -5

    layout = {'images': [],
            'paper_bgcolor': 'lightgray'}
    for _, item in df_data.iterrows():
        cur_valence = df_line[df_line['time'] == item['time']].iloc[0]['valence']
        if item['icon']:
            layout['images'].append({
                "x": (item['time'].hour-min_time)/\
                        (max_time-min_time),
                "y": (cur_valence-min_valence)/\
                        (max_valence - min_valence + 1),
                'sizex': 0.08, 'sizey': 0.08,
                'source': "/assets/%s"%item['icon'],
                'xanchor': "left",
                'xref': "paper",
                'yanchor': "center",
                'yref': "paper"})

            layout['images'].append({
                "x": (item['time'].hour-min_time)/\
                        (max_time-min_time),
                "y": (cur_valence-min_valence)/\
                        (max_valence - min_valence + 1),
                'sizex': 0.08, 'sizey': 0.08,
                'source': "/assets/green.png" if item['type'] == 'Pos' else\
                        "/assets/red.png",
                'xanchor': "left",
                'xref': "paper",
                'yanchor': "center",
                'yref': "paper",
                'opacity': 0.3})

    if UV:
        UV_exposure = go.Scatter(x = df_line['time'], y = df_line['UV'], name = "UV")
        return {'data': [valence, UV_exposure],
                'layout': layout}
    else:
        return {'data': [valence], 'layout': layout}


def pie_chart(df, selected = 'Pos'):
    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    chosen = df[df['type'] == selected]
    label = chosen['action']
    value = chosen['effects']
    text = chosen['time'].astype(str)
    pie = {'data': [{'type': 'pie',
                     'name': "Pie-charts",
                     'text': text,
                     'labels': label,
                     'values': value,
                     'textinfo': 'label+percent',
                     'direction': 'clockwise',
                     'marker': {
                         'colors': ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']}}],
           'layout': {"title": "Pie-chart (%s)" %selected}}
    fig = go.Figure(pie)
    fig.update_traces(hoverinfo='text', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=3)))
    fig.update_layout(
        autosize = False,
        width = 500,
        height = 500,
        margin=dict(l=100, r=10, t=10, b=10),
        paper_bgcolor="LightSteelBlue",
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    return fig



app.layout = html.Div(id='bottom', children = [
    html.Div(className = 'right', children = [
        html.Table(className = 'content-table', children = [
            html.Thead([
                html.Tr([
                    # html.Th(id='pos', children = ["Positive"]),
                    html.Th([
                    html.Button("Positive", id='Pos', n_clicks=0),
                    ]),
                    # html.Th(id='neg', children = ["Negative"])
                    html.Th([
                    html.Button("Negative", id='Neg', n_clicks=0)
                    ]),
                ])
                ]),
            html.Tbody(id="table")
            ]),
        html.Div(className = 'tooltip', children = [
            html.Div(className = 'help', children = ["?"]),
            html.Div(className = 'tooltiptext', children = [
                'Click on Positive or Negative'])
            ]),
        ]),
    html.Div(id='line2', children = [
        html.Div(id='line1', className = 'left', children = [
            html.H1("Positive & Negative Mood Influencers"),
            html.Div(id='plot', children=[dcc.Graph(id = 'line_plot')]),

            html.Div([
                dcc.RangeSlider(
                    id='slider',
                    min=min_t,
                    max=max_t,
                    step=1,
                    value=[9, 21],
                    marks={
                        1: '1AM',
                        2: '2AM',
                        3: '3AM',
                        4: '4AM',
                        5: '5AM',
                        6: '6AM',
                        7: '7AM',
                        8: '8AM',
                        9: '9AM',
                        10: '10AM',
                        11: '11AM',
                        12: '12AM',
                        13: '1PM',
                        14: '2PM',
                        15: '3PM',
                        16: '4PM',
                        17: '5PM',
                        18: '6PM',
                        19: '7PM',
                        20: '8PM',
                        21: '9PM',
                        22: '10PM',
                        23: '11PM',
                        24: '12PM'
                    }
                )]),

            html.Center([html.Button("Show UV Exposure", id = "UV_button",
                n_clicks = 0)])
            ], style={'display': 'block'}),
        ],
        style={'display': 'block'}),
    html.Div(className = 'left', children = [html.Center([
    html.Div(id='pie1', children=[dcc.Graph(id='pie-chart1')]
             , style={'display': 'none'}),
    html.Div(id='pie2', children=[dcc.Graph(id='pie-chart2')]
             , style={'display': 'none'})
    ])]),

    ])


@app.callback(
        [Output("line_plot", 'figure'), Output("table", 'children')], [Input("UV_button", "n_clicks"), Input("slider", "value")]
        )
def callback(n_clicks, time):
    if n_clicks % 2 == 0:
        return line_plot(placeholder_data_line, placeholder_data, time, UV=False), fill_table(placeholder_data, time)
    else:
        return line_plot(placeholder_data_line, placeholder_data, time, UV=True), fill_table(placeholder_data, time)


@app.callback(
   [Output('pie1', 'style'), Output('pie-chart1', 'figure'),
       Output('line1', 'style'), Output('Pos', 'style')],
   [Input('Pos', 'n_clicks')])
def pospie(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, pie_chart(placeholder_data), {'display': 'block'}, {'background-color': 'lightgray'}
    else:
        return {'display': 'block'}, pie_chart(placeholder_data, 'Pos'), {'display': 'none'}, {}


@app.callback(
   [Output('pie2', 'style'), Output('pie-chart2', 'figure'),
       Output('line2', 'style'), Output('Neg', 'style')],
   [Input('Neg', 'n_clicks')])
def negpie(n_clicks):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, pie_chart(placeholder_data, 'Neg'), {'display': 'block'}, {'background-color': 'lightgray'}
    else:
        return {'display': 'block'}, pie_chart(placeholder_data, 'Neg'), {'display': 'none'}, {}



if __name__ == '__main__':
    app.run_server(debug=True)
