import boto3
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import date, time, datetime
from data import DataManager
from os.path import isfile

app = dash.Dash(__name__)
app.title = 'Mood Swingers'
server = app.server

datam = DataManager()
UIDs = datam.uids

min_t = placeholder_data_line['time'].min().hour
max_t = placeholder_data_line['time'].max().hour


# placeholder_data = pd.DataFrame({'action': ['Facebook (App)', 'Call',
#     'Weather Change', 'Exercise', 'Youtube (App)', 'Netflix (App)', 'Extra'],
#     'type': ['Neg', 'Neg', 'Neg', 'Pos', 'Pos', 'Pos', 'Pos'],
#     'time': [time(13, 00), time(15, 0), time(12, 0), time(11, 0),
#         time(19, 0), time(17, 0), None],
#     'icon': ['fb.png', 'phone.png', 'weather.png', 'workout.png',
#         'yt.png', 'netflix.png', None],
#     'effects': [4, 2, 1, 3, 5, 1, 2]
#         })

# placeholder_data_line = pd.DataFrame({
#     'time': [time(i) for i in range(9,22)],
#     'valence': [3, 5, 6, 7, 4, 2, -1, -4, -2, 0, 2, 3, 3],
#     'UV': [0.5, 0.5, 1, 5, 4, 5, 0.5, 0.5, 0.5, 3, 0, 0, 0]})

def fill_table(df):
    if df is None:
        return []
    
    pos, neg = [], []
    for i,val in enumerate(df['types']):
        if val == 'Pos':
            pos.append(df['actions'][i])
        else:
            neg.append(df['actions'][i])

    maxlen = min(len(pos), len(neg))

    rows = []

    for p,n in zip(pos, neg):
        rows.append(html.Tr([html.Td([p]), html.Td([n])]))

    if len(pos) > maxlen:
        for p in pos[maxlen:]:
            rows.append(html.Tr([html.Td([p]), ""]))
    elif len(neg) > maxlen:
        for n in neg[maxlen:]:
            rows.append(html.Tr([html.Td([n]), ""]))

    return rows

def interpolate_time(df, time):
    df = df.set_index('time')
    x = pd.Series(index = [time], data = [np.nan])
    y = pd.concat([x,df]).sort_index().interpolate('time')
    return y[time:time]

def line_plot(df_line, df_data, UV = False):
    if df_line is None or df_data is None:
        return {'data': [None]}

    uv_data = df_line[1]
    df_line = df_line[0]

    valence = go.Scatter(x = df_line['time'], y = df_line['Valence'],
            name = "Valence")

    # df_data.dropna(subset=['time'], inplace=True)
    max_time = 21
    min_time = 9
    max_valence = max(df_line['Valence'])
    min_valence = -2

    layout = {'images': [],
            'paper_bgcolor': 'lightgray'}
    for i in range(len(df_data['actions'])):
        name = df_data['actions'][i]
        if not name in df_data['meta']:
            continue
        time = pd.to_datetime(df_data['meta'][name]['times'][0], unit='ns')
        cur_valence = interpolate_time(df_line, time)['Valence']

        icon_file = 'assets/%s.png'%name
        if isfile(icon_file):
            layout['images'].append({
                "x": (time.hour-min_time)/\
                        (max_time-min_time),
                "y": (cur_valence-min_valence)/\
                        (max_valence - min_valence + 1),
                'sizex': 0.08, 'sizey': 0.08,
                'source': icon_file,
                'xanchor': "left",
                'xref': "paper",
                'yanchor': "center",
                'yref': "paper"})

            layout['images'].append({
                "x": (time.hour-min_time)/\
                        (max_time-min_time),
                "y": (cur_valence-min_valence)/\
                        (max_valence - min_valence + 1),
                'sizex': 0.08, 'sizey': 0.08,
                'source': "/assets/green.png" if df_data['types'][i] == 'Pos' else\
                        "/assets/red.png",
                'xanchor': "left",
                'xref': "paper",
                'yanchor': "center",
                'yref': "paper",
                'opacity': 0.3})

            layout['yaxis'] = {'title': 'Valence'}

    if UV:
        UV_exposure = go.Scatter(x = uv_data.index, y = uv_data, name = "UV", yaxis='y2')
        layout['yaxis2'] = {'title': 'UV', 'overlaying': 'y', 'side': 'right'}
        return {'data': [valence, UV_exposure],
                'layout': layout}
    else:
        return {'data': [valence], 'layout': layout}


# def pie_chart(df, selected = 'Pos'):
#     colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
#     chosen = df[df['type'] == selected]
#     label = chosen['action']
#     value = chosen['effects']
#     text = chosen['time'].astype(str)
#     pie = {'data': [{'type': 'pie',
#                      'name': "Pie-charts",
#                      'text': text,
#                      'labels': label,
#                      'values': value,
#                      'textinfo': 'label+percent',
#                      'direction': 'clockwise',
#                      'marker': {
#                          'colors': ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']}}],
#            'layout': {"title": "Pie-chart (%s)" %selected}}
#     fig = go.Figure(pie)
#     fig.update_traces(hoverinfo='text', textfont_size=20,


def pie_chart(df, selected = 'Pos'):
    if df is None:
        return None

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

    label, value = [], []
    for i in range(len(df['actions'])):
        if df['types'][i] == selected:
            label.append(df['actions'][i])
            value.append(df['effects'][i])

    pie = go.Pie(labels=label, values=value, textinfo='label+percent', name = "pie")
    layout = go.Layout(
        {
            "title": "Pie-chart (%s)" %selected
        }
    )
    fig = go.Figure(data=[pie], layout = layout)
    fig.update_traces(hoverinfo='value', textfont_size=15,
                      marker=dict(colors=colors, line=dict(color='#000000', width=3)))
    fig.update_layout(
        autosize = False,
        width = 500,
        height = 500,
        margin=dict(l=100, r=10, t=10, b=10),

        # paper_bgcolor="LightSteelBlue",
        # hoverlabel=dict(
        #     bgcolor="white",
        #     font_size=16,
        #     font_family="Rockwell"
        # )
    # )
        paper_bgcolor="LightSteelBlue"
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
            html.Tbody(id='table-body')
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
            dcc.Dropdown(id='uid', options=[{'label': x, 'value': x} for x in UIDs], value=UIDs[0]),
            dcc.Loading(id="loading-1", type="default", children=html.Div(id="loading-output-1")),
            html.Div(id='plot', children=[dcc.Graph(id = 'line_plot')]),
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
    dcc.Store(id='sync')
    ])

@app.callback(
    [Output('sync', 'data'),Output("loading-output-1", "children")],
    [Input('uid','value')])
def sync(uid):
    print("[INFO] Starting sync")
    datam.update(uid)
    print("[INFO] End sync")
    return True, True

@app.callback(
        [Output('table-body', 'children')],
        [Input('sync','data')])
def callback_table(_):
    return [fill_table(datam.data)]

@app.callback(
        Output("line_plot", 'figure'),
        [Input("UV_button", "n_clicks"), Input('sync','data')]
        )
def callback(n_clicks, _):
    if n_clicks % 2 == 0:
        return line_plot(datam.line_data, datam.data, UV=False)
    else:
        return line_plot(datam.line_data, datam.data, UV=True)


@app.callback(
   [Output('pie1', 'style'), Output('pie-chart1', 'figure'),
       Output('line1', 'style'), Output('Pos', 'style')],
   [Input('Pos', 'n_clicks'), Input('sync','data')])
def pospie(n_clicks, _):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, pie_chart(datam.data), {'display': 'block'}, {'background-color': 'lightgray'}
    else:
        return {'display': 'block'}, pie_chart(datam.data, 'Pos'), {'display': 'none'}, {}


@app.callback(
   [Output('pie2', 'style'), Output('pie-chart2', 'figure'),
       Output('line2', 'style'), Output('Neg', 'style')],
   [Input('Neg', 'n_clicks'), Input('sync','data')])
def negpie(n_clicks, _):
    if n_clicks % 2 == 0:
        return {'display': 'none'}, pie_chart(datam.data, 'Neg'), {'display': 'block'}, {'background-color': 'lightgray'}
    else:
        return {'display': 'block'}, pie_chart(datam.data, 'Neg'), {'display': 'none'}, {}

if __name__ == '__main__':
    app.run_server(debug=True)
