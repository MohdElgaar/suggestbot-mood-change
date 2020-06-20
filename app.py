import os
import boto3
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from datetime import date, time, datetime, timedelta
from data import DataManager
from os.path import isfile

app = dash.Dash(__name__)
app.title = 'Mood Swingers'
server = app.server

datam = DataManager()
UIDs = datam.uids

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
    datam.n_pos = len(pos)
    datam.n_neg = len(neg)

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

def line_plot(df_line, df_data, UV = False, date=None):
    if df_line is None or df_data is None:
        return {'data': [None]}

    uv_data = df_line[1]
    df_line = df_line[0]

    if date:
        date_dt = datetime.strptime(date, '%Y-%m-%d')
        df_line = df_line.loc[(df_line['time'] >= date_dt) & (df_line['time'] <= date_dt + timedelta(days=1))]
        uv_data = uv_data.loc[(uv_data.index >= date_dt) & (uv_data.index <= date_dt + timedelta(days=1))]

    valence = go.Scatter(x = df_line['time'], y = df_line['Valence'],
            name = "Valence")

    layout = {'images': [],
            'paper_bgcolor': 'lightgray',
            'xaxis': {'showgrid': False,
                'title': 'Time'},
            'yaxis': {'title': 'Emotion Polarity (valence)'}
            }

    if date is not None:
        max_time = max(df_line['time']).hour
        min_time = min(df_line['time']).hour
        max_valence = max(df_line['Valence'])
        min_valence = min(df_line['Valence'])

        for i in range(len(df_data['actions'])):
            name = df_data['actions'][i]
            if not name in df_data['meta']:
                continue
            time = pd.to_datetime(df_data['meta'][name]['times'][0], unit='ns')
            if time < date_dt or time > date_dt + timedelta(days=1):
                continue
            
            cur_valence = interpolate_time(df_line, time).iloc[0]['Valence']
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


    if UV:
        UV_exposure = go.Scatter(x = uv_data.index, y = uv_data, name = "UV", yaxis='y2')
        layout['yaxis2'] = {'title': 'UV', 'overlaying': 'y', 'side': 'right'}
        return {'data': [valence, UV_exposure],
                'layout': layout}
    else:
        return {'data': [valence], 'layout': layout}

def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

def pie_chart(df, selected = 'Pos'):
    if df is None:
        return None

    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    name = {'Pos': 'Positive', 'Neg': 'Negative'}

    label, value, text = [], [], []
    for i in range(len(df['actions'])):
        if df['types'][i] == selected:
            l= df['actions'][i]
            label.append(l)
            value.append(df['effects'][i])
            s = ""
            if l in df['meta']:
                for j in range(min(len(df['meta'][l]['times']),len(df['meta'][l]['durations']))):
                    time = str(pd.to_datetime(df['meta'][l]['times'][j]))[:16]
                    duration = timedelta(milliseconds=df['meta'][l]['durations'][j])
                    duration = strfdelta(duration, "duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    s += "{} ({}) <br>".format(time, duration)
            text.append(s)

    pie = go.Pie(labels=label, values=value,
            text = text,
            textinfo = 'percent+label',
            hovertemplate = "<b>%{label}</b><br><br>" + "%{text}",
            name = "pie")
    layout = go.Layout(
        {
            "title": "Pie-chart (%s)" %name[selected]
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
                'Click on Positive or Negative for more information!'])
            ]),
        ]),
    html.Div(id='line2', children = [
        html.Div(id='line1', className = 'left', children = [
            html.H1("Positive & Negative Mood Influencers"),
            dcc.Dropdown(id='uid', options=[{'label': x, 'value': x} for x in UIDs], value=UIDs[0], clearable=False),
            dcc.Loading(id="loading-1", type="default", children=html.Div(id="loading-output-1")),
            html.Div(id='plot', children=[dcc.Graph(id = 'line_plot')]),
            dcc.DatePickerSingle(id="date_pick", initial_visible_month=datetime(2019,5, 1),),
            html.Br(),
            html.Button("Clear Date", id = "date_clear", n_clicks = 0),
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
        [Output('date_pick', 'min_date_allowed'),
            Output('date_pick', 'max_date_allowed'),
            Output('date_pick', 'initial_visible_month')],
        [Input('sync','data')])
def callback_datepick(_):
    df_line = datam.line_data[0]
    max_time = max(df_line['time']).date()
    min_time = min(df_line['time']).date()
    return min_time, max_time, min_time

@app.callback(
        Output("date_pick", "date"),
        [Input("date_clear", "n_clicks")]
        )
def update_date(_):
    return None

@app.callback(
        Output("line_plot", 'figure'),
        [Input("UV_button", "n_clicks"), Input('sync','data'),
            Input("date_pick", "date")]
        )
def callback(n_clicks, _, date):
    if n_clicks % 2 == 0:
        return line_plot(datam.line_data, datam.data, UV=False, date=date)
    else:
        return line_plot(datam.line_data, datam.data, UV=True, date=date)


@app.callback(
   [Output('pie1', 'style'), Output('pie-chart1', 'figure'),
       Output('line1', 'style'), Output('Pos', 'style')],
   [Input('Pos', 'n_clicks'), Input('sync','data')])
def pospie(n_clicks, _):
    if n_clicks % 2 == 0 or datam.n_pos == 0:
        return {'display': 'none'}, pie_chart(datam.data), {'display': 'block'}, {'background-color': 'gray'}
    else:
        return {'display': 'block'}, pie_chart(datam.data, 'Pos'), {'display': 'none'}, {}


@app.callback(
   [Output('pie2', 'style'), Output('pie-chart2', 'figure'),
       Output('line2', 'style'), Output('Neg', 'style')],
   [Input('Neg', 'n_clicks'), Input('sync','data')])
def negpie(n_clicks, _):
    if n_clicks % 2 == 0 or datam.n_neg == 0:
        return {'display': 'none'}, pie_chart(datam.data, 'Neg'), {'display': 'block'}, {'background-color': 'gray'}
    else:
        return {'display': 'block'}, pie_chart(datam.data, 'Neg'), {'display': 'none'}, {}

if __name__ == '__main__':
    app.run_server(debug=True)
