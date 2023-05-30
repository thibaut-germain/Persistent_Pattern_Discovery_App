from dash import Dash, dcc, html, Input, Output, Patch,State 
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import io

from neighborhood import KNN
from persistence import ThresholdPersistenceMST,BasicPersistence
from post_processing import PostProcessing
from utils import get_relative_lag,get_barycenter

RGB_COLORS = [px.colors.hex_to_rgb(color) for color in px.colors.qualitative.Plotly]
SIGNAL_COLOR = "rgba(0,0,0,0.2)"


def ktanh(x,alpha,beta):
    norm_factor = np.tanh(beta**2*alpha) - np.tanh(-alpha*(4-beta**2))
    dists =  np.tanh(beta**2*alpha) - np.tanh(-alpha*(x**2-beta**2))
    y = 2 * np.sqrt(dists/norm_factor)
    return y

def birth_cut_dct(persistence,p_cut,b_cut,alpha,beta)->None: 
    """Compute the dictionnary of birth cut per motif
    """
    pers  = persistence.copy()
    pers[:,:-1] = ktanh(pers[:,:-1],alpha,beta)
    mask = pers[:,1] - pers[:,0] > p_cut
    b_cut_dct = {}
    for line in pers[mask]: 
        if line[0]<= b_cut:
            b_cut_dct[int(line[-1])] = min(line[1],b_cut)
    return b_cut_dct

def persistence_with_thresholds(mst,p_cut,b_cut,b_cut_dct,alpha,beta)->None:
    """Compute persitence based on given thresholds.
    """ 
    tpmst = ThresholdPersistenceMST(persistence_threshold=p_cut,birth_threshold=b_cut,birth_individual_threshold=b_cut_dct) 
    mst = mst.copy()
    mst[:,-1] = ktanh(mst[:,-1],alpha,beta)
    tpmst.fit(mst)
    return tpmst.connected_components_

def fit_post_processing(mp,connected_components,wlen,alpha,beta,remove_outlier,outlier_threshold): 
    #get idx_lst and birth profile
    idx_lst = []
    for seed,idxs in connected_components.items(): 
        idx_lst.append(idxs)
    mp = mp.copy()
    mp = ktanh(mp,alpha,beta)
    pp = PostProcessing(wlen,None,remove_outlier,outlier_threshold/100.0)
    pp.fit(idx_lst,mp)
    return pp.prediction_birth_list_

def fit_on_submit(mst,mp,persistence,wlen,p_cut,b_cut,alpha,beta,remove_outlier,outlier_threshold): 
    b_cut_dct = birth_cut_dct(persistence,p_cut,b_cut,alpha,beta)
    connected_components = persistence_with_thresholds(mst,p_cut,b_cut,b_cut_dct,alpha,beta)
    pred,birth = fit_post_processing(mp,connected_components,wlen,alpha,beta,remove_outlier,outlier_threshold)
    return pred,birth

def display_signal(signal): 
    patch = Patch()
    data = [dict(type= "lines",y = signal,mode = "lines" ,line = dict(color = SIGNAL_COLOR),showlegend=False)]
    patch["data"] = data
    return patch

def update_signal_figure(signal,pred,birth): 
    
    min_birth = np.min(np.hstack(birth))
    max_birth = np.max(np.hstack(birth))
    transparancy = lambda x : 1 - 0.8 * ((x - min_birth)/(max_birth-min_birth))
    
    patch = Patch()
    data = [dict(type= "lines",y = signal,mode = "lines" ,line = dict(color = SIGNAL_COLOR),showlegend=False)]
    shapes = []

    for key,(t_pred,t_birth) in enumerate(zip(pred,birth)): 
        for i,((s,e),b)in enumerate(zip(t_pred,t_birth)):
            color_idx = key%len(RGB_COLORS)
            color = f"rgba({RGB_COLORS[color_idx][0]},{RGB_COLORS[color_idx][1]},{RGB_COLORS[color_idx][2]},{transparancy(b)})"
            t_dct = dict(type = "lines", x = np.arange(s,e),y = signal[s:e],mode = "lines",line = dict(color = color),name = f"motif {key+1}",legendgroup=str(key),showlegend= i==0)
            data.append(t_dct)
            t_dct = dict(type = "line", x0 = s,x1 = s,y0 = 0,y1 = 1, yref = "y domain", line = dict(color = "black", dash = "dash", width = 1))
            shapes.append(t_dct)

    patch["data"] = data
    patch["layout"]["shapes"] = shapes
    return patch

def get_motif_plots(signal,pred,wlen): 
    pattern_lst = []
    for i,lst in enumerate(pred): 
        #dataset & lags
        dataset = []
        for start,end in lst: 
            dataset.append(signal[start:end])
        dataset = [(ts - np.mean(ts)/np.std(ts)) for ts in dataset]
        lags = get_relative_lag(dataset,wlen)
        #index & color
        idx = i%3 +1
        if idx ==1: 
            xaxis = "x"
            yaxis = "y"
        else: 
            xaxis = f"x{idx}"
            yaxis = f"y{idx}"
        idx_color = i%len(RGB_COLORS)
        color = f"rgb({RGB_COLORS[idx_color][0]},{RGB_COLORS[idx_color][1]},{RGB_COLORS[idx_color][2]})"
        

        t_lst = []
        for j,ts in enumerate(dataset): 
            x = np.arange(lags[j],lags[j]+ len(ts))
            t_dct = dict(type = "lines", x =x.tolist(),y=ts.tolist(),mode = "lines", line = dict(color = SIGNAL_COLOR), showlegend = False, xaxis = xaxis, yaxis = yaxis)
            t_lst.append(t_dct)
        pattern_lst.append(t_lst)
        
        x,avg_pattern = get_barycenter(dataset,lags)
        t_dct = dict(type = "lines", x =x.tolist(),y=avg_pattern.tolist(),mode = "lines", line = dict(color = color), name = f"motif {i+1}", showlegend = True, xaxis = xaxis, yaxis = yaxis)
        pattern_lst[-1].append(t_dct)
    return pattern_lst

def update_motif_figure(trace_lst): 
    patch = Patch()
    patch["data"] = []
    for lst in trace_lst: 
        patch["data"] +=lst
    return patch




app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    
    html.Div([
        dcc.Store(id = "signal_store"),
        dcc.Store(id = "mst_store"),
        dcc.Store(id = "mp_store"),
        dcc.Store(id = "persistence_store"),
        dcc.Store(id = "motif_plot_store"),
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload_data',
                    children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
                    style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                )
            ]),
            dbc.Col([
                html.Div(id="upload_status")
            ]),
            dbc.Col([
                dbc.Input(id="window_length",type = "number", placeholder="Window length")
            ]),
            dbc.Col([
                dbc.Input(id="n_neighbor",type = "number", placeholder="number of neighbors")
            ]),
            dbc.Col([
                html.Div(dbc.Button("Run",id="run"),className="text-center")
            ]),
        ],className="bg-light", align="center"),
        dbc.Row([
            dbc.Col([
                html.Div("Parameters",style={"font-weight": "bold"}),
                dbc.Row([
                    dbc.Col(html.Div("Persitent cut", style={"writing-mode" : "vertical-rl", "transform" : "rotate(-180deg)"}),width = "auto", style = {"width" : "10px"}, align="center"),
                    dbc.Col(dcc.Slider(0,2,marks=None,value= 0,vertical =True,id = "p_cut_slider", verticalHeight = 400),width = "auto", style = {"width" : "10px"}),
                    dbc.Col([
                        dcc.Graph(
                            id = "persistent_diagram",
                            figure = {
                                "data" :[
                                    dict(
                                    type = "scatter",
                                    x = [],
                                    y = [],
                                    mode = "markers"
                                    ),
                                    dict(
                                    type = "scatter",
                                    x = [0,2,2],
                                    y = [0,2,0],
                                    fill = "toself",
                                    marker=dict(size=1,color = "black"),
                                    hovertemplate='<extra></extra>',
                                    ),

                                ],
                                "layout" : dict(
                                    margin =dict(l=20,r=20,t=20,b=20),
                                    xaxis = dict(range = [-0.05,2.05]),
                                    yaxis = dict(range = [-0.05,2.05]),
                                    height = 400,
                                    width = 400,
                                    showlegend = False,
                                    shapes = [
                                        dict(type = "line",
                                            x0=0,
                                            x1=0,
                                            xref = "x",
                                            y0=0,
                                            y1=2,
                                            yrel = "y",
                                            line = dict(width = 1.5, color ='red' )
                                            ),
                                        dict(type = "line",
                                            x0=0,
                                            x1=2,
                                            xref = "x",
                                            y0=0,
                                            y1=2,
                                            yrel = "y",
                                            line = dict(width = 1.5, color ='red' )
                                            ),
                                        dict(type = "line",
                                            x0=0,
                                            x1=2,
                                            xref = "x",
                                            y0=0,
                                            y1=2,
                                            yrel = "y",
                                            line = dict(width = 1.5, color ='black' )
                                            ),
                                        dict(type = "line",
                                            x0=0,
                                            x1=2,
                                            xref = "x",
                                            y0=0,
                                            y1=2,
                                            yrel = "y",
                                            line = dict(width = 1.5, color ='black' )
                                            )
                                    ]     
                                ),
                            },
                        ),
                        dbc.Row(dcc.Slider(0,2,marks=None,value =0, id="b_cut_slider"),style = {"width" : "400px", "height" : "10px"}),
                        dbc.Row(html.Div("Birth cut"),style = {"width" : "400px", "text-align" : "center", "height" : "10px"}, align="center")
                    ],width = "auto",style = {"width" : "450px"}),
                    dbc.Col([
                        dbc.Row([
                        dcc.Graph(
                            id = "distance_plot",
                            figure = {
                                "data" : [
                                    dict(type = "line",y = np.hstack((np.linspace(2,0,101)[:-1],np.linspace(0,2,100))),name = "Euclidean"),
                                    dict(type = "line",y = np.hstack((np.linspace(2,0,101)[:-1],np.linspace(0,2,100))),name = "Tanh")
                                ],
                                "layout" : dict(
                                    margin =dict(l=20,r=20,t=20,b=20),
                                    height = 200,
                                    yaxis = dict(range = [-0.05,2.05]), 
                                    xaxis= dict(tickvals = [  0,  50, 100, 150, 200],ticktext =["2","1","0","1","2"]),
                                    legend = dict(xanchor = "left",x = 0.01, yanchor ="bottom", y = 0.06)    
                                ),
                            },
                        ),
                        dbc.Col([
                            html.Div("Slope"),
                            html.Div("Offset")
                        ],width = "auto"),
                        dbc.Col([
                            dcc.Slider(0.01,10,step =0.01,marks=None,value= 0.01,id = "alpha_slider"),
                            dcc.Slider(0,2,marks=None,value =0, id="beta_slider")
                        ]),
                        ]),
                        html.Hr(),
                        dcc.Checklist([" Remove outliers"], id = "Remove_outlier"),
                        dbc.Row([
                            dbc.Col(html.Div("Threshold"),width = 6),
                            dbc.Col(dcc.Dropdown([f"{i}%" for i in range(1,100)], "25%",id="threshold_outlier"),width=6)
                        ],align="center"),
                        html.Hr(),
                        dbc.Row([
                            html.Div("",style={"height" : "10px"}),
                            dbc.Col(html.Div(dbc.Button("Apply change",id="apply_change"),className="text-center"))
                    ]),
                    ]),
                        ])
            ],width = 6, class_name="border border-dark rounded"),
            dbc.Col([
                html.Div("Motifs",style={"font-weight": "bold"}),
                dbc.Row([
                    dbc.Col([
                    dcc.Dropdown(["Motifs 1 to 3"],value="Motifs 1 to 3",id="motif_selection"),
                    dcc.Graph(
                        id="motif_plot",
                        figure=dict(
                            data = [],
                            layout = {
                                'template': '...',
                                'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0],'matches': 'x3', 'showticklabels': False},
                                'xaxis2': {'anchor': 'y2', 'domain': [0.0, 1.0],'matches': 'x3', 'showticklabels': False},
                                'xaxis3': {'anchor': 'y3', 'domain': [0.0, 1.0]},
                                'yaxis': {'anchor': 'x', 'domain': [0.7333333333333333, 1.0]},
                                'yaxis2': {'anchor': 'x2', 'domain': [0.36666666666666664, 0.6333333333333333]},
                                'yaxis3': {'anchor': 'x3', 'domain': [0.0, 0.26666666666666666]}
                            }
                        )
                    )
                ],width = 12),
                ])
            ],width = 6, class_name="border border-dark rounded"),
        ]),
        dbc.Row([
            html.Div("Signal",style={"font-weight": "bold"}),
            dcc.Graph(
                id="signal_plot",
                figure=dict(
                    layout = dict(
                        margin =dict(l=20,r=20,t=20,b=20),
                        xaxis = dict(rangeselector=dict(buttons=dict(step="all")),rangeslider=dict(visible=True))
                    )
                )
            
            )
        ],class_name="border border-dark rounded"),
    ]),
    fluid = True,
)

@app.callback(
    Output(component_id="signal_store", component_property="data"),
    Output(component_id="upload_status", component_property="children"),
    Input(component_id="upload_data",component_property="contents"),
    Input(component_id="upload_data",component_property="filename"),
)
def upload_data(content,filename):
    if content is None: 
        raise PreventUpdate
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try: 
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),header=None)
        elif "txt" in filename: 
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),header=None)
        else: 
            return None, "Format not supported."
    except: 
        return None, "There was an error processing this file."
    signal = df.values[:,-1].tolist()
    json_signal = json.dumps(signal)
    return json_signal,f"{filename} uploaded"

@app.callback(
        Output(component_id="mst_store", component_property="data"),
        Output(component_id="mp_store", component_property="data"),
        Output(component_id="persistence_store", component_property="data"),
        Input(component_id="run", component_property="n_clicks"),
        State(component_id="window_length", component_property="value"),
        State(component_id="n_neighbor", component_property="value"),
        State(component_id="signal_store", component_property="data")
)
def store_mst_mp(n_clicks,wlen,n_neighbor,json_signal):
    if n_clicks is None: 
        raise PreventUpdate
    signal = np.array(json.loads(json_signal))
    knn = KNN(n_neighbor,wlen,"UnitEuclidean")
    knn.fit(signal)
    bp = BasicPersistence()
    bp.fit(knn.filtration_)
    mst = bp.mst_.tolist()
    json_mst = json.dumps(mst)
    mp = knn.dists_[:,0].tolist()
    json_mp = json.dumps(mp)
    persistence = bp.get_persistence(True)
    persistence[-1,1] = np.max(persistence[:-1,1])
    mask = persistence[:,1]- persistence[:,0] > 0 
    persistence = persistence[mask].tolist()
    json_persistence = json.dumps(persistence)
    return json_mst, json_mp,json_persistence



@app.callback(
    Output(component_id = "persistent_diagram", component_property = "figure"),
    Input(component_id = "b_cut_slider", component_property = "value"),
    Input(component_id = "p_cut_slider", component_property = "value"),
    Input(component_id = "alpha_slider", component_property = "value"),
    Input(component_id = "beta_slider", component_property = "value"),
    Input(component_id="persistence_store", component_property="data")
)
def update_birth_cut(b_cut,p_cut,alpha,beta,json_persistence): 
    if (json_persistence is None)*(p_cut == 0): 
        raise PreventUpdate
    persistence = np.array(json.loads(json_persistence))
    patch = Patch()
    #update cuts
    patch["layout"]["shapes"][0]["x0"] = b_cut
    patch["layout"]["shapes"][0]["x1"] = b_cut
    patch["layout"]["shapes"][1]["x1"] = 2-p_cut
    patch["layout"]["shapes"][1]["y0"] = p_cut
    #update data layout distance
    y = ktanh(persistence[:,:-1],alpha,beta)
    patch["data"][0]["x"] = y[:,0] 
    patch["data"][0]["y"] = y[:,1] 
    return patch

@app.callback(
    Output(component_id = "distance_plot", component_property = "figure"),
    Input(component_id = "alpha_slider", component_property = "value"),
    Input(component_id = "beta_slider", component_property = "value")
)
def update_birth_cut(alpha,beta): 
    x = np.hstack((np.linspace(2,0,101)[:-1],np.linspace(0,2,100)))
    y = ktanh(x,alpha,beta)
    patch = Patch()
    patch["data"][1]["y"] = y
    return patch

@app.callback(
    Output(component_id="signal_plot", component_property="figure"),
    Output(component_id="motif_selection",component_property="options"),
    Output(component_id="motif_selection",component_property="value"),
    Output(component_id="motif_plot_store", component_property="data"),
    Input(component_id="apply_change", component_property="n_clicks"),
    Input(component_id="signal_store",component_property="data"),
    State(component_id = "b_cut_slider", component_property = "value"),
    State(component_id = "p_cut_slider", component_property = "value"),
    State(component_id = "alpha_slider", component_property = "value"),
    State(component_id = "beta_slider", component_property = "value"),
    State(component_id="Remove_outlier",component_property="value"),
    State(component_id="threshold_outlier",component_property="value"),
    State(component_id="mst_store", component_property="data"),
    State(component_id="mp_store",component_property="data"),
    State(component_id="persistence_store", component_property="data"),
    State(component_id="window_length", component_property="value")
)
def update_signal_motif_plots(n_clicks,json_signal,b_cut,p_cut,alpha,beta,remove_outlier,outlier_threshold,json_mst,json_mp,json_persistence,wlen): 
    if json_signal is None: 
        raise PreventUpdate
    if remove_outlier is not None: 
        remove_outlier = True
    else: 
        False
    signal = np.array(json.loads(json_signal))
    patch_signal = display_signal(signal)
    try:
        mst = np.array(json.loads(json_mst))
        mp = np.array(json.loads(json_mp))
        persistence = np.array(json.loads(json_persistence))
        pred,birth = fit_on_submit(mst,mp,persistence,wlen,p_cut,b_cut,alpha,beta,remove_outlier,int(outlier_threshold[:-1]))
        patch_signal = update_signal_figure(signal,pred,birth)
        n_drop = int(len(pred)/3)  + (len(pred)%3 > 0)
        options = [f"Motifs {3*i+1} to {3*i+3}" for i in range(n_drop)]
        motif_plots = get_motif_plots(signal,pred,wlen)
        json_motif = json.dumps(motif_plots)
    except:
        options = ["Motifs 1 to 3"]
        json_motif = None  
    return patch_signal,options,options[0],json_motif

@app.callback(
    Output(component_id="motif_plot",component_property="figure"),
    Input(component_id="motif_selection",component_property="value"),
    State(component_id="motif_plot_store", component_property="data")
)
def update_motif_plot(value,data):
    if data is None: 
        raise PreventUpdate
    lst = value.split(" ")
    start = int(lst[1])-1
    end = int(lst[-1])
    plots = json.loads(data)
    patch = update_motif_figure(plots[start:min(len(plots),end)])
    return patch


if __name__ == '__main__':
    app.run_server(debug=True)

