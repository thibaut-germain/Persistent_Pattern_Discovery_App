from dash import Dash, dcc, html, Input, Output, Patch,State, ctx 
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import io
from scipy.stats import norm
import os

from neighborhood import KNN
from persistence import ThresholdPersistenceMST,BasicPersistence
from post_processing import PostProcessing
from utils import get_relative_lag,get_barycenter
from threshold import otsu_jump


RGB_COLORS = [px.colors.hex_to_rgb(color) for color in px.colors.qualitative.Plotly]
SIGNAL_COLOR = "rgba(0,0,0,0.2)"


def ktanh(x,alpha,beta):
    norm_factor = np.tanh(beta**2*alpha) - np.tanh(-alpha*(4-beta**2))
    dists =  np.tanh(beta**2*alpha) - np.tanh(-alpha*(x**2-beta**2))
    y = 2 * np.sqrt(dists/norm_factor)
    return y

def cs_div(X,sigma=0.1): 
    # X shape (n_ts,n_alpha,n_beta)
    c1 = -np.log(np.sum(norm.cdf((2-X)/sigma) + norm.cdf(X/sigma),axis=0))
    c2 = 0.5*np.log(np.sum(np.exp(-(X[:,None,:,:]-X[None,:,:])**2/(4*sigma**2)),axis=(0,1)))
    c3 = -0.5*np.log(np.sqrt(np.pi)*sigma)
    return c1 + c2 + c3

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
    m_b,m_d = persistence[-1,:-1]
    if (p_cut <= m_d-m_b)*(b_cut>=m_b):
        b_cut_dct = birth_cut_dct(persistence,p_cut,b_cut,alpha,beta)
        connected_components = persistence_with_thresholds(mst,p_cut,b_cut,b_cut_dct,alpha,beta)
        pred,birth = fit_post_processing(mp,connected_components,wlen,alpha,beta,remove_outlier,outlier_threshold)
        return pred,birth
    else: 
        return None,None

def display_signal(signal): 
    patch = Patch()
    data = [dict(type= "lines",y = signal,mode = "lines" ,line = dict(color = SIGNAL_COLOR),showlegend=False)]
    patch["data"] = data
    patch["layout"]["shapes"] = []
    return patch

def update_signal_figure(signal,pred,birth): 

    patch = Patch()
    data = [dict(type= "lines",y = signal,mode = "lines" ,line = dict(color = SIGNAL_COLOR),showlegend=False)]
    try: 
        min_birth = np.min(np.hstack(birth))
        max_birth = np.max(np.hstack(birth))
        transparancy = lambda x : 1 - 0.8 * ((x - min_birth)/(max_birth-min_birth))
        shapes = []
        for key,(t_pred,t_birth) in enumerate(zip(pred,birth)): 
            for i,((s,e),b)in enumerate(zip(t_pred,t_birth)):
                color_idx = key%len(RGB_COLORS)
                color = f"rgba({RGB_COLORS[color_idx][0]},{RGB_COLORS[color_idx][1]},{RGB_COLORS[color_idx][2]},{transparancy(b)})"
                t_dct = dict(type = "lines", x = np.arange(s,e),y = signal[s:e],mode = "lines",line = dict(color = color),name = f"motif {key+1}",legendgroup=str(key),showlegend= i==0)
                data.append(t_dct)
                t_dct = dict(type = "line", x0 = s,x1 = s,y0 = 0,y1 = 1, yref = "y domain", line = dict(color = "black", dash = "dash", width = 1))
                shapes.append(t_dct)
        patch["layout"]["shapes"] = shapes
    except: 
        patch["layout"]["shapes"] = None

    patch["data"] = data
    
    return patch

def get_motif_plots(signal,pred,wlen,remove_outlier, outlier_threshold): 
    pattern_lst = []
    for i,lst in enumerate(pred): 
        #dataset & lags
        dataset = []
        for start,end in lst: 
            dataset.append(signal[start:end])
        dataset = [(ts - np.mean(ts)/np.std(ts)) for ts in dataset]
        if remove_outlier: 
            twlen = int(np.mean([len(ts) for ts in dataset])*(1-outlier_threshold))
        else:
            twlen = np.min([len(ts) for ts in dataset])
        lags = get_relative_lag(dataset,twlen)
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




app = Dash(__name__,external_stylesheets=[dbc.themes.LUX])
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
                html.Div([
                    dbc.Button("Guidelines", id="guidelines",n_clicks=0,color="info", className="me-1"),
                    dbc.Offcanvas(
                        [
                            html.P("This dashboard was designed for interactive and visual discovery of recurrent patterns (also called motifs) of variable length in univariate time series. It interfaces the persistence-based motif discovery algorithm PEPA [1]."),
                            html.P("As depicted in Figure.1, the workflow of PEPA algortihm can be broken down into three main steps:"),
                            html.Ul([
                                html.Li(html.P([html.U("(Step 1), From time series to graph:"), " Transforming a time series into a graph where nodes represent subsequences and edges are weighted with the distance between subsequences. The graph is an adaptation of the k-nearest neighbor graph."])),
                                html.Li(html.P([html.U("(Step 2.a-2.b), Graph clustering:"), " The idendification of clusters is done from a visual summary of the graph through the persistent diagram. The diagram is built such that clusters associated to motifs are located around its top left corner. Further explanation can be found in the section dedicated to graph clustering."])),
                                html.Li(html.P([html.U("(Step 3), From clusters to motifs:"), " Merging temporally adjacent subsequences in each cluster to form the variable length motifs."]))
                            ]),
                            dbc.Card(
                                    [
                                        dbc.CardImg(src=app.get_asset_url("method_overview.pdf"), style={"max-width": "100%"}, top=True),
                                        dbc.CardBody(
                                            html.P("Figure 1: Workflow of PEPA algorithm", className="card-text",style={"text-align" : "center"})
                                        ),
                                    ],
                                    style={"width": "100%"},
                            ),
                            html.P(" "),
                            html.P("The dashboad follows the workflow of PEPA algorithm, it is also decomposed in three main area:"),
                            html.Ul([
                                html.Li(html.P([html.U("Upper:"), " This block is associated to step 1. It is designed to upload a time-series, set parameters dedicated to the graph construction and run the graph construction."])),
                                html.Li(html.P([html.U("Middle-left:")," This block is associated with step 2.a & 2.b. It gives control over the clustering algorithm and it provides feedbacks to refined the clusters."])),
                                html.Li(html.P([html.U("Middle-right & Lower:"), " These two block are associated to the step 3. The lower block displays the signal and highlights the motifs once discovered. The middle-right block displays selected motifs individualy."])),
                                
                            ]),
                            html.P(" "),
                            html.H4("Steps & Blocks description"),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            html.P("Steps to build the graph from a time series are presented in this section. The nodes of the graph correspond to overlapping subsequences of a time series and they all have the same length (called window length). Each nodes is connected to its K nearest neighobrs according to a distance between subsequences. The window length, the number of neighbors and the distance between subsequences are parameters defined by the user. The block inputs form must be fill from left to right with the following procedure:"),
                                            html.Ol([
                                                html.Li(html.P([html.U("Upload a time series:"), " You must drag and drop or selrct a file. The file should correspond to an univariate time series. It must be a one column file in .txt or .csv format with samples separeted with comas (\",\"). Once the file is uploaded, the signal will be displayed in the lower block. Each time you upload a new time series, all parameters of the dashboard will be set to their default value."])),
                                                html.Li(html.P([html.U("Set window length:"), " The window length corresponds to the subquence length. It is an integer that specifies the number of samples within a subsequence. It is recommended to set it to the length of the smallest expected motif. The length of motifs discovered will range approximatly between 50% and 150% of the window length. "])),
                                                html.Li(html.P([html.U("Set the number of neighbor:"), " This parameters specifies the number of connections from a node to its nearest neighbors. A large number of neighbors helps to connected similar subsequences together however it slows down the running time. It is an integer and it is recommended to set it to 5 or 10."])),
                                                html.Li([html.P([html.U("Set the distance between subsequences:"), "The distance between subsequences corresponds to an euclidean distance combined with a subseuquence normalization procedure. There is two normalization available:", html.Ul([html.Li(html.P([html.U("Z-normalization:"), " It removes deformations caused by amplitude and offset shifts."])),html.Li(html.P([html.U("LT-normalization:"), "It removes deformations caused by amplitude, offset and linear shifts. This normalization is well suited for time-series with a trend as it able to remove deformations induced by the trend at the scale of the subsequence. However if a motif is similar to an affine line it will not be detected. It is the default normalization."]))])]),
                                                    html.Div(
                                                        dbc.Card(
                                                                [
                                                                    dbc.CardImg(src=app.get_asset_url("transformations.pdf"), style={"max-width": "100%"}, top=True,className="img-fluid mx-auto d-block"),
                                                                    dbc.CardBody(
                                                                        html.P("Figure 2: Illustration of deformations", className="card-text",style={"text-align" : "center"})
                                                                    ),
                                                                ],
                                                            style={"width": "70%"},
                                                        ),
                                                    className= "d-flex justify-content-center align-items-center"
                                                    )
                                                ]),
                                                html.Li([html.P([html.U("Click on the \"Run\" button:")," It runs the algorithm to construct the graph. Depending on the length of the time series it may take some time for the algorithm to run. While it is running, a spinner is displayed next to the \"Run\" button. At the end of the execution, the message \"Graph construction successfully performed.\" will replace the spinner."])])
                                            ])
                                        ], 
                                        title="1. From time series to graph -- Upper block"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P("This block gives interactive control over the graph clustering algorithm. A cluster is a connected subgraph of the time series graph. Indeed, the time series graph is such that edges with small distances connect subsequences overlapping a motif. Therefore, motifs can be retrieved by searching for subgraphs whose edges have small distances. The persistent-based graph clustering algorithm is well suited to identify such subgraphs. At its core, the algorithm allocates birth and death dates to connected subgraphs. The birth date is the smallest distance within the subgraph, and the death date is the distance associated with the edge that connects the subgraph to another with an earlier birth date. The collection of birth and death dates offers a visual and informative summarization of the time series graph through the persistence diagram (The persistence of a subgraph is its lifespan). The persistent diagram is a scatter plot with birth on the x-axis and death on the y-axis. As depicted in Figure.3, the diagram helps to identify motifs from the rest of the time series. Indeed, subgraphs associated with motifs have early birth and late death, as each motif has several similar occurrences and differs from the rest of the time series. Therefore, subgraphs of motifs are located in the upper left corner of the persistent diagram. On the other hand, random subsequences have late birth and death; they are located at the right of the diagram. To identify the motif area, a threshold on the birth (birth cut) and a threshold on the persistence (persistence cut) threshold must be set."),
                                            html.Div(
                                                dbc.Card(
                                                    [
                                                        dbc.CardImg(src=app.get_asset_url("motif_persistent_diragram_presentation.png"), style={"max-width": "100%"}, top=True,className="img-fluid mx-auto d-block"),
                                                        dbc.CardBody(
                                                            html.P("Figure 3: Persistence diagram & motif area description", className="card-text",style={"text-align" : "center"})
                                                        ),
                                                    ],
                                                    style={"width": "50%"},
                                                ),
                                                className= "d-flex justify-content-center align-items-center"
                                            ),
                                            html.P("Birth cut slider and persistent cut slider must be used to set the motif area. The button \"Automatic Cut\" on the upper right corner of the persistence diagram sets thresholds with a heuristic. The birth cut is set with the Otsu heuristic, and the persistence cut is set with the second-largest persistence gap. A detailed description of the heuristic can be found in [1]."),
                                            html.P("In some cases, points on the persistent diagram concentrate on small areas, making the choice of thresholds hard. To leverage this issue, the profile of the distance measure can be modified using a sigmoid kernel controlled with two parameters. Both parameters help to spread or concentrate points over the persistence diagram and thus ease setting the thresholds. The first parameter controls the slope. A high slope enforces the distance to be minimal only for the most similar subsequences, and the distance between other subsequences is maximal (equal to 2). Conversely, a small slope leads to a distance identical to the Euclidean distance. The second parameter controls the offset. The kernelized distance remains small when the Euclidean distance is below the offset, and the slope strengthens this effect. The right figure displays the modified distance profile, and the sliders below control the slope and offset. The default slope and offset are set so that the kernelized distance matches the Euclidean distance. As a setting helper, the button \"Spread\" sets the slope and the offset such that births are, as much as possible, uniformly distributed on the x-axis."), 
                                            html.P("The block's lower right section helps remove outliers from the motif occurrence sets. When activated, it removes occurrences whose length is lower or greater than a user-defined percentage of the average length observed within the motif occurrence set. In most cases, this cluster post-processing step is not necessary. By default, there is no post-processing."),
                                            html.P("Finally, once all parameters are set, click on \"Apply Change\", and the motifs will be displayed.")
                                        ], title="2. Graph clustering -- Middle-left block"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                          html.P([html.U("Motifs display -- Middle right block:"), " Once the clustering is done, motifs are displayed in the middle right blocks. The number of motifs displayed is at most three, and the displayed motifs are selected from the input form located above. By default, the first three motifs are displayed. All occurrences within a motif set are aligned through cross-correlation and shown in gray. The colored line corresponds to the average motif."]),
                                          html.P([html.U("Time series display -- Lower block:")," Once the time series is uploaded, it is displayed in gray in the lower block. Once the clustering is done, the motifs are also shown. A unique color is attributed to each motif, matching the color in the motif display block. All motif occurrences are highlighted, and a dash vertical line indicates the start of an occurrence. The color transparency informs about the similarity between occurrences; the more transparent a color, the less similar the occurrence is to any other occurrence. It is possible to focus on (respectively remove) a motif by double-clicking (respectively one-clicking) on its legend. Zooming in (respectively zooming out) is also possible by clicking and dragging the desired area (double clicking on the display)."])  
                                        ], 
                                        title="3. Signal and motifs display -- Middle-right & Lower blocks"
                                    ),
                                ],
                                start_collapsed=True,
                            ),
                            html.P(" "),
                            html.P("[1]  thibaut, germain ....")
                            
                        ],
                        id="offcanvas",
                        title = "Guidelines",
                        is_open = False,
                        style={
                            "width": "1000px"
                        }
                    )
                ])
            ]),
            dbc.Col([
                dcc.Upload(
                    id='upload_data',
                    children=html.Div(['Drag and Drop or ',html.U('Select Files')]),
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
                dbc.Input(id="window_length",type = "number", placeholder="window length (n. samples)")
            ]),
            dbc.Col([
                dbc.Input(id="n_neighbor",type = "number", placeholder="number of neighbors")
            ]),
            dbc.Col([
                dbc.Select(id="normalization",options=[{"label": "LT-normalization", "value" : "LTNormalizedEuclidean"},{"label": "Z-normalization", "value" : "UnitEuclidean"}],value="LTNormalizedEuclidean")
            ]),
            dbc.Col([
                html.Div(dbc.Button("Run",id="run"),className="text-center"),
            ]),
            dbc.Col([
                html.Div(dbc.Spinner(html.Div(id="neighbor_status"))),
            ]),
        ],className="bg-light", align="center"),
        dbc.Row([
            dbc.Col([
                html.Div("Settings",style={"font-weight": "bold"}),
                dbc.Row([
                    dbc.Col(html.Div("Persistence diagram"),style = {"width" : "200px", "text-align" : "left"},width="auto"),
                    dbc.Col(html.Div(dbc.Button("Spread",id="spread", size ="sm",color="secondary",outline=True)),style = {"width" : "70px", "text-align" : "center", "height" : "50px"},width="auto"), 
                    dbc.Col(html.Div(dbc.Button("Automatic cut",id="automatic_cut", size ="sm",color="secondary",outline=True)),style = {"width" : "200px", "text-align" : "center", "height" : "50px"},width="auto"), 
                    dbc.Col(html.Div("Distance settings"),style = {"text-align" : "left"}),
                ]),
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
                        dbc.Row(html.Div("Birth cut"),style = {"width" : "400px", "text-align" : "center", "height" : "10px"}, align="center"),
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
                            dbc.Col(html.Div(dbc.Button("Apply change",id="apply_change",size="sm"),className="text-center"))
                    ]),
                    ]),
                        ])
            ],width = 6, class_name="border border-dark rounded"),
            dbc.Col([
                html.Div("Motifs display",style={"font-weight": "bold"}),
                dbc.Row([
                    dbc.Col([
                    dcc.Dropdown(["Motif 1"],value="Motif 1",multi=True,id="motif_selection"),
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
            html.Div("Signal display",style={"font-weight": "bold"}),
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
    Output(component_id="offcanvas",component_property="is_open"),
    Input(component_id="guidelines",component_property="n_clicks"),
    [State(component_id="offcanvas", component_property="is_open")],
)
def toggle_offcanvas(n1,is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output(component_id="signal_store", component_property="data"),
    Output(component_id="upload_status", component_property="children"),
    Output(component_id="Remove_outlier",component_property="value"),
    Output(component_id="threshold_outlier",component_property="value"),
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
    return json_signal,f"{filename} uploaded",[],"25%"

@app.callback(
        Output(component_id="mst_store", component_property="data"),
        Output(component_id="mp_store", component_property="data"),
        Output(component_id="persistence_store", component_property="data"),
        Output(component_id="neighbor_status", component_property="children"),
        Input(component_id="run", component_property="n_clicks"),
        Input(component_id="upload_data",component_property="filename"),
        State(component_id="window_length", component_property="value"),
        State(component_id="n_neighbor", component_property="value"),
        State(component_id="normalization", component_property="value"),
        State(component_id="signal_store", component_property="data")
)
def store_mst_mp(n_clicks,filename,wlen,n_neighbor,normalization,json_signal):
    if (n_clicks is None)*(filename is None): 
        raise PreventUpdate
    if ctx.triggered_id == "run":
        signal = np.array(json.loads(json_signal))
        knn = KNN(n_neighbor,wlen,normalization)
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
        message = "Graph construction sucessfully performed"
        return json_mst, json_mp,json_persistence,message
    else: 
        return None,None,None,None
@app.callback(
        Output(component_id = "alpha_slider", component_property = "value"),
        Output(component_id = "beta_slider", component_property = "value"), 
        Input(component_id="spread",component_property="n_clicks"),
        Input(component_id="upload_data",component_property="filename"),
        State(component_id="persistence_store", component_property="data"),
)    
def spread_persistence_diagram(n_clicks,filename,json_persistence):
    if (json_persistence is None): 
        raise PreventUpdate
    if ctx.triggered_id == "spread": 
        pers = np.array(json.loads(json_persistence))[:,:-1]
        pers = pers[(pers[:,1]-pers[:,0])>0]
        births = pers[:,0]
        sigma = np.linspace(0.01,10,50)
        offset = np.linspace(0,2,50)
        ss,oo = np.meshgrid(sigma,offset)
        births = ktanh(births[:,None,None],ss[None,:,:],oo[None,:,:])
        divs = cs_div(births,0.1)
        idx = np.argmin(divs)
        i,j = np.unravel_index(idx,divs.shape)
        s,o = sigma[j],offset[i]
        return s,o
    else: 
        return 0.01,0.

@app.callback(
    Output(component_id = "p_cut_slider", component_property = "value"),
    Output(component_id = "b_cut_slider", component_property = "value"),
    Input(component_id = "automatic_cut", component_property = "n_clicks"),
    Input(component_id="upload_data",component_property="filename"),
    State(component_id="persistence_store", component_property="data"),
    State(component_id = "alpha_slider", component_property = "value"),
    State(component_id = "beta_slider", component_property = "value"), 
)
def automatic_cut(n_clicks,filename,json_persistence,alpha,beta): 
    if (json_persistence is None): 
        raise PreventUpdate
    if ctx.triggered_id == "automatic_cut":
        persistence = np.array(json.loads(json_persistence))
        X = ktanh(persistence[:,:-1],alpha,beta)
        return otsu_jump(X,2)
    else: 
        return 0.01 , 0.



@app.callback(
    Output(component_id = "persistent_diagram", component_property = "figure"),
    Input(component_id = "b_cut_slider", component_property = "value"),
    Input(component_id = "p_cut_slider", component_property = "value"),
    Input(component_id = "alpha_slider", component_property = "value"),
    Input(component_id = "beta_slider", component_property = "value"),
    Input(component_id="persistence_store", component_property="data"),
    Input(component_id="upload_data",component_property="filename"),
    
)
def update_birth_cut(b_cut,p_cut,alpha,beta,json_persistence,filename): 
    patch = Patch()
    #update cuts
    patch["layout"]["shapes"][0]["x0"] = b_cut
    patch["layout"]["shapes"][0]["x1"] = b_cut
    patch["layout"]["shapes"][1]["x1"] = 2-p_cut
    patch["layout"]["shapes"][1]["y0"] = p_cut
    if ctx.triggered_id == "upload_data":
        patch["data"][0]["x"] = [] 
        patch["data"][0]["y"] = []
    else:
        if (json_persistence is None)*(p_cut == 0): 
            raise PreventUpdate
        persistence = np.array(json.loads(json_persistence))
        #update data layout distance
        y = ktanh(persistence[:,:-1],alpha,beta)
        patch["data"][0]["x"] = y[:,0] 
        patch["data"][0]["y"] = y[:,1] 
    return patch

@app.callback(
    Output(component_id = "distance_plot", component_property = "figure"),
    Input(component_id = "alpha_slider", component_property = "value"),
    Input(component_id = "beta_slider", component_property = "value"),
)
def update_distance(alpha,beta): 
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
    Input(component_id="upload_data",component_property="filename"),
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
def update_signal_motif_plots(n_clicks,json_signal,filename,b_cut,p_cut,alpha,beta,remove_outlier,outlier_threshold,json_mst,json_mp,json_persistence,wlen): 
    if json_signal is None: 
        raise PreventUpdate
    
    if remove_outlier == " Remove outliers": 
        remove_outlier = True
    else: 
        False
    signal = np.array(json.loads(json_signal))
    patch_signal = display_signal(signal)
    options = ["Motif 1"]
    s_option = None
    json_motif = None  
    if ctx.triggered_id != "upload_data":
        mst = np.array(json.loads(json_mst))
        mp = np.array(json.loads(json_mp))
        persistence = np.array(json.loads(json_persistence))
        pred,birth = fit_on_submit(mst,mp,persistence,wlen,p_cut,b_cut,alpha,beta,remove_outlier,int(outlier_threshold[:-1]))
        patch_signal = update_signal_figure(signal,pred,birth)
        if not pred is None:
            options = [f"Motif {i+1}" for i in range(len(pred))]
            s_option = options[:min(3,len(options))]
            motif_plots = get_motif_plots(signal,pred,wlen,remove_outlier,float(outlier_threshold[:-1])/100.)
            json_motif = json.dumps(motif_plots)
        
        
    return patch_signal,options,s_option,json_motif

@app.callback(
    Output(component_id="motif_plot",component_property="figure"),
    Input(component_id="motif_selection",component_property="value"),
    Input(component_id="upload_data",component_property="filename"),
    State(component_id="motif_plot_store", component_property="data")
)
def update_motif_plot(value,filename,data):
    if ctx.triggered_id != "upload_data":
        if data is None: 
            raise PreventUpdate
    
        if type(value) == str: 
            idxs =[int(value.split(" ")[-1])-1]
        else:
            idxs = [int(val.split(" ")[-1])-1 for val in value]
        plots = json.loads(data)
        tplots = []
        for i,idx in enumerate(idxs[:3]): 
            plot = plots[idx]
            for dct in plot: 
                if i == 0: 
                    dct["xaxis"] = "x"
                    dct["yaxis"] = "y"
                else: 
                    dct["xaxis"] = f"x{i+1}"
                    dct["yaxis"] = f"y{i+1}"
            tplots.append(plot)
        patch = update_motif_figure(tplots)
        return patch
    else:
        patch = Patch()
        patch["data"] = []
        return patch



if __name__ == '__main__':
    app.run_server(debug=True)

