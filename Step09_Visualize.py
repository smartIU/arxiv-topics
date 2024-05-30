import pandas as pd
import plotly.express as px
import webbrowser 

from dash import Dash, html, dcc, ctx, callback, Output, Input
from datetime import datetime
from itertools import cycle
from statsmodels.tsa.seasonal import seasonal_decompose
from threading import Timer

from arxiv_topics.db import DB
from arxiv_topics.config import Config


class arxiv_dash:
    """ plotly / dash app to visualize trends for arXiv topics and categories """
    

    PORT = 8087
    
    URL = 'http://arxiv.org/abs/'
    
    PRESELECTION = 'Please click on a point of a curve to view representative papers.'
    
    COLORS = px.colors.cyclical.IceFire[1:] #px.colors.sequential.ice #px.colors.qualitative.GM10        
    
    FILTER_TOPICS = 20
    
    VIEW_PAPERS = 6
    
    STYLES = {
        'title': {
            'textAlign': 'center',
            'margin-block-end': '0.2em',
            'color': '#486691'
        },
        'selection-container': {
            'border': 'thin #486691 solid',
            'padding': '10px'
        },
        'div': {
            'float':'left',
            'margin-right':'75px'
        },
        'labels': {
            'display': 'inline-block',
            'width': '60px',
            'vertical-align': 'top'
        },
        'choices': {
            'display': 'inline-block'
        },
        'radio': {          
            'display': 'inline-block'
        },
        'papers-container': {
            'border': 'thin #486691 solid',
            'padding': '10px',
            'position': 'relative',
            'margin-top': '-50px'
        }
    }      
    
    
    def __init__(self, name, df_topics, df_approximation, df_categories):
        
        self.df_topics = df_topics
        self.df_approximation = df_approximation
        self.df_categories = df_categories

        self.current_cluster = 'Topics'
        self.current_stat = 'Percentage'
        self.current_filter = 'Top'
        
        self.range_start = '2014-05-01'
        self.range_end = df_topics['publication'].max()
        
        self.current_view = df_approximation
        
        
        self._app = Dash(name)

        self._app.layout = html.Div([
            html.H1(children='Arxiv Topics', style=self.STYLES['title']),
            html.Div([
                html.Div([
                    html.Label('Cluster: ', style=self.STYLES['labels'])
                  , dcc.RadioItems([self._rI('Topics')
                                  , self._rI('Topics (incl. approximate distribution)', 'Approximation')
                                  , self._rI('Categories')], self.current_cluster
                                  , id='cluster-choice', style=self.STYLES['choices'])              
                        ],style=self.STYLES['div']),
                html.Div([
                    html.Label('Trends: ', style=self.STYLES['labels'])
                  , dcc.RadioItems([self._rI('Percentage'), self._rI('Count')], self.current_stat
                                  , id='stats-choice', style=self.STYLES['choices'])
                        ],style=self.STYLES['div']),
                html.Div([
                    html.Label('Filter: ', style=self.STYLES['labels'])
                  , dcc.RadioItems([self._rI(f'Rising {self.FILTER_TOPICS}', 'Top')
                                  , self._rI('All')
                                  , self._rI(f'Falling {self.FILTER_TOPICS}', 'Bottom')], self.current_filter
                                  , id='filter-choice', style=self.STYLES['choices'])
                        ])
                    ], style=self.STYLES['selection-container']),
            dcc.Graph(id='graph-content'),
            html.Div(id='papers-content', style=self.STYLES['papers-container'])
        ])

    def _rI(self, label, value = None):
        if value is None:
            value = label
        
        return { 'label': html.Div([label], style=self.STYLES['radio']), 'value': value }


    def start_server(self):
        """ start local dash server and open in browser """
        Timer(3, webbrowser.open(f'http://localhost:{self.PORT}', new=0)).start()
        self._app.run_server(debug=False, port=self.PORT)


    @callback(
        Output('graph-content', 'figure'),
        Input('cluster-choice', 'value'),
        Input('stats-choice', 'value'),
        Input('filter-choice', 'value'),
        Input('graph-content', 'relayoutData'),
        prevent_initial_call=True)
    def update_graph(clusterChoice, statChoice, filterChoice, relayoutData):
        """ update the graph based on radio button and range selection """
        
        app.current_cluster = clusterChoice
        app.current_stat = statChoice
        app.current_filter = filterChoice

        lines = 'label'
        x = 'publication'        

        if statChoice == 'Count':
            y = 'trend_count' # trend for 'paper_count'
            label_y = 'Number of monthly papers'
            slope = 'slope_count'
        else:
            y = 'trend_percent' # trend for 'total_percent'
            label_y = 'Percentage of monthly papers'
            slope = 'slope_percent'
            
        if clusterChoice == 'Topics':
            legend = 'Topics'
            df = app.df_topics  
        elif clusterChoice == 'Approximation':
            legend = 'Topics'
            df = app.df_approximation 
        else:
            legend = 'Categories'
            df = app.df_categories


        # set the same color when belonging to the same parent topic/archive
        # NOTE: there are far more topics with hierarchy 1 than colors in the px.colors swatches
        #     , so topics (with hierarchy 0) will still have the same color, even without being similar
        colors = cycle(app.COLORS)
        parents = pd.DataFrame(df['parent'].unique()).set_index(0)
        parents['color'] = [next(colors) for i in range(parents.shape[0])]                 
        colors = df.join(parents, on='parent').set_index('label')['color']
        colors = colors[~colors.index.duplicated()].to_dict()

        
        # handle zoom
        if relayoutData is not None:
        
            if 'xaxis.autorange' in relayoutData:
                app.range_start = df['publication'].min()
                app.range_end = df['publication'].max()
            if 'xaxis.range' in relayoutData:
                app.range_start = relayoutData['xaxis.range'][0]
                app.range_end = relayoutData['xaxis.range'][1]
            else:
                if 'xaxis.range[0]' in relayoutData:
                    app.range_start = relayoutData['xaxis.range[0]']
                if 'xaxis.range[1]' in relayoutData:
                    app.range_end = relayoutData['xaxis.range[1]']            
           
        
        # filter
        if filterChoice == 'All':
            app.current_view = df
            
        else:            
            
            dff = df[df['publication'].between(app.range_start, app.range_end)].groupby(by='id')
            
            if filterChoice == 'Top':
                dff = dff.tail(3).groupby(by='id')[slope].max().nlargest(app.FILTER_TOPICS).index.unique()                      
            else:
                dff = dff.tail(3).groupby(by='id')[slope].min().nsmallest(app.FILTER_TOPICS).index.unique()
            
            app.current_view = df[df['id'].isin(dff)]
         
         
        # create figure
        figure = px.line(app.current_view, x=x, y=y, line_shape='spline'
                       , color=lines, color_discrete_map=colors
                       , labels={x:'',y:label_y,lines:legend})
                       
        figure.update_layout(xaxis=dict(rangeslider=dict(visible=True)
                                      , range=[app.range_start, app.range_end]
                                      , type='date'))

        if statChoice == 'Percentage':
            figure.layout.yaxis.tickformat = '.2%'

                
        # stop the graph from reseting when only changing the range / zoom
        figure.layout.uirevision = clusterChoice + statChoice
        
        return figure


    @callback(
        Output('papers-content', 'children'),
        Input('graph-content', 'clickData'),
        Input('cluster-choice', 'value'),
        Input('stats-choice', 'value'),
        Input('filter-choice', 'value'))
    def display_papers(clickData, clusterChoice, statChoice, filterChoice):
        """ display a number of representative papers based on clicked curve """
    
        if ctx.triggered_id != 'graph-content' or clickData is None or 'points' not in clickData or len(clickData['points']) == 0:
            return app.PRESELECTION
            

        point = clickData['points'][0]
        curve = point['curveNumber']
        year = point['x'][:4]
        month = point['x'][5:7]
        
        selected_id = app.current_view['id'].unique()[curve]
        
        if app.current_cluster == 'Categories':
            papers = db.get_representative_papers_for_category(selected_id, year, month, app.VIEW_PAPERS)
        elif app.current_cluster == 'Topics':
            papers = db.get_representative_papers_for_topic(selected_id, year, month, app.VIEW_PAPERS, False)
        else:
            papers = db.get_representative_papers_for_topic(selected_id, year, month, app.VIEW_PAPERS, True)

        label = app.current_view['label'].unique()[curve]


        children = [html.P(f'Selection of papers for "{label}" published {year}-{month}:')]
       
        for paper in papers.itertuples():
            children.append(html.A(paper.title, id=paper.arxiv_id, href=f'{app.URL}{paper.arxiv_id}'))
            children.append(html.Br())

        return children        


def calculate_trends(stats, slope_periods = 6, incl_parent = False):
    """ calculate smooth trend lines from the raw paper statistics
    
      Arguments:
        min_percentage: minimum percentage of monthly papers a topic has to cover, at least once, to be included
        slope_periods: difference in months for calculating the slope (actually just the increase/decrease) of the trend line
        incl_parent: additionally calculate the trend in relation to the parent topic (not implemented yet)
    """
    
    df_trends = pd.DataFrame()

    ids = stats['id'].unique()
    for id in ids:            
        
        df = stats[stats['id'] == id]
        
        # create continuous monthly date range index
        df.index = pd.DatetimeIndex(df['publication'])
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='MS'), fill_value=0)

        # calculate trend lines
        df['trend_count'] = seasonal_decompose(df['paper_count'], model='additive', extrapolate_trend='freq').trend.clip(0).round()
        df['trend_percent'] = seasonal_decompose(df['total_percent'], model='additive', extrapolate_trend='freq').trend.clip(0)
        
        # TODO: implement
        # could be used to show trends within archives/topics in the upper hierarchy
        #if incl_parent:
        #    df['trend_parent'] = seasonal_decompose(df['parent_percent'], model='additive', extrapolate_trend='freq').trend.clip(0)

        # calculate rolling slope of trend
        df['slope_count'] = df['trend_count'].rolling(window=slope_periods).apply(lambda x: x.iloc[-1] - x.iloc[0])
        df['slope_percent'] = df['trend_percent'].rolling(window=slope_periods).apply(lambda x: x.iloc[-1] - x.iloc[0])        

        if df_trends.empty:
            df_trends = df
        else:
            df_trends = pd.concat([df_trends, df])

    return df_trends[df_trends['label'] != 0]


if __name__ == '__main__':    

    config = Config()
    db = DB(config.database)

    print('calculating trends')
    df_topics = calculate_trends(db.get_topic_stats_by_month(incl_approximation=False))    
    df_approximation = calculate_trends(db.get_topic_stats_by_month(incl_approximation=True))    
    df_categories = calculate_trends(db.get_category_stats_by_month())

    print('starting server')
    app = arxiv_dash(__name__, df_topics, df_approximation, df_categories)
    app.start_server()   
