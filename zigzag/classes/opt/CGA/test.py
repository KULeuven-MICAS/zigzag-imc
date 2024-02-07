import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'surface'}, {'type': 'surface'}]])

fig.append_trace( 
     go.Mesh3d(
        # 8 vertices of a cube
        x=[0.608, 0.608, 0.998, 0.998, 0.608, 0.608, 0.998, 0.998],
        y=[0.091, 0.963, 0.963, 0.091, 0.091, 0.963, 0.963, 0.091],
        z=[0.140, 0.140, 0.140, 0.140, 0.571, 0.571, 0.571, 0.571],

        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#DC143C',
        flatshading = True,
        name='a'
    ), row=1,col=1)
fig.append_trace( 
     go.Mesh3d(
        # 8 vertices of a cube
        x=[0.608, 0.608, 0.998, 0.998, 0.608, 0.608, 0.998, 0.998],
        y=[0.091, 0.963, 0.963, 0.091, 0.091, 0.963, 0.963, 0.091],
        z=[0.140, 0.140, 0.140, 0.140, 0.571, 0.571, 0.571, 0.571],

        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#DC143E',
        flatshading = True,
        name='a'
    ), row=1, col=2                   ) 

fig.update_traces(showlegend=True, selector=dict(type='mesh3d'))
fig.show()
