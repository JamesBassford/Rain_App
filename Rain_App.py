import streamlit as st
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import altair as alt
from mpl_toolkits.basemap import Basemap

#wide mode
st.set_page_config(layout="wide")
#write the title
st.title('Rainfall Analysis Dashboard')

#files
hist_p25 = xr.open_dataset('25km/hist/p25/hist_p25_av.nc')
hist_cp4 = xr.open_dataset('25km/hist/cp4/hist_cp4_av.nc')
fc_p25 = xr.open_dataset('25km/rcp85/p25/fc_p25_av.nc')
fc_cp4 = xr.open_dataset('25km/rcp85/cp4/fc_cp4_av.nc')

#convert to mm per day
hist_p25 = hist_p25*86400
hist_cp4 = hist_cp4*86400
fc_p25 = fc_p25*86400
fc_cp4 = fc_cp4*86400

#calculate differences
diff_hist = (hist_cp4 - hist_p25)
diff_fc = (fc_cp4 - fc_p25)
diff_cp4 = fc_cp4 - hist_cp4
diff_p25 = fc_p25 - hist_p25
diff_diff = diff_fc - diff_hist

#monthly averages
hist_cp4_splits = np.array_split(hist_cp4['pr'],12)
fc_cp4_splits = np.array_split(fc_cp4['pr'],12)
hist_p25_splits = np.array_split(hist_p25['pr'],12)
fc_p25_splits = np.array_split(fc_p25['pr'],12)

#making dataframes 
hp25 = hist_p25.to_dataframe().reset_index()
fp25 = fc_p25.to_dataframe().reset_index()
hcp4 = hist_cp4.to_dataframe().reset_index()
fcp4 = fc_cp4.to_dataframe().reset_index()

dh = diff_hist.to_dataframe().reset_index()
dh = diff_fc.to_dataframe().reset_index()
dc = diff_cp4.to_dataframe().reset_index()
dp = diff_p25.to_dataframe().reset_index()

dd = diff_diff.to_dataframe().reset_index()

########################################################################


# Make shortcut to Basemap object, 
# not specifying projection type for this example
m = Basemap() 

# Make trace-generating function (return a Scatter object)
def make_scatter(x,y):
    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.scatter.Line(color="white"),
        name=' '  # no name on hover
    )

# Functions converting coastline/country polygons to lon/lat traces
def polygons_to_traces(poly_paths, N_poly):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    traces = []  # init. plotting list 

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)
        
        # add plot.ly plotting options
        traces.append(make_scatter(lon_cc,lat_cc))
     
    return traces

# Function generating coastline lon/lat traces
def get_coastline_traces():
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces(poly_paths, N_poly)

# Function generating country lon/lat traces
def get_country_traces():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces(poly_paths, N_poly)

# Get list of of coastline and country lon/lat traces
traces_cc = get_coastline_traces()+get_country_traces()


#######################


##make maps
c1 = go.Contour(
    z=hp25['pr'],
    x=hp25['lon'],
    y=hp25['lat'],
    colorscale="jet",
    contours=dict(start=0, end=15, size=1)
)

c2 = go.Contour(
    z=hcp4['pr'],
    x=hcp4['lon'],
    y=hcp4['lat'],
    colorscale="jet",
    contours=dict(start=0, end=15, size=1)
)

c3 = go.Contour(
    z=fp25['pr'],
    x=fp25['lon'],
    y=fp25['lat'],
    colorscale="jet",
    contours=dict(start=0, end=15, size=1)
)

c4 = go.Contour(
    z=fcp4['pr'],
    x=fcp4['lon'],
    y=fcp4['lat'],
    colorscale="jet",
    contours=dict(start=0, end=15, size=1)
)

map_data1 = go.Data([c1]+traces_cc)
map_data2 = go.Data([c2]+traces_cc)
map_data3 = go.Data([c3]+traces_cc)
map_data4 = go.Data([c4]+traces_cc)

layout = go.Layout(        # highlight closest point on hover
    xaxis=go.layout.XAxis(range=[min(hp25['lon']),max(hp25['lon'])]),
    yaxis=go.layout.YAxis(range=[min(hp25['lat']),max(hp25['lat'])]),
    height=700,
    width=600
    ) 
        

map1 = go.Figure(data=map_data1, layout=layout)
map2 = go.Figure(data=map_data2, layout=layout)
map3 = go.Figure(data=map_data3, layout=layout)
map4 = go.Figure(data=map_data4, layout=layout)


labels=dict(x="Day", y="Latitude", color="Average Rainfall (mm/day)")


#create graph objects
plot1 = [go.Heatmap(x= hist_p25['time'], 
                   y = hist_p25['lat'], z=hist_p25['pr'].mean(dim='lon').transpose(), 
                   zsmooth = 'best', colorscale='jet', zmin = 0, zmax = 15)
        ]

plot2 = [go.Heatmap(x= fc_p25['time'], 
                   y = fc_p25['lat'], z=fc_p25['pr'].mean(dim='lon').transpose(), 
                   zsmooth = 'best', colorscale='jet', zmin = 0, zmax = 15)
        ]

plot3 = [go.Heatmap(x= hist_cp4['time'], 
                   y = hist_cp4['lat'], z=hist_cp4['pr'].mean(dim='lon').transpose(), 
                   zsmooth = 'best', colorscale='jet', zmin = 0, zmax = 15)
        ]

plot4 = [go.Heatmap(x= fc_cp4['time'], 
                   y = fc_cp4['lat'], z=fc_cp4['pr'].mean(dim='lon').transpose(), 
                   zsmooth = 'best', colorscale='jet', zmin = 0, zmax = 15)
        ]

#create figures
p25_hist = go.Figure(plot1)
p25_fc = go.Figure(plot2)
cp4_hist = go.Figure(plot3)
cp4_fc = go.Figure(plot4)

#add titles and axis labels
p25_hist.update_layout(title='P25 Historical Rainfall', xaxis_title="Day", yaxis_title="Latitude", width=600)
p25_fc.update_layout(title='P25 Future Rainfall', xaxis_title="Day", yaxis_title="Latitude", width=600)
cp4_hist.update_layout(title='CP4 Historical Rainfall', xaxis_title="Day", yaxis_title="Latitude", width=600)
cp4_fc.update_layout(title='CP4 Future Rainfall', xaxis_title="Day", yaxis_title="Latitude", width=600)

fig_list = [p25_hist, cp4_hist, p25_fc, cp4_fc]
models_list = ['p25_hist', 'cp4_hist', 'p25_fc', 'cp4_fc', 'none']
map_list = [map1,map2,map3,map4]

fig1 = p25_hist
fig2 = p25_fc

line1 = hp25.groupby('time').mean().drop(['lat','lon'],axis=1)
line2 = hcp4.groupby('time').mean().drop(['lat','lon'],axis=1)
line3 = fp25.groupby('time').mean().drop(['lat','lon'],axis=1)
line4 = fcp4.groupby('time').mean().drop(['lat','lon'],axis=1)

pr_lines = pd.DataFrame({'p25_hist': line1['pr'], 'cp4_hist': line2['pr'], 'p25_fc': line3['pr'], 'cp4_fc': line4['pr']})

st.subheader('Zonal and Annual Means')

col1, col2 = st.columns([2,2])       
  
    
with col1:
    
    model1 = st.selectbox('Zonal Average 1',models_list)
    
    for i in range(4):
        if model1 == models_list[i]:
            fig1 = fig_list[i]
        
    if model1 != 'none':
        st.write(fig1, use_column_width=True)

    map_model1 = st.selectbox('Annual Mean map 1',models_list)

    for i in range(4):
        if map_model1 == models_list[i]:
            showmap1 = map_list[i]

    if map_model1 != 'none':
        st.write(showmap1)

with col2:
    
    model2 = st.selectbox('Zonal Average 2',models_list)
    
    for i in range(4):
        if model2 == models_list[i]:
            fig2 = fig_list[i]
    
    if model2 != 'none':
        st.write(fig2, use_column_width=True)

    map_model2 = st.selectbox('Annual Mean map 2',models_list)

    for i in range(4):
        if map_model2 == models_list[i]:
            showmap2 = map_list[i]

    if map_model2 != 'none':
        st.write(showmap2)


st.subheader('Domain Average Time Series')

col3, col4 = st.columns([2,10])  

with col3:

    check1 = st.checkbox('p25_hist')
    check2 = st.checkbox('cp4_hist')
    check3 = st.checkbox('p25_fc')
    check4 = st.checkbox('cp4_fc')

    area = st.radio('Line or Area plot?',('Line','Area'))

checklist = [check1,check2,check3,check4]

with col4:

    linedata = []
    for i in range(4):
        if checklist[i]==True:
          linedata.append(models_list[i])

    if area == 'Line':
        st.line_chart(pr_lines[linedata])
    else:
        st.area_chart(pr_lines[linedata])

        
#making the monthly plots

st.subheader('Monthly Averages')

months = ['jan','feb','mar','apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

m_figs = ['hist_p25_splits.png', 'hist_cp4_splits.png', 'fc_p25_splits.png', 'fc_cp4_splits.png']

model3 = st.selectbox(
    'Shows months for',models_list)
    
for i in range(4):
    if model3 == models_list[i]:
        monthly = m_figs[i]

if model3 != 'none':
    st.image(monthly, use_column_width=True)





#things to do next
    #make the monthly means prettier 
    #start including difference plots
    #histograms bro







