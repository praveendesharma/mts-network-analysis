# project.py


import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def create_detailed_schedule(schedule, stops, trips, bus_lines):

    final_df = pd.merge(schedule, stops, on='stop_id', how='left')
    final_df = pd.merge(final_df, trips, on='trip_id', how='left')
    
    final_df['route_id'] = pd.Categorical(final_df['route_id'], categories=bus_lines, ordered=True)
    
    stop_counts = final_df.groupby(['route_id', 'trip_id']).size().reset_index(name='stop_count')
    
    final_df = pd.merge(final_df, stop_counts, on=['route_id', 'trip_id'], how='left')
    
    # sort in the following order: route_id (for bus lines), stop_count(), trip_id, stop_sequence
    final_df = final_df.sort_values(['route_id', 'stop_count', 'trip_id', 'stop_sequence'])
    
    final_df = final_df.drop(columns=['stop_count'])
    
    final_df = final_df.set_index('trip_id')
    final_df = final_df.dropna()

    return final_df

def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    for bus in bus_df['route_id'].unique():
        bus_id = bus_df[bus_df['route_id'] == bus]
        fig.add_trace(go.Scattermapbox(
            lat = bus_id['stop_lat'],
            lon = bus_id['stop_lon'],
            name = f"Bus Line {bus}",
            hoverinfo = 'text',
            text = bus_id['stop_name']
        ))
    
    return fig


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
    detailed_schedule=detailed_schedule.reset_index()
    # get the trip_ids of the trips going through the station
    trip_arr=detailed_schedule[detailed_schedule['stop_name']==station_name].\
        reset_index().trip_id.tolist()
    
    # create an array of the trips going through the station
    trip_arr=detailed_schedule[detailed_schedule['stop_name']==\
                               station_name].reset_index().trip_id.tolist()
    # create an array of the stop sequence for each corresponding trip
    stop_seq=detailed_schedule[detailed_schedule['stop_name']==\
                               station_name].stop_sequence.tolist()

    # next station lst
    next_station=[]
    for i in range(len(trip_arr)):
        # create a df of the individual trip to find the next stop
        x=detailed_schedule[(detailed_schedule['trip_id']==trip_arr[i])]

        # check if there is a next stop i.e the current stop is not the last stop
        if(x.shape[0]>stop_seq[i]):
            next_station.append(x.iloc[stop_seq[i]].stop_name)

    return np.array(next_station)
    
def find_tripid(station_name,detailed_schedule):
    detailed_schedule=detailed_schedule.reset_index()
    # get the trip_ids of the trips going through the station
    trip_arr=detailed_schedule[detailed_schedule['stop_name']==station_name].\
        reset_index().trip_id.tolist()
    
    # create an array of the trips going through the station
    trip_arr=detailed_schedule[detailed_schedule['stop_name']==\
                               station_name].reset_index().trip_id.tolist()
    # create an array of the stop sequence for each corresponding trip
    stop_seq=detailed_schedule[detailed_schedule['stop_name']==\
                               station_name].stop_sequence.tolist()

    # next station lst
    next_station=[]
    for i in range(len(trip_arr)):
        # create a df of the individual trip to find the next stop
        x=detailed_schedule[(detailed_schedule['trip_id']==trip_arr[i])]

        # check if there is a next stop i.e the current stop is not the last stop
        if(x.shape[0]>stop_seq[i]):
            next_station.append(x.iloc[stop_seq[i]].trip_id)

    return np.array(next_station)





    


def bfs(start_station, end_station, detailed_schedule):

    # ensure start and end stations are in the df
    if start_station not in detailed_schedule['stop_name'].unique():
        return f"Start station {start_station} not found."
    if end_station not in detailed_schedule['stop_name'].unique():
        return f"End station '{end_station}' not found."
    
    # create a double-ended queue for the bfs algo and a visited set to keep 
    # track of unique stations that have been visited.
    queue = deque([(start_station, [start_station])])
    visited = set([start_station])
    

    while queue:


        curr_station, shortest_route = queue.popleft()
        
        # If final destination is reached
        if curr_station == end_station:
            # Create df as per specifications
            shortest_route_df = []
            for i, station in enumerate(shortest_route, 1):
                station_info = detailed_schedule[detailed_schedule['stop_name'] == station].iloc[0]
                shortest_route_df.append({
                    'stop_name': station,
                    'stop_lat': station_info['stop_lat'],
                    'stop_lon': station_info['stop_lon'],
                    'stop_num': i
                })
            return pd.DataFrame(shortest_route_df)
        
        # Get neighbors of the current station
        neighbors = find_neighbors(curr_station, detailed_schedule)
        
        # Add the neighbors that are not in visited to the queue
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, shortest_route + [neighbor]))
    
    return "No path found"






# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    # final array that will keep track of all the timings 
    timings = []

    # total number of busses
    number_of_buses = (1440-360)//tau

    # generating a random minutes with lower range 360 and upper range 1440
    arrival_times_sorted = np.sort(np.random.uniform(360,1440,number_of_buses))

    # differences between consecutive arrival timings
    interval = np.diff(arrival_times_sorted)
    # adding the difference between the first bus arrival and 6am
    interval = np.insert(interval,0,arrival_times_sorted[0]-360)

    # loop through all the arrival_times_sorted
    for i in range(len(arrival_times_sorted)):
        # the random time that was generated
        time = arrival_times_sorted[i]
        # convert mins to hr:min:sec
        hours = int(time//60)
        mins = int(time%60)
        secs = int(60*(time-(hours*60+mins)))
        # append everything to the timings array that will be made into a df.
        timings.append({
            "Arrival Time": f"{hours:02d}:{mins:02d}:{secs:02d}",
            "Interval" : interval[i]
        })
    return pd.DataFrame(timings)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def simulate_wait_times(arrival_times_df, n_passengers):

    # list for bus timings in mins
    bus_timings_mins = []

    # bus timings arr in hr:min:secs
    bus_times_init = arrival_times_df['Arrival Time'].tolist()

    # loop through each time and convert it to mins
    for time in bus_times_init:
        time_mins = time.split(':')
        time_mins[0] = int(time_mins[0])*60
        time_mins[1] = int(time_mins[1])
        time_mins[-1] = int(time_mins[-1])/60
        bus_timings_mins.append(sum(time_mins))

    # generating a random minutes with lower range 360 and upper range that is 
    # the timing of the last bus.
    pax_arrival_times_sorted = np.sort(np.random.uniform(360,max(bus_timings_mins),n_passengers))

    # bus index column
    bus_index_col = []

    # wait time col
    wait_time = []

    # pax arrival time col
    pax_arrival = []

    # bus arrival times
    bus_arrival_time = []

    
    # track passenger index
    curr_pax_idx = 0  
    
    # loop through each passenger
    while curr_pax_idx < len(pax_arrival_times_sorted):
        pax_time = pax_arrival_times_sorted[curr_pax_idx]
        
        # find the next available bus for this pax
        for i in range(len(bus_timings_mins)):
            bus_timings_mins=np.sort(bus_timings_mins)
            if bus_timings_mins[i] >= pax_time:
                # find the first bus that arrives after this pax
                bus_index_col.append(i)
                wait_time.append(bus_timings_mins[i] - pax_time)
                curr_pax_idx += 1
                break
    
    # loop through pax timings
    for i in range(len(pax_arrival_times_sorted)):
        # the random time that was generated
        time = pax_arrival_times_sorted[i]
        # convert mins to hr:min:sec
        hours = int(time//60)
        mins = int(time%60)
        secs = int(60*(time-(hours*60+mins)))
        # append everything to the pax_arrival array that will be made into a df.
        pax_arrival.append(f"{hours:02d}:{mins:02d}:{secs:02d}")
    
    # loop through the indices and add relevant bus timings to arrival time
    for idx in bus_index_col:
        bus_arrival_time.append(arrival_times_df['Arrival Time'].iloc[idx])

    # create final df and assign the cols with the corresponding arrays
    final_df = pd.DataFrame(columns=['Passenger Arrival Time','Bus Arrival Time','Bus Index','Wait Time'])
    final_df['Passenger Arrival Time'] = pax_arrival
    final_df['Bus Arrival Time'] = bus_arrival_time
    final_df['Bus Index'] = bus_index_col
    final_df['Wait Time'] = wait_time
    
    return final_df


def visualize_wait_times(wait_times_df, timestamp):
    # Convert string to datetime
    wait_times_df['Passenger Arrival Time'] = pd.to_datetime(wait_times_df['Passenger Arrival Time'], format='%H:%M:%S').dt.time
    wait_times_df['Bus Arrival Time'] = pd.to_datetime(wait_times_df['Bus Arrival Time'], format='%H:%M:%S').dt.time
    
    # Convert timestamp to time so that it can be compared to the passenger timings
    timestamp_time = timestamp.time()
    end_time = (timestamp + pd.Timedelta(hours=1)).time()
    print(end_time)
    # query the df for that one hr block for passenger
    pax_times_df = wait_times_df[
        (wait_times_df['Passenger Arrival Time'] >= timestamp_time) & 
        (wait_times_df['Passenger Arrival Time'] <= end_time)
    ]
    # query the df for that one hr block for buses
    bus_times_df = wait_times_df[
        (wait_times_df['Bus Arrival Time'] >= timestamp_time) & 
        (wait_times_df['Bus Arrival Time'] <= end_time)
    ]

    # make the figure
    fig=go.Figure()

    def time_to_minutes(t, start_hour):
        return (t.hour - start_hour) * 60 + t.minute + t.second/60
    
    bus_timings = bus_times_df['Bus Arrival Time'].unique()
    print(bus_timings)
    bus_minutes = [time_to_minutes(t, timestamp.hour) for t in bus_timings]


    fig.add_trace(go.Scatter(
        x=bus_minutes,
        y=[0]* len(bus_timings),
        mode='markers',
        name='Buses',
        marker=dict(color='blue', size=10)
    ))

    x_pax_time = pax_times_df['Passenger Arrival Time'].apply(lambda t: time_to_minutes(t, timestamp.hour))
    y_pax_time = pax_times_df['Wait Time']
    
    fig.add_trace(go.Scatter(
        x=x_pax_time,
        y=y_pax_time,
        mode='markers',
        name='Passengers',
        marker=dict(color='red', size=10)
    ))
    
    for xi, yi in zip(x_pax_time, y_pax_time):
        fig.add_trace(go.Scatter(
                x=[xi, xi],
                y=[0, yi],
                mode='lines',
                line=dict(color='red', width=1, dash='dot'),
                showlegend=False
            ))

    return fig
