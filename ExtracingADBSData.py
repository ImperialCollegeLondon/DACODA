import numpy as np
from traffic.data import opensky
import pandas as pd
from haversine import inverse_haversine, Direction, haversine
from traffic.data import eurofirs, airports
from matplotlib import pyplot as plt
import matplotlib as mpl
from cartes.crs import OSGB
import cartopy.crs as ccrs
from traffic.drawing import TransverseMercator, countries, lakes, ocean, rivers
import pyproj
import airportsdata
import utm
import time
import math
import random
import os

mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['legend.title_fontsize'] = 16
mpl.rcParams['font.family'] = 'serif'


# Function to calculate the change in degree for lon and lat for a given km.
def boundary(coords, distance):
    west = inverse_haversine(coords, distance, Direction.WEST)
    east = inverse_haversine(coords, distance, Direction.EAST)
    north = inverse_haversine(coords, distance, Direction.NORTH)
    south = inverse_haversine(coords, distance, Direction.SOUTH)
    return [west[1], south[0], east[1], north[0]]

def traffic_data_for_model_and_airport(time_start, time_stop, model, airport):
    # Reads all model aircraft in database.
    model_database = pd.read_csv('List_of_aircraft_csv/' + model + '_Database.csv')
    
    airport_icao = airportsdata.load('IATA')[airport]['icao']
    lat = airportsdata.load('IATA')[airport]['lat']
    lon = airportsdata.load('IATA')[airport]['lon']
    
    airport_coords = [lat, lon]

    # Airport Coords - Centre of airports Heathrow - LHR Gatwick - LGW


    # Creating a boundary box with radius/width
    radius = 10
    bounds = boundary(airport_coords, radius)
    # bounds_LHR = [west, south, east, north]

    # Obtains transponder data for departure from LHR or LGW.
    flight = opensky.history(
        time_start,
        stop=time_stop,
        airport=airport_icao,
        bounds=bounds,
    )
    
    # Filters result to only obtain a geo-altitude less than 5000 ft.
    flights_below_5000 = flight.data.where(flight.data['geoaltitude'] <= 5000)
    flights_below_5000.dropna(
        axis=0,
        how='any',
        inplace=True
    )

    # Sorts DataFrame into callsigns and in order of timestamp.
    flights_below_5000.sort_values(['callsign', 'timestamp'], inplace=True)

    # Checks if aircraft is the model and filters out the results.
    list_of_model = flights_below_5000.isin(model_database.icao24.tolist())
    list_of_model_any = list_of_model.any(axis=1)
    list_of_model_filtered = flights_below_5000.loc[list_of_model_any]
    
    if list_of_model_filtered.empty:
        print("There were no " + model + " flights at " + airport)
        exit()

    # Splitting data into deperatures and arrivals
    list_of_model_filtered_depart = list_of_model_filtered.where(list_of_model_filtered['origin'] == airport_icao)
    list_of_model_filtered_depart = list_of_model_filtered_depart.where(list_of_model_filtered_depart['destination'] != airport_icao)
    list_of_model_filtered_depart.dropna(
        axis=0,
        how='any',
        inplace=True
    )

    list_of_model_filtered_arrival = list_of_model_filtered.where(list_of_model_filtered['destination'] == airport_icao)
    list_of_model_filtered_arrival.dropna(
        axis=0,
        how='any',
        inplace=True
    )

    # Creates a CSV of model flights departing and arrival from LHR or LGW in a given time frame.
    list_of_model_filtered.to_csv(airport + '/' + model + '_' + airport + '/' + model + ' ' + airport + ' time - ' +
                                  time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    # Create CSV for DEPARTURES ONLY
    list_of_model_filtered_depart.to_csv(
        airport + '/' + model + '_' + airport + '/' + 'Departure/DEPART_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    # Create CSV for ARRIVALS ONLY
    list_of_model_filtered_arrival.to_csv(
        airport + '/' + model + '_' + airport + '/' + 'Arrival/ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    print('There were a total of ' + str(len(set(list_of_model_filtered.callsign))) + ' ' + model +
          ' aircraft that departed or arrived at ' + airport + ' on ' + time_start[:10])

def aircraft_model_database(model, manuf_capital, manuf_lower):
    aircraft_database = pd.read_csv('List_of_aircraft_csv/Aircraft_Database.csv')
    mask = aircraft_database['model'].notnull()
    model_only_database = aircraft_database[mask & aircraft_database['model'].str.startswith(model)]
    model_only_database = pd.concat(
        [model_only_database,
         aircraft_database[mask & aircraft_database['model'].str.startswith(manuf_capital + ' ' + model)]], axis=0)
    model_only_database = pd.concat(
        [model_only_database,
         aircraft_database[mask & aircraft_database['model'].str.startswith(manuf_lower + ' ' + model)]],
        axis=0)
    # model_only_database = aircraft_database.drop(aircraft_database[aircraft_database.typecode != model].index)
    model_only_database.to_csv('List_of_aircraft_csv/' + model + '_Database.csv')
    return model_only_database

def convert_to_SI(time_start, time_stop, model, airport):
    # Loading in both DEPARTURE AND ARRIVAL data ready to be converted
    depart_data = pd.read_csv(
        airport + '/' + model + '_' + airport + '/' + 'Departure/DEPART_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    arrive_data = pd.read_csv(
        airport + '/' + model + '_' + airport + '/' + 'Arrival/ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    # p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')

    # long_lat_depart = p(depart_data['longitude'], depart_data['latitude'])
    # depart_data['longitude'] = long_lat_depart[0]
    # depart_data['latitude'] = long_lat_depart[1]
    depart_data['altitude'] = depart_data['altitude'] * 0.3048
    depart_data['geoaltitude'] = depart_data['geoaltitude'] * 0.3048
    depart_data['vertical_rate'] = depart_data['vertical_rate'] * 0.00508
    depart_data['timestamp'] = depart_data['timestamp'].map(lambda x: int(time.mktime(pd.to_datetime(x).timetuple())))
    depart_data['groundspeed'] = depart_data['groundspeed'] * 0.514444

    depart_data.to_csv(
        airport + '/' + model + '_' + airport + '/' + 'Departure/UTM_DEPART_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    # long_lat_arrive = p(arrive_data['longitude'], arrive_data['latitude'])
    # arrive_data['longitude'] = long_lat_arrive[0]
    # arrive_data['latitude'] = long_lat_arrive[1]
    arrive_data['altitude'] = arrive_data['altitude'] * 0.3048
    arrive_data['geoaltitude'] = arrive_data['geoaltitude'] * 0.3048
    arrive_data['vertical_rate'] = arrive_data['vertical_rate'] * 0.00508
    arrive_data['timestamp'] = arrive_data['timestamp'].map(lambda x: int(time.mktime(pd.to_datetime(x).timetuple())))
    arrive_data['groundspeed'] = arrive_data['groundspeed'] * 0.514444

    arrive_data.to_csv(
        airport + '/' + model + '_' + airport + '/' + 'Arrival/UTM_ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

def additional_long_lat_depart(time_start, time_stop, model, airport):
    UTM_depart = pd.read_csv(
        airport + '/' + model + '_' + airport + '/' + 'Departure/UTM_DEPART_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')


    # Working out the indexes for a change in aircraft for the DEPART data
    diff_aircraft_depart = UTM_depart['callsign'].ne(UTM_depart['callsign'].shift()).astype(int)
    indices_depart = diff_aircraft_depart[diff_aircraft_depart == 1].index
    indices_depart = indices_depart.tolist()
    indices_depart.append(len(UTM_depart['callsign']))

    # Making a fresh Pandas Dataframe with only the columns
    final_data_frame_depart = pd.DataFrame(columns=['alert', 'altitude', 'callsign', 'day', 'destination', 'firstseen', 'geoaltitude', 'groundspeed', 'hour', 'icao24', 'last_position', 'lastseen', 'latitude', 'longitude', 'onground', 'origin', 'spi', 'squawk', 'timestamp', 'track', 'vertical_rate','runway','drone_target_runway'])
    final_data_frame_depart['alert'] = final_data_frame_depart['alert'].astype(bool)
    final_data_frame_depart['onground'] = final_data_frame_depart['onground'].astype(bool)
    final_data_frame_depart['spi'] = final_data_frame_depart['spi'].astype(bool)

    mu = 0.05
    thrust = 120000  # For one engine - N
    weight = 9.81 * 73900  # Weight of average A320
    rho = 1.2  # Assume this as a density
    V_r = 72.0222  # Rotate speed in m/s
    S_ref = 122.6  # Wing Area
    Vs = V_r / 1.1
    V_tr = 1.15 * Vs

    Cl = weight / (0.5 * rho * V_r ** 2 * S_ref)
    Cd0 = 0.078  # From GROUND_ROLL_DIST article
    k = 0.0334  # From GROUND_ROLL_DIST article

    K_t = 2 * thrust / weight - mu
    K_a = ((rho * S_ref) / (2 * weight)) * (mu * Cl - Cd0 - k * Cl ** 2)

    t_rotate = 2      # Time required to rotate
    t_transition = 2  # Time required to transition into a climb

    pull_up_radius = V_tr ** 2 / (9.81 * (1.2 - 1))  # Radius of circular motion upon pull up
    FPA = 10  # Just for calcs

    ground_roll_dist = (1 / (2 * 9.81 * K_a)) * math.log((K_t + K_a * V_r ** 2) / K_t)
    rotate_dist = t_rotate * V_r
    transition_dist = pull_up_radius * math.sin(FPA * math.pi / 180)

    incorrect_flights = 0
    total_flights = 0
    


    for i in range(0, len(indices_depart)-1):
        total_flights += 1
        one_set_of_aircraft = UTM_depart[indices_depart[i]:indices_depart[i+1]]

        h_init = one_set_of_aircraft['geoaltitude'].iloc[0]
        V_init = one_set_of_aircraft['groundspeed'].iloc[0]
        V_vertical_init = one_set_of_aircraft['vertical_rate'].iloc[0]
        if V_vertical_init < 17.5:
                V_vertical_init = 17.5
        
        gamma_init = np.arcsin(V_vertical_init / V_init)
        long_init = one_set_of_aircraft['longitude'].iloc[0]
        lat_init = one_set_of_aircraft['latitude'].iloc[0]
        t_init = one_set_of_aircraft['timestamp'].iloc[0]
        onground_init = one_set_of_aircraft['onground'].iloc[0]
        track_init = one_set_of_aircraft['track'].iloc[0]
        
        # Identifying the runway and the bearing of the runway
        runway = runway_area_identifier(airport, track_init, lat_init, long_init)
        airport_icao = airportsdata.load('IATA')[airport]['icao']
        airport_df = pd.DataFrame(airports[airport_icao].runways[:])
        runway_bearings = airport_df['bearing']
        runway_names = airport_df['name']
        
        if runway == None:
            incorrect_flights += 1
            continue
        
        for i in range(0,len(runway_names)):
            if runway == runway_names[i]:
                bearing = runway_bearings[i] * np.pi/180


        # Initializing arrays
        geoaltitude = [h_init]
        V_aircraft = [V_init]
        V_vertical = [V_vertical_init]
        gamma = [gamma_init]
        long = [long_init]
        lat = [lat_init]
        t = [t_init]
        onground = [onground_init]
        #bearing = track_init * np.pi/180

        h_tr = pull_up_radius * (1 - math.cos(gamma[0]))
        
        

        ################################# CLIMB #######################################
        # ASSUMPTIONS:
        # - Constant gamma through climb
        # - Constant velocity with no acc
        while min(geoaltitude) >= h_tr:
            
            V_vertical.append(V_vertical[-1])
            t.append(t[-1]-1)  # Goes back in time by one second each step
            gamma.append(gamma[-1])
            V_aircraft.append(V_aircraft[-1])
            geoaltitude.append(geoaltitude[-1] - V_vertical[-1])
            V_horizontal = math.cos(gamma[-1]) * V_aircraft[-1]
            long_and_lat = inverse_haversine([lat[-1],long[-1]], V_horizontal/1000, bearing + np.pi)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(False)

        ############################# TRANSITION #####################################
        # ASSUMPTIONS:
        # - Circular motion during the transition phase
        # - Time for transition roughly 5 seconds
        # - Constant rate of change of gamma to 0 over time
        gamma_array = np.linspace(gamma[-1], 0, t_transition+1)

        for i in range(1, len(gamma_array)):
            t.append(t[-1]-1)
            geoaltitude.append(pull_up_radius * (1 - math.cos(gamma_array[i])))
            circular_motion_acc = V_aircraft[-1] ** 2 / pull_up_radius
            V_aircraft.append(V_aircraft[-1] - circular_motion_acc)
            if V_aircraft[-1] < V_r:
                V_aircraft[-1] = V_r
            V_vertical.append(V_aircraft[-1] * math.sin(gamma_array[i]))
            V_horizontal = math.cos(gamma_array[i]) * V_aircraft[-1]
            long_and_lat = inverse_haversine([lat[-1], long[-1]], V_horizontal / 1000, bearing + np.pi)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(False)
            if geoaltitude[-1] == 0:
                onground[-1] = True

        ############################### ROTATE ######################################
        # ASSUMPTIONS:
        # - Holds velocity constant during the rotate phase

        for i in range(0,t_rotate):
            t.append(t[-1]-1)
            geoaltitude.append(0)
            V_aircraft.append(V_aircraft[-1])
            V_vertical.append(0)
            V_horizontal = V_aircraft[-1]
            long_and_lat = inverse_haversine([lat[-1], long[-1]], V_horizontal / 1000, bearing + np.pi)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(True)

        ############################ GROUND ROLL ###################################
        # Assumptions:
        # - Obtained values for Cd0 and Cl from reference papers
        # - Obtained equations from first year intro to aero notes
        while min(V_aircraft) > 0:
            t.append(t[-1]-1)
            geoaltitude.append(0)
            ground_acc = 9.81 * (K_t + K_a * V_aircraft[-1] ** 2)
            V_aircraft.append(V_aircraft[-1] - ground_acc)
            if V_aircraft[-1] < 0:
                V_aircraft[-1] = 0
            V_vertical.append(0)
            dist = (1 / (2 * 9.81 * K_a)) * math.log((K_t + K_a * V_aircraft[-2] ** 2) / (K_t + K_a * V_aircraft[-1] ** 2))
            long_and_lat = inverse_haversine([lat[-1], long[-1]], dist / 1000, bearing + np.pi)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(True)

        geoaltitude.pop(0)
        V_aircraft.pop(0)
        V_vertical.pop(0)
        long.pop(0)
        lat.pop(0)
        t.pop(0)
        onground.pop(0)

        geoaltitude.reverse()
        V_aircraft.reverse()
        V_vertical.reverse()
        long.reverse()
        lat.reverse()
        t.reverse()
        onground.reverse()
        altitude = [x+(one_set_of_aircraft['altitude'].iloc[0] - one_set_of_aircraft['geoaltitude'].iloc[0]) for x in geoaltitude]  # Heathrow at an elevation of 25m
        callsign = [one_set_of_aircraft['callsign'].iloc[0]] * len(V_aircraft)
        day = [one_set_of_aircraft['day'].iloc[0]] * len(V_aircraft)
        destination = [one_set_of_aircraft['destination'].iloc[0]] * len(V_aircraft)
        firstseen = [one_set_of_aircraft['firstseen'].iloc[0]] * len(V_aircraft)
        hour = [one_set_of_aircraft['hour'].iloc[0]] * len(V_aircraft)
        icao24 = [one_set_of_aircraft['icao24'].iloc[0]] * len(V_aircraft)
        last_pos = [one_set_of_aircraft['last_position'].iloc[0]] * len(V_aircraft)
        lastseen = [one_set_of_aircraft['lastseen'].iloc[0]] * len(V_aircraft)
        origin = [one_set_of_aircraft['origin'].iloc[0]] * len(V_aircraft)
        spi = [one_set_of_aircraft['spi'].iloc[0]] * len(V_aircraft)
        squawk = [one_set_of_aircraft['squawk'].iloc[0]] * len(V_aircraft)
        alert = [one_set_of_aircraft['alert'].iloc[0]] * len(V_aircraft)
        track = [bearing * 180/np.pi] * len(V_aircraft)
        
        # Identifies if the aircraft has overshot the runway through the transformation change
        cuttoff = runway_cuttoff(airport, track[0], lat[0], long[0], 0)
        
        if cuttoff:
            incorrect_flights += 1
            continue
            
        droneTargetRunway = DroneTargetRunwayIdentifier(airport, track_init, lat_init, long_init, 0) # 0 - departure
        
        take_off = pd.DataFrame({'alert':alert, 'altitude':altitude, 'callsign':callsign, 'day':day, 'destination':destination, 'firstseen':firstseen, 'geoaltitude':geoaltitude, 'groundspeed':V_aircraft, 'hour':hour, 'icao24':icao24, 'last_position':last_pos, 'lastseen':lastseen, 'latitude':lat, 'longitude':long, 'onground':onground, 'origin':origin, 'spi':spi, 'squawk':squawk, 'timestamp':t, 'track':track, 'vertical_rate':V_vertical})
        one_set_of_aircraft = pd.concat([take_off, one_set_of_aircraft], ignore_index=True)
        one_set_of_aircraft['runway'] = runway
        one_set_of_aircraft['drone_target_runway'] = droneTargetRunway
        final_data_frame_depart = pd.concat([final_data_frame_depart, one_set_of_aircraft], ignore_index=True)

    airport_lon = airportsdata.load('IATA')[airport]['lon']
    airport_lat = airportsdata.load('IATA')[airport]['lat']
    utm_zone = utm.from_latlon(airport_lat,airport_lon)[2]
    
    p = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    long_lat_depart = p(final_data_frame_depart['longitude'], final_data_frame_depart['latitude'])
    final_data_frame_depart['longitude'] = long_lat_depart[0]
    final_data_frame_depart['latitude'] = long_lat_depart[1]

    final_data_frame_depart = final_data_frame_depart.iloc[:, :-1]
    final_data_frame_depart = final_data_frame_depart.iloc[:, :-1]

    final_data_frame_depart.to_csv(
        airport + '/' + model + '_' + airport + '/' + 'Departure/TAKEOFF_UTM_DEPART_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')
    
    print("There are a total of " + str(total_flights) + " departing flights at " + airport)
    print("There are a total of " + str(incorrect_flights) + " incorrect departing flights transformed at " + airport)
    
    return total_flights, incorrect_flights

def additional_long_lat_arrive(time_start, time_stop, model, airport):
    UTM_arrive = pd.read_csv(
        airport + '/' + model + '_' + airport + '/' + 'Arrival/UTM_ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    # Working out the indexes for a change in aircraft for the arrive data
    diff_aircraft_arrive = UTM_arrive['callsign'].ne(UTM_arrive['callsign'].shift()).astype(int)
    indices_arrive = diff_aircraft_arrive[diff_aircraft_arrive == 1].index
    indices_arrive = indices_arrive.tolist()
    indices_arrive.append(len(UTM_arrive['callsign']))

    # Making a fresh Pandas Dataframe with only the columns
    final_data_frame_arrive = pd.DataFrame(
        columns=['alert', 'altitude', 'callsign', 'day', 'destination', 'firstseen', 'geoaltitude', 'groundspeed',
                 'hour', 'icao24', 'last_position', 'lastseen', 'latitude', 'longitude', 'onground', 'origin', 'spi',
                 'squawk', 'timestamp', 'track', 'vertical_rate','runway','drone_target_runway'])
    final_data_frame_arrive['alert'] = final_data_frame_arrive['alert'].astype(bool)
    final_data_frame_arrive['onground'] = final_data_frame_arrive['onground'].astype(bool)
    final_data_frame_arrive['spi'] = final_data_frame_arrive['spi'].astype(bool)

    mu = 0.5
    thrust = 120000  # For one engine - N
    reverse_thrust = -0.45 * thrust  # Reverse thrust for one engine - N
    weight = 9.81 * 64500  # Weight of average A320 on landing
    rho = 1.2  # Assume this as a density
    V_r = 72.0222  # Rotate speed in m/s
    S_ref = 122.6  # Wing Area
    Vs = V_r / 1.1
    V_flare = 1.23 * Vs
    V_td = 1.15 * Vs

    Cl = weight / (0.5 * rho * V_r ** 2 * S_ref)
    Cd0 = 0.120  # From GROUND_ROLL_DIST article - LANDING
    k = 0.0334  # From GROUND_ROLL_DIST article

    K_t = 2 * reverse_thrust / weight - mu
    K_a = ((rho * S_ref) / (2 * weight)) * (mu * Cl - Cd0 - k * Cl ** 2)

    t_freeroll = 2  # Time the aircraft rolls freely without applying the breaks
    t_flare = 3  # Duration of flare of aircraft

    V_cuttoff = 55 * 0.514444 # 55 knots is the cuttoff 

    pull_up_radius = V_flare ** 2 / (9.81 * (1.2 - 1))  # Radius of circular motion upon pull up
    
    incorrect_flights = 0
    total_flights = 0

    for i in range(0, len(indices_arrive)-1):
        total_flights += 1
        one_set_of_aircraft = UTM_arrive[indices_arrive[i]:indices_arrive[i + 1]]

        h_init = one_set_of_aircraft['geoaltitude'].iloc[-1]
        V_init = one_set_of_aircraft['groundspeed'].iloc[-1]
        V_vertical_init = one_set_of_aircraft['vertical_rate'].iloc[-1]
        if V_vertical_init > -6.5:
                V_vertical_init = -6.5
        gamma_init = np.arcsin((-1*V_vertical_init) / V_init)
        long_init = one_set_of_aircraft['longitude'].iloc[-1]
        lat_init = one_set_of_aircraft['latitude'].iloc[-1]
        t_init = one_set_of_aircraft['timestamp'].iloc[-1]
        onground_init = one_set_of_aircraft['onground'].iloc[-1]
        track_init = one_set_of_aircraft['track'].iloc[-1]
        
        # Identifying the runway and the bearing of the runway
        runway = runway_area_identifier(airport, track_init, lat_init, long_init)
        airport_icao = airportsdata.load('IATA')[airport]['icao']
        airport_df = pd.DataFrame(airports[airport_icao].runways[:])
        runway_bearings = airport_df['bearing']
        runway_names = airport_df['name']
        
        if runway == None:
            incorrect_flights += 1
            continue
        
        for i in range(0,len(runway_names)):
            if runway == runway_names[i]:
                bearing = runway_bearings[i] * np.pi/180

        # Initializing arrays
        geoaltitude = [h_init]
        V_aircraft = [V_init]
        V_vertical = [V_vertical_init]
        gamma = [gamma_init]
        long = [long_init]
        lat = [lat_init]
        t = [t_init]
        onground = [onground_init]
        #bearing = track_init * np.pi / 180

        h_flare = pull_up_radius * (1 - math.cos(gamma[0]))


        #################################### APPROACH ############################################
        while min(geoaltitude) >= h_flare:
            t.append(t[-1] + 1)  # Goes back in time by one second each step
            gamma.append(gamma[-1])
            V_vertical.append(V_vertical[-1])
            V_aircraft.append(V_aircraft[-1])
            geoaltitude.append(geoaltitude[-1] + V_vertical[-1])
            V_horizontal = math.cos(gamma[-1]) * V_aircraft[-1]
            long_and_lat = inverse_haversine([lat[-1], long[-1]], V_horizontal / 1000, bearing)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(False)

        ############################# FLARE #####################################
        # ASSUMPTIONS:
        # - Circular motion during the transition phase
        # - Time for Flare roughly 5 seconds
        # - Constant rate of change of gamma to 0 over time
        gamma_array = np.linspace(gamma[-1], 0, t_flare + 1)

        for i in range(1, len(gamma_array)):
            t.append(t[-1] + 1)
            geoaltitude.append(pull_up_radius * (1 - math.cos(gamma_array[i])))
            circular_motion_acc = V_aircraft[-1] ** 2 / pull_up_radius
            V_aircraft.append(V_aircraft[-1] - circular_motion_acc)
            if V_aircraft[-1] > V_flare:
                V_aircraft[-1] = V_flare
            V_vertical.append(-1 * V_aircraft[-1] * math.sin(gamma_array[i]))
            V_horizontal = math.cos(gamma_array[i]) * V_aircraft[-1]
            long_and_lat = inverse_haversine([lat[-1], long[-1]], V_horizontal / 1000, bearing)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(False)
            if geoaltitude[-1] == 0:
                onground[-1] = True

        ############################### TOUCHDOWN FREE ROLL ######################################
        # ASSUMPTIONS:
        # - Holds velocity constant during the rotate phase

        for i in range(0, t_freeroll):
            t.append(t[-1] + 1)
            geoaltitude.append(0)
            V_aircraft.append(V_aircraft[-1])
            V_vertical.append(0)
            V_horizontal = V_aircraft[-1]
            long_and_lat = inverse_haversine([lat[-1], long[-1]], V_horizontal / 1000, bearing)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(True)

        ############################ GROUND ROLL ###################################
        # Assumptions:
        # - Obtained values for Cd0 and Cl from reference papers
        # - Obtained equations from first year intro to aero notes
        # - RAYMER FOR MU and REVERSE THRUST
        while min(V_aircraft) > V_cuttoff:
            t.append(t[-1] + 1)
            geoaltitude.append(0)
            ground_acc = 9.81 * (K_t + K_a * V_aircraft[-1] ** 2)
            V_aircraft.append(V_aircraft[-1] + ground_acc)
            V_vertical.append(0)
            dist = (1 / (2 * 9.81 * K_a)) * math.log(
                (K_t + K_a * V_aircraft[-1] ** 2) / (K_t + K_a * V_aircraft[-2] ** 2))
            long_and_lat = inverse_haversine([lat[-1], long[-1]], dist / 1000, bearing)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(True)

        K_t = -mu  # NO MORE REVERSE THRUST DUE TO A RESTRICTION ON USAGE ONLY BRAKES

        while min(V_aircraft) > 0:
            t.append(t[-1] + 1)
            geoaltitude.append(0)
            ground_acc = 9.81 * (K_t + K_a * V_aircraft[-1] ** 2)
            V_aircraft.append(V_aircraft[-1] + ground_acc)
            if V_aircraft[-1] < 0:
                V_aircraft[-1] = 0
            V_vertical.append(0)
            dist = (1 / (2 * 9.81 * K_a)) * math.log(
                (K_t + K_a * V_aircraft[-1] ** 2) / (K_t + K_a * V_aircraft[-2] ** 2))
            long_and_lat = inverse_haversine([lat[-1], long[-1]], dist / 1000, bearing)
            long.append(long_and_lat[1])
            lat.append(long_and_lat[0])
            onground.append(True)

        geoaltitude.pop(0)
        V_aircraft.pop(0)
        V_vertical.pop(0)
        long.pop(0)
        lat.pop(0)
        t.pop(0)
        onground.pop(0)

        altitude = [x + (one_set_of_aircraft['altitude'].iloc[0] - one_set_of_aircraft['geoaltitude'].iloc[0]) for x in geoaltitude]  # Heathrow at an elevation of 25m
        callsign = [one_set_of_aircraft['callsign'].iloc[0]] * len(V_aircraft)
        day = [one_set_of_aircraft['day'].iloc[0]] * len(V_aircraft)
        destination = [one_set_of_aircraft['destination'].iloc[0]] * len(V_aircraft)
        firstseen = [one_set_of_aircraft['firstseen'].iloc[0]] * len(V_aircraft)
        hour = [one_set_of_aircraft['hour'].iloc[0]] * len(V_aircraft)
        icao24 = [one_set_of_aircraft['icao24'].iloc[0]] * len(V_aircraft)
        last_pos = [one_set_of_aircraft['last_position'].iloc[0]] * len(V_aircraft)
        lastseen = [one_set_of_aircraft['lastseen'].iloc[0]] * len(V_aircraft)
        origin = [one_set_of_aircraft['origin'].iloc[0]] * len(V_aircraft)
        spi = [one_set_of_aircraft['spi'].iloc[0]] * len(V_aircraft)
        squawk = [one_set_of_aircraft['squawk'].iloc[0]] * len(V_aircraft)
        alert = [one_set_of_aircraft['alert'].iloc[0]] * len(V_aircraft)
        track = [bearing * 180/np.pi] * len(V_aircraft)

        cuttoff = runway_cuttoff(airport, track[0], lat[-1], long[-1], 1)
        
        
        
        # cuttoff = 1 if the aircraft has overran the runway upon arrival
        if cuttoff:
            incorrect_flights += 1
            continue
            
        droneTargetRunway = DroneTargetRunwayIdentifier(airport, track_init, lat_init, long_init, 1) #Arrival - 1
        
        
        landing = pd.DataFrame(
            {'alert': alert, 'altitude': altitude, 'callsign': callsign, 'day': day, 'destination': destination,
             'firstseen': firstseen, 'geoaltitude': geoaltitude, 'groundspeed': V_aircraft, 'hour': hour,
             'icao24': icao24, 'last_position': last_pos, 'lastseen': lastseen, 'latitude': lat, 'longitude': long,
             'onground': onground, 'origin': origin, 'spi': spi, 'squawk': squawk, 'timestamp': t, 'track': track,
             'vertical_rate': V_vertical})
        one_set_of_aircraft = pd.concat([one_set_of_aircraft, landing], ignore_index=True)
        one_set_of_aircraft['runway'] = runway
        one_set_of_aircraft['drone_target_runway'] = droneTargetRunway
        final_data_frame_arrive = pd.concat([final_data_frame_arrive, one_set_of_aircraft], ignore_index=True)

    airport_lon = airportsdata.load('IATA')[airport]['lon']
    airport_lat = airportsdata.load('IATA')[airport]['lat']
    utm_zone = utm.from_latlon(airport_lat,airport_lon)[2]
    
    p = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    long_lat_arrive = p(final_data_frame_arrive['longitude'], final_data_frame_arrive['latitude'])
    final_data_frame_arrive['longitude'] = long_lat_arrive[0]
    final_data_frame_arrive['latitude'] = long_lat_arrive[1]

    final_data_frame_arrive = final_data_frame_arrive.iloc[:, :-1]
    final_data_frame_arrive = final_data_frame_arrive.iloc[:, :-1]

    final_data_frame_arrive.to_csv(
        airport + '/' + model + '_' + airport + '/' + 'Arrival/LANDING_UTM_ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')
    
    print("There are a total of " + str(total_flights) + " arrival flights at " + airport)
    print("There are a total of " + str(incorrect_flights) + " incorrect arrival flights transformed at " + airport)
    
    return total_flights, incorrect_flights

def plotting_trajectories(time_start, time_stop, model, airport, arrive_depart):
    # Database for either arrival or departure
    if arrive_depart == 'depart':
        database_NO_CHANGE = pd.read_csv(
            airport + '/' + model + '_' + airport + '/' + 'Departure/UTM_DEPART_' + model + ' ' + airport + ' time - ' +
            time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

        database_CHANGE = pd.read_csv(airport + '/' + model + '_' + airport + '/' + 'Departure/TAKEOFF_UTM_DEPART_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    elif arrive_depart == 'arrive':
        database_NO_CHANGE = pd.read_csv(
        airport + '/' + model + '_' + airport + '/' + 'Arrival/UTM_ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

        database_CHANGE = pd.read_csv(airport + '/' + model + '_' + airport + '/' + 'Arrival/LANDING_UTM_ARRIVE_' + model + ' ' + airport + ' time - ' +
        time_start.replace(':', '-') + ' to ' + time_stop.replace(':', '-') + '.csv')

    # Airport Coords - Centre of airports Heathrow - LHR Gatwick - LGW
    airport_icao = airportsdata.load('IATA')[airport]['icao']
    lat = airportsdata.load('IATA')[airport]['lat']
    lon = airportsdata.load('IATA')[airport]['lon']
    
    airport_coords = [lat, lon]

    # Creating a boundary box with radius/width
    radius = 10
    bounds = boundary(airport_coords, radius)
    # bounds_LHR = [west, south, east, north]

    # Converting UTM back to longitude and latitude for the plots.
    p = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
    long_lat_CHANGE = p(database_CHANGE['longitude'], database_CHANGE['latitude'],inverse=True)
    database_CHANGE['longitude'] = long_lat_CHANGE[0]
    database_CHANGE['latitude'] = long_lat_CHANGE[1]

    # Working out the indexes for a change in aircraft for the NO_CHANGE data
    diff_aircraft_NO_CHANGE = database_NO_CHANGE['callsign'].ne(database_NO_CHANGE['callsign'].shift()).astype(int)
    indices_NO_CHANGE = diff_aircraft_NO_CHANGE[diff_aircraft_NO_CHANGE == 1].index
    indices_NO_CHANGE = indices_NO_CHANGE.tolist()
    indices_NO_CHANGE.append(len(database_NO_CHANGE['callsign']))

    # Working out the indexes for a change in aircraft for the CHANGED data
    diff_aircraft_CHANGE = database_CHANGE['callsign'].ne(database_CHANGE['callsign'].shift()).astype(int)
    indices_CHANGE = diff_aircraft_CHANGE[diff_aircraft_CHANGE == 1].index
    indices_CHANGE = indices_CHANGE.tolist()
    indices_CHANGE.append(len(database_CHANGE['callsign']))

    random_number = int(random.uniform(0,len(indices_CHANGE)-1))
    
    x = random_number
    aircraft_NO_CHANGE = database_NO_CHANGE[indices_NO_CHANGE[x]:indices_NO_CHANGE[x + 1]]

    aircraft_CHANGE = database_CHANGE[indices_CHANGE[x]:indices_CHANGE[x + 1]]

    fig, ax = plt.subplots(
        1, figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )

    ax.add_feature(countries(scale="50m"))
    ax.add_feature(rivers(scale="50m"))
    ax.add_feature(lakes(scale="50m"))
    ax.add_feature(ocean(scale="50m"))
    # ax.spines["geo"].set_visible(False)
    airports[airport_icao].plot(ax,labels=dict(fontsize=17))
    plt.plot(aircraft_NO_CHANGE['longitude'], aircraft_NO_CHANGE['latitude'],
             color='red', linewidth=4,
             transform=ccrs.PlateCarree(),label="Flight " + aircraft_NO_CHANGE['callsign'].iloc[0]
             )
    # ax.set_extent((west, east, south, north))
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    # ax.set_extent((-7, 13, 40, 58))
    #plt.title(airport + ' flight trajectory base data: ' + arrive_depart, fontsize=25)
    plt.legend(fontsize="20")
    plt.tight_layout()
    #plt.savefig('Plots/' + airport + '_flight_trajectory_base_data-' + arrive_depart + '.png')
    #plt.close()
    plt.show()

    
    fig, ax = plt.subplots(
        1, figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )
    ax.add_feature(countries(scale="50m"))
    ax.add_feature(rivers(scale="50m"))
    ax.add_feature(lakes(scale="50m"))
    ax.add_feature(ocean(scale="50m"))
    # ax.spines["geo"].set_visible(False)
    airports[airport_icao].plot(ax,labels=dict(fontsize=17))
    plt.plot(aircraft_CHANGE['longitude'], aircraft_CHANGE['latitude'],
             color='blue', linewidth=4,
             transform=ccrs.PlateCarree(),label="Flight " + aircraft_CHANGE['callsign'].iloc[0] + " extrapolated data"
             )
    plt.plot(aircraft_NO_CHANGE['longitude'], aircraft_NO_CHANGE['latitude'],
             color='red', linewidth=4,
             transform=ccrs.PlateCarree(),label="Flight " + aircraft_NO_CHANGE['callsign'].iloc[0] + " base data"
             )

    # ax.set_extent((west, east, south, north))
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    # ax.set_extent((-7, 13, 40, 58))
    #plt.title(airport + ' flight trajectory including extrapolation: ' + arrive_depart, fontsize=25)
    plt.legend(fontsize="20")
    plt.tight_layout()
    #plt.savefig('Plots/' + airport + '_flight_trajectory_including_extrapolation-' + arrive_depart + '.png')
    #plt.close()
    plt.show()
    
    
    
    # PLOT TO SHOW ALL TRAJECTORIES FOR A320 BEFORE ANY CHANGES
    fig, ax = plt.subplots(
        1, figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )

    ax.add_feature(countries(scale="50m"))
    ax.add_feature(rivers(scale="50m"))
    ax.add_feature(lakes(scale="50m"))
    ax.add_feature(ocean(scale="50m"))
    # ax.spines["geo"].set_visible(False)
    airports[airport_icao].plot(ax, labels=dict(fontsize=17))
    aircraft_NO_CHANGE = database_NO_CHANGE[indices_NO_CHANGE[0]:indices_NO_CHANGE[0 + 1]]
    plt.plot(aircraft_NO_CHANGE['longitude'], aircraft_NO_CHANGE['latitude'],
             color='red', linewidth=2,
             transform=ccrs.PlateCarree(),label="Aircraft Trajectory"
             )
    for i in range(1,len(indices_NO_CHANGE)-1):
        aircraft_NO_CHANGE = database_NO_CHANGE[indices_NO_CHANGE[i]:indices_NO_CHANGE[i + 1]]
        plt.plot(aircraft_NO_CHANGE['longitude'], aircraft_NO_CHANGE['latitude'],
                 color='red', linewidth=2,
                 transform=ccrs.PlateCarree()
                 )
    # ax.set_extent((west, east, south, north))
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    # ax.set_extent((-7, 13, 40, 58))
    # plt.title(airport + ' flight trajectory base data: ' + arrive_depart, fontsize=25)
    plt.legend(fontsize="20")
    plt.tight_layout()
    plt.show()
    
    # PLOT TO SHOW ALL TRAJECTORIES FOR A320 AFTER TRANSFORMATION CHANGES
    fig, ax = plt.subplots(
        1, figsize=(10, 10),
        subplot_kw=dict(projection=OSGB())
    )

    ax.add_feature(countries(scale="50m"))
    ax.add_feature(rivers(scale="50m"))
    ax.add_feature(lakes(scale="50m"))
    ax.add_feature(ocean(scale="50m"))
    # ax.spines["geo"].set_visible(False)
    airports[airport_icao].plot(ax, labels=dict(fontsize=17))
    aircraft_CHANGE = database_CHANGE[indices_CHANGE[0]:indices_CHANGE[0 + 1]]
    plt.plot(aircraft_CHANGE['longitude'], aircraft_CHANGE['latitude'],
             color='red', linewidth=2,
             transform=ccrs.PlateCarree(),label="Aircraft Trajectory"
             )
    for i in range(1,len(indices_CHANGE)-1):
        aircraft_CHANGE = database_CHANGE[indices_CHANGE[i]:indices_CHANGE[i + 1]]
        plt.plot(aircraft_CHANGE['longitude'], aircraft_CHANGE['latitude'],
                 color='red', linewidth=2,
                 transform=ccrs.PlateCarree()
                 )
    # ax.set_extent((west, east, south, north))
    ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]))
    # ax.set_extent((-7, 13, 40, 58))
    # plt.title(airport + ' flight trajectory base data: ' + arrive_depart, fontsize=25)
    plt.legend(fontsize="20")
    plt.tight_layout()
    plt.show()
    
# NEW
def runway_identidier_points(airport_icao):
    airport = airports[airport_icao].runways
    airport_lon = airportsdata.load()[airport_icao]['lon']
    airport_lat = airportsdata.load()[airport_icao]['lat']
    
    utm_zone = utm.from_latlon(airport_lat,airport_lon)[2]

    
    boxed_area_points = pd.DataFrame()
    
    distance_along = 0.75
    distance_perpe = 0.09
    

    for runway in airport:
        lat_lon = [runway[0], runway[1]]   # Start of runway latitude, longitude
        bearing = runway[2]*np.pi/180
        bearing_perpe = bearing - np.pi/2
        ### two step transformation of the pointes required
        point_nw = inverse_haversine(lat_lon, distance_along, bearing - np.pi)
        point_nw = inverse_haversine(point_nw, distance_perpe, bearing_perpe)

        point_ne = inverse_haversine(lat_lon, distance_along, bearing)
        point_ne = inverse_haversine(point_ne, distance_perpe, bearing_perpe)

        point_se = inverse_haversine(lat_lon, distance_along, bearing)
        point_se = inverse_haversine(point_se, distance_perpe, bearing_perpe + np.pi)

        point_sw = inverse_haversine(lat_lon, distance_along, bearing - np.pi)
        point_sw = inverse_haversine(point_sw, distance_perpe, bearing_perpe + np.pi)
        
        if bearing > np.pi:
            ### two step transformation of the pointes required
            point_nw = inverse_haversine(lat_lon, distance_along, bearing)
            point_nw = inverse_haversine(point_nw, distance_perpe, bearing_perpe + np.pi)

            point_ne = inverse_haversine(lat_lon, distance_along, bearing - np.pi)
            point_ne = inverse_haversine(point_ne, distance_perpe, bearing_perpe + np.pi)

            point_se = inverse_haversine(lat_lon, distance_along, bearing - np.pi)
            point_se = inverse_haversine(point_se, distance_perpe, bearing_perpe)

            point_sw = inverse_haversine(lat_lon, distance_along, bearing)
            point_sw = inverse_haversine(point_sw, distance_perpe, bearing_perpe)
            
        # 0 - latitude / 1 - longitude
        runway_name = str(runway[3])
        data = {
            (runway_name, 'latitude'): [point_nw[0], point_ne[0], point_se[0], point_sw[0]],
            (runway_name, 'longitude'): [point_nw[1], point_ne[1], point_se[1], point_sw[1]],
        }
        new_runway = pd.DataFrame(data)
        boxed_area_points = pd.concat([boxed_area_points, new_runway], axis=1)
        

    return boxed_area_points

# NEW
def runway_area_identifier(airport, aircraft_bearing, aircraft_lat, aircraft_lon):
    # airport is in IATA
    airport_icao = airportsdata.load('IATA')[airport]['icao']
    # Airport runway areas in Lon and Lat
    airport_runway_areas = runway_identidier_points(airport_icao)
    airport_df = pd.DataFrame(airports[airport_icao].runways[:])

    runway_bearings = airport_df['bearing']
    runway_names = airport_df['name']

    # Iterate through the runway list
    for i in range(0,len(runway_bearings)):
        # Identify if the bearing of the aircraft lines up with the runway bearing
        if (runway_bearings[i] - 10.0) <= aircraft_bearing <= (runway_bearings[i] + 10.0):
            #print(airport_runway_areas[runway_names[i]])
            
            # Obtaining the longitude and latitudes of the boxed areas used to identify the runways 
            runway_boxed_lat =  airport_runway_areas[(runway_names[i],'latitude')]
            runway_boxed_lon = airport_runway_areas[(runway_names[i],'longitude')]
            # 0-NW 1-NE 2-SE 3-SW of boxed area
            
            # Obtaining the gradients to verify which runway (R or L) the aircraft is either landing or taking off from
            gradient_north = (runway_boxed_lat[1]-runway_boxed_lat[0]) / (runway_boxed_lon[1] - runway_boxed_lon[0])
            gradient_south = (runway_boxed_lat[2]-runway_boxed_lat[3]) / (runway_boxed_lon[2] - runway_boxed_lon[3])
            
            # Obtaining the c-intercept for the straight line 
            c_inter_north = runway_boxed_lat[0] - gradient_north*runway_boxed_lon[0]
            c_inter_south = runway_boxed_lat[3] - gradient_south*runway_boxed_lon[3]
            
            # Calculating the north and south boundaries that the aircraft must be within
            north_boundary_y = aircraft_lon*gradient_north + c_inter_north
            south_boundary_y = aircraft_lon*gradient_south + c_inter_south
            
            
            if south_boundary_y <= aircraft_lat <= north_boundary_y:
                return runway_names[i]    

# NEW    
def runway_cuttoff(airport, aircraft_bearing, aircraft_lat, aircraft_lon, depart_arrive):
    
    # depart_arrive: 1 - arrive, 0 - depart
    
    '''
    Need to identify the cut off line for which if the aircraft goes part this line, then the aircraft is forced to stop at that point
    OR the aircraft could be completely removed from the dataset
    the aircraft has to be on one side of the runway cuttoff
    
    if it is an arrival, it must be the oppsite side of the runway for a cuttoff
    For departure, it is the same runway that the aircraft is taking off from.
    '''
    airport_icao = airportsdata.load('IATA')[airport]['icao']
    airport_df = pd.DataFrame(airports[airport_icao].runways[:])
    runway_names = airport_df['name']
    
    # Average runway width plus a bit
    runway_width = 0.06
    runway_saftey = 0.015
    
    # ARRIVAL FLIGHTS
    if (depart_arrive):
        if aircraft_bearing > 180:
            target_runway_bearing = aircraft_bearing - 180 # In degrees
        else:
            target_runway_bearing = aircraft_bearing + 180 # In degrees
    else:
        target_runway_bearing = aircraft_bearing
    
    target_runway = runway_area_identifier(airport, target_runway_bearing, aircraft_lat, aircraft_lon)
    
    target_runway_bearing = target_runway_bearing * np.pi/180
    
    for i in range(0, len(runway_names)):
        if target_runway == runway_names[i]:
            lat_lon = [airport_df['latitude'].iloc[i], airport_df['longitude'].iloc[i]]
            
            point_safety = inverse_haversine(lat_lon, runway_saftey, target_runway_bearing + np.pi)
            
            point_n = inverse_haversine(point_safety, runway_width/2.0, target_runway_bearing - np.pi/2)
            point_s = inverse_haversine(point_safety, runway_width/2.0, target_runway_bearing + np.pi/2)
            
            if target_runway_bearing > np.pi:
                point_safety = inverse_haversine(lat_lon, runway_saftey, target_runway_bearing - np.pi)
                
                point_n = inverse_haversine(lat_lon, runway_width/2.0, target_runway_bearing + np.pi/2)
                point_s = inverse_haversine(lat_lon, runway_width/2.0, target_runway_bearing - np.pi/2)
        
        
            coefficients = np.polyfit([point_n[1], point_s[1]], [point_n[0], point_s[0]], 1)
            gradient = coefficients[0]
            intercept = coefficients[1]
            y_bound = gradient * aircraft_lon + intercept
    
            if np.pi/2 <= target_runway_bearing < 3*np.pi/2:
                if y_bound >= aircraft_lat:
                    return 0
                    #print("AIRCRAFT HAS LANDED/DEPARTED WITHIN LIMITS")
                else:
                    return 1
                    #print("AIRCRAFT DID NOT LAND/DEPART WITHIN LIMITS")
            else:
                if y_bound <= aircraft_lat:
                    return 0
                    #print("AIRCRAFT HAS LANDED/DEPARTED WITHIN LIMITS")
                else:
                    return 1
                    #print("AIRCRAFT DID NOT LAND/DEPART WITHIN LIMITS")

# NEW            
def airport_error(airport, date, total_flights_a, incorrect_flights_a, total_flights_d, incorrect_flights_d):
    file_name = "Airport_error_check.csv"
    date = date[:10]
    if os.path.exists(file_name):
        airport_error_check_df = pd.read_csv('Airport_error_check.csv', header=[0,1], index_col=0)
        
        if total_flights_d == 0:
            departError = 0
        else:
            departError = incorrect_flights_d / total_flights_d    
        if total_flights_a == 0:
            arriveError = 0
        else:
            arriveError = incorrect_flights_a / total_flights_a
        
        
        new_data = {
            ('Airport','-'): [airport],
            ('Date','-'): [date],
            ('Departure', 'Total Flights'): [total_flights_d],
            ('Departure', 'Incorrect Flights'): [incorrect_flights_d],
            ('Departure', 'Flights Used for Sim'): [total_flights_d - incorrect_flights_d],
            ('Departure', 'Error Percentage / %'): [departError * 100],
            ('Arrival', 'Total Flights'): [total_flights_a],
            ('Arrival', 'Incorrect Flights'): [incorrect_flights_a],
            ('Arrival', 'Flights Used for Sim'): [total_flights_a - incorrect_flights_a],
            ('Arrival', 'Error Percentage / %'): [arriveError * 100],
        }
        
        new_df = pd.DataFrame(new_data)
        
        # Checks if there is an entry for the same airport and date in the csv file already
        for i in range(0,len(airport_error_check_df[('Airport','-')])):
            if (airport_error_check_df[('Airport','-')].loc[i] == airport) and (airport_error_check_df[('Date','-')].loc[i] == date):
                airport_error_check_df.iloc[i] = new_df.iloc[0]
                break
            if i == len(airport_error_check_df[('Airport','-')]) - 1:
                airport_error_check_df = pd.concat([airport_error_check_df, new_df], ignore_index=True)
    else:
        if total_flights_d == 0:
            departError = 0
        else:
            departError = incorrect_flights_d / total_flights_d    
        if total_flights_a == 0:
            arriveError = 0
        else:
            arriveError = incorrect_flights_a / total_flights_a
        
        
        
        data = {
            ('Airport','-'): [airport],
            ('Date','-'): [date],
            ('Departure', 'Total Flights'): [total_flights_d],
            ('Departure', 'Incorrect Flights'): [incorrect_flights_d],
            ('Departure', 'Flights Used for Sim'): [total_flights_d - incorrect_flights_d],
            ('Departure', 'Error Percentage / %'): [departError * 100],
            ('Arrival', 'Total Flights'): [total_flights_a],
            ('Arrival', 'Incorrect Flights'): [incorrect_flights_a],
            ('Arrival', 'Flights Used for Sim'): [total_flights_a - incorrect_flights_a],
            ('Arrival', 'Error Percentage / %'): [arriveError * 100],
        }
        airport_error_check_df = pd.DataFrame(data)
        
    airport_error_check_df.to_csv("Airport_error_check.csv")

# NEW              
def runway_drone_area_points(airport_iata, model, half_width, half_length):
    airport_icao = airportsdata.load('IATA')[airport_iata]['icao']
    
    airport = airports[airport_icao].runways
    airport_lon = airportsdata.load()[airport_icao]['lon']
    airport_lat = airportsdata.load()[airport_icao]['lat']
    utm_zone = utm.from_latlon(airport_lat,airport_lon)[2]
    
    # Define the projection
    p = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    
    drone_area_points = pd.DataFrame()

    for runway in airport:
        lat_lon = [runway[0], runway[1]]   # Start of runway latitude, longitude
        bearing = runway[2]*np.pi/180
        bearing_perpe = bearing - np.pi/2
        ### two step transformation of the pointes required
        point_nw = inverse_haversine(lat_lon, half_length, bearing - np.pi)
        point_nw = inverse_haversine(point_nw, half_width, bearing_perpe)

        point_ne = inverse_haversine(lat_lon, half_length, bearing)
        point_ne = inverse_haversine(point_ne, half_width, bearing_perpe)

        point_se = inverse_haversine(lat_lon, half_length, bearing)
        point_se = inverse_haversine(point_se, half_width, bearing_perpe + np.pi)

        point_sw = inverse_haversine(lat_lon, half_length, bearing - np.pi)
        point_sw = inverse_haversine(point_sw, half_width, bearing_perpe + np.pi)
        
        if bearing > np.pi:
            ### two step transformation of the pointes required
            point_nw = inverse_haversine(lat_lon, half_length, bearing)
            point_nw = inverse_haversine(point_nw, half_width, bearing_perpe + np.pi)

            point_ne = inverse_haversine(lat_lon, half_length, bearing - np.pi)
            point_ne = inverse_haversine(point_ne, half_width, bearing_perpe + np.pi)

            point_se = inverse_haversine(lat_lon, half_length, bearing - np.pi)
            point_se = inverse_haversine(point_se, half_width, bearing_perpe)

            point_sw = inverse_haversine(lat_lon, half_length, bearing)
            point_sw = inverse_haversine(point_sw, half_width, bearing_perpe)
        runway_name = str(runway[3])
        
        # CONVERSION INTO UTM UNITS
        point_nw = p(point_nw[1],point_nw[0])
        point_ne = p(point_ne[1],point_ne[0])
        point_se = p(point_se[1],point_se[0])
        point_sw = p(point_sw[1],point_sw[0])
        
        data = {
            (runway_name, 'latitude'): [point_nw[1], point_ne[1], point_se[1], point_sw[1]],
            (runway_name, 'longitude'): [point_nw[0], point_ne[0], point_se[0], point_sw[0]],
        }
        new_runway = pd.DataFrame(data)
        drone_area_points = pd.concat([drone_area_points, new_runway], axis=1)

    drone_area_points.to_csv(airport_iata + '/' + model + '_' + airport_iata + '/' + airport_iata + "_drone_area.csv")

# NEW    
def DroneTargetRunwayIdentifier(airport, aircraft_bearing, aircraft_lat, aircraft_lon, depart_arrive):
    # airport is in IATA
    airport_icao = airportsdata.load('IATA')[airport]['icao']
    # Airport runway areas in Lon and Lat
    airport_runway_areas = runway_identidier_points(airport_icao)
    airport_df = pd.DataFrame(airports[airport_icao].runways[:])

    runway_bearings = airport_df['bearing']
    runway_names = airport_df['name']
    
    #depart_arrive: 1 - arrive 0 - depart
    if (not depart_arrive):
        if aircraft_bearing <= 180:
            aircraft_bearing = aircraft_bearing + 180
        else:
            aircraft_bearing = aircraft_bearing - 180
        

    # Iterate through the runway list
    for i in range(0,len(runway_bearings)):
        # Identify if the bearing of the aircraft lines up with the runway bearing
        if (runway_bearings[i] - 10.0) <= aircraft_bearing <= (runway_bearings[i] + 10.0):
            #print(airport_runway_areas[runway_names[i]])
            
            # Obtaining the longitude and latitudes of the boxed areas used to identify the runways 
            runway_boxed_lat =  airport_runway_areas[(runway_names[i],'latitude')]
            runway_boxed_lon = airport_runway_areas[(runway_names[i],'longitude')]
            # 0-NW 1-NE 2-SE 3-SW of boxed area
            
            # Obtaining the gradients to verify which runway (R or L) the aircraft is either landing or taking off from
            gradient_north = (runway_boxed_lat[1]-runway_boxed_lat[0]) / (runway_boxed_lon[1] - runway_boxed_lon[0])
            gradient_south = (runway_boxed_lat[2]-runway_boxed_lat[3]) / (runway_boxed_lon[2] - runway_boxed_lon[3])
            
            # Obtaining the c-intercept for the straight line 
            c_inter_north = runway_boxed_lat[0] - gradient_north*runway_boxed_lon[0]
            c_inter_south = runway_boxed_lat[3] - gradient_south*runway_boxed_lon[3]
            
            # Calculating the north and south boundaries that the aircraft must be within
            north_boundary_y = aircraft_lon*gradient_north + c_inter_north
            south_boundary_y = aircraft_lon*gradient_south + c_inter_south
            
            
            if south_boundary_y <= aircraft_lat <= north_boundary_y:
                return runway_names[i]  

# NEW            
def drone_points(airport, model, radius):
    airport_lon = airportsdata.load("IATA")[airport]['lon']
    airport_lat = airportsdata.load("IATA")[airport]['lat']
    centre_coords = [airport_lat, airport_lon]
    
    utm_zone = utm.from_latlon(airport_lat,airport_lon)[2]
    

    dx_points = 100

    points = [None] * dx_points
    heading = [None] * dx_points
    angles = np.linspace(0, 2 * math.pi, dx_points)

    # Obtaining Coords of points of drones around the airport
    for i in range(0, dx_points):
        points[i] = inverse_haversine(centre_coords, radius, angles[i])
        heading[i] = angles[i] + math.pi
        if heading[i] > 2 * math.pi:
            heading[i] = heading[i] - 2 * math.pi

    dlat, dlon = zip(*points)

    p = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    drone_UTM = p(dlon, dlat)

    drone_long_lat = pd.DataFrame(columns=['longitude', 'latitude', 'heading'])
    drone_long_lat['longitude'] = drone_UTM[0]
    drone_long_lat['latitude'] = drone_UTM[1]
    drone_long_lat['heading'] = heading

    drone_long_lat.to_csv(airport + '/' + model + '_' + airport + '/' + 'DroneStartPos/drone_inital_positions_' + str(radius) + 'km.csv')

# NEW    
def make_dir(airport, model):
    arrive = "Arrival"
    depart = "Departure"
    dronePos = "DroneStartPos"
    model_path = model + "_" + airport
    
    try:
        path_1a = os.path.join(airport, model_path)
        path_1b = os.path.join(path_1a, arrive)
        os.makedirs(path_1b, exist_ok=True)

        path_2a = os.path.join(airport, model_path)
        path_2b = os.path.join(path_2a, depart)
        os.makedirs(path_2b, exist_ok=True)
        
        path_1c = os.path.join(path_1a,dronePos)
        os.makedirs(path_1c,exist_ok=True)
        
        

    except OSError as error:
        print("ERROR in making directory")


def obtain_flight_data(airport, model, start_time, stop_time):
    # Makes folder for airport and aircraft
    make_dir(airport, model)
    # Obtains all the data from Opensky for airport and aircraft
    traffic_data_for_model_and_airport(start_time, stop_time, model, airport)
    # Convert to SI Units
    convert_to_SI(start_time, stop_time, model, airport)
    # Transformation of flights and identifies the errors in transformations
    total_flights_d, incorrect_flights_d = additional_long_lat_depart(start_time, stop_time, model, airport)
    total_flights_a, incorrect_flights_a = additional_long_lat_arrive(start_time, stop_time, model, airport)
    # Export flight error data to csv
    airport_error(airport, start_time, total_flights_a, incorrect_flights_a, total_flights_d, incorrect_flights_d)
    # Obtain drone area points for the airport for each runway
    runway_drone_area_points(airport, model, 0.25, 1.5)
    
    for i in np.arange(1.0, 5.5, 0.5):
        drone_points(airport, model, i)

"""
A320_database = aircraft_model_database('A320', 'AIRBUS', 'Airbus')
B737_database = aircraft_model_database('737', 'BOEING', 'Boeing')
B747_database = aircraft_model_database('747', 'BOEING', 'Boeing')
A380_database = aircraft_model_database('A380', 'AIRBUS', 'Airbus')
B777_database = aircraft_model_database('777', 'BOEING', 'Boeing')
A350_database = aircraft_model_database('A350', 'AIRBUS', 'Airbus')
A321_database = aircraft_model_database('A321', 'AIRBUS', 'Airbus')
"""

model = "A320"

### 10 busiest UK airports

start_time = "2019-08-04 04:00"
stop_time = "2019-08-04 23:45"

## LHR - London Heathrow Airport
airport = "LHR"

### LGW - London Gatwick Airport
# airport = "LGW"

### MAN - Manchester Airport
# airport = "MAN"

### STN - Stansted Airport
# airport = "STN"

### LTN - Luton Airport
# airport = "LTN"

### EDI - Edinburgh Airport
# airport = "EDI"

### BHX - Birmingham Airport
# airport = "BHX"

### GLA - Glasgow Airport
# airport = "GLA"

### BRS - Bristol Airport
# airport = "BRS"

## NCL - Newcastle Airport
# airport = "NCL"


### International airports

### FRA - Frankfurt Airport
# start_time = "2019-09-19 04:00"
# stop_time = "2019-09-19 23:45"
# airport = "FRA"

### FCO - Rome Fiumicino Airport
# start_time = "2019-09-19 04:00"
# stop_time = "2019-09-19 23:45"
# airport = "FCO"


obtain_flight_data(airport, model, start_time, stop_time)
#plotting_trajectories(start_time, stop_time, model, airport, 'depart')
#plotting_trajectories(start_time, stop_time, model, airport, 'arrive')