import geopandas as gpd
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math

#State of Arizona
state_filepath = "tl_2023_us_state.shp"
states = gpd.read_file(state_filepath)
arizona = states.loc[states["NAME"] == "Arizona"].copy()
arizona["INTPTLAT"] = pd.to_numeric(arizona["INTPTLAT"], errors="coerce")
arizona["INTPTLON"] = pd.to_numeric(arizona["INTPTLON"], errors="coerce")
arizona_gpd = gpd.GeoDataFrame(arizona, geometry="geometry")

#Arizona Counties General Information
county_filepath = "tl_2023_us_county.shp"
counties = gpd.read_file(county_filepath)
arizona_counties = counties.loc[counties["STATEFP"] == "04"].copy()
arizona_counties["INTPTLAT"] = pd.to_numeric(arizona_counties["INTPTLAT"], errors="coerce")
arizona_counties["INTPTLON"] = pd.to_numeric(arizona_counties["INTPTLON"], errors="coerce")
arizona_counties_gpd = gpd.GeoDataFrame(arizona_counties, geometry="geometry")

#Arizona Census Tracts General Information
census_tract_filepath = "Census_Tracts_Data_Arizona_2023.csv"
census_tract_general = pd.read_csv(census_tract_filepath, encoding='utf-8', dtype={"STATEFIPS": str, "COUNTYFIPS": str, "STCOFIPS": str, "TRACT": str, "TRACTFIPS": str})

#Census Tracts Geometric Coordinates
shapefile_filepath = "tl_rd22_04_tract.shp"
census_tract_coordinates = gpd.read_file(shapefile_filepath)

specific_columns = ["GEOID", "INTPTLAT", "INTPTLON", "geometry"]
census_tract = pd.merge(census_tract_general, census_tract_coordinates[specific_columns], left_on="TRACTFIPS", right_on="GEOID", how='left')
census_tract = census_tract.drop(columns=["NRI_ID", "STATEABBRV", "STATEFIPS", "COUNTYTYPE", "COUNTYFIPS", "STCOFIPS", "TRACT", "RISK_VALUE", "RISK_SCORE", "RISK_RATNG", "RISK_SPCTL", "GEOID"])

census_tract["INTPTLAT"] = pd.to_numeric(census_tract["INTPTLAT"], errors="coerce")
census_tract["INTPTLON"] = pd.to_numeric(census_tract["INTPTLON"], errors="coerce")

census_tract.rename(columns={"OID_": "ID", "INTPTLAT": "Latitude", "INTPTLON": "Longitude"}, inplace=True)

#Assign specific risk values to each wildfire risk level.
risk_values = {
    "No Rating": 0,
    "Very Low": 1,
    "Relatively Low": 2,
    "Relatively Moderate": 3,
    "Relatively High": 4,
    "Very High": 5,
}

census_tract["Wildfire Risk Value"] = census_tract["WFIR_RISKR"].map(risk_values)

#Convert to GeoDataframe
census_tract_gpd = gpd.GeoDataFrame(census_tract, geometry="geometry")

#Satellite Coverage (EPSG:6933 to obtain unit coordinates in terms of meters)
census_tract_gpd = census_tract_gpd.to_crs("EPSG:6933")
arizona_gpd = arizona_gpd.to_crs("EPSG:6933")

#Satellite radius is in meters (m).
radius = 120000
arizona_center = arizona_gpd.geometry.centroid.iloc[0]

#Point in the format of (Longitude, Latitude)
center_point = Point(arizona_center.x, arizona_center.y)

satellite_coverage = center_point.buffer(radius)

intersections_polygon = census_tract_gpd[census_tract_gpd.intersects(satellite_coverage)]

average_risk_value_satellite = intersections_polygon["Wildfire Risk Value"].mean()

if average_risk_value_satellite <= 2:
    new_polygons = []
    for index, county in census_tract_gpd.iterrows():
        census_tract_ID = county["ID"]
        census_tract_state = county["STATE"]
        census_tract_FIPS = county["TRACTFIPS"]
        census_tract_population = county["POPULATION"]
        census_tract_risk_level = county["WFIR_RISKR"]
        census_tract_risk_value = county["Wildfire Risk Value"]
        
        #The difference operation computes the set-theoretic difference of the geometries. It returns a new geometric object that represents the portion of the county geometry that does not overlap with the satellite_coverage
        new_polygon = county.geometry.difference(satellite_coverage)
        
        if not new_polygon.is_empty:
            new_polygons.append({"ID": census_tract_ID, "State": census_tract_state, "TractFIPS": census_tract_FIPS, "Population": census_tract_population, "Wildfire Risk Level": census_tract_risk_level, "Wildfire Risk Value": census_tract_risk_value, "Polygon Geometry": new_polygon})
     
    satellite_census_tracts = gpd.GeoDataFrame(new_polygons, geometry="Polygon Geometry", crs="EPSG:6933")
    
    #Census tracts excluding satellite coverage (EPSG:4269 to change meters coordinates back into latitude/longitude coordinates)
    satellite_census_tracts = satellite_census_tracts.to_crs("EPSG:4269")
    
    #Final Data
    final_census_tracts = satellite_census_tracts
    
else:
    #Convert Census Tract and State data back into latitude/longitude coordinates.
    census_tract_gpd = census_tract_gpd.to_crs("EPSG:4269")
    arizona_gpd = arizona_gpd.to_crs("EPSG:4269")
    
    #Final Data
    final_census_tracts = census_tract_gpd
    pass

#Create the nodes for drones to be potentially located at (in degrees)
point_spacing = 0.65

bounds = final_census_tracts.total_bounds

lon_min, lat_min, lon_max, lat_max = bounds

# Apply conditional rounding
lon_min = math.floor(lon_min)
lat_min = math.floor(lat_min)
lon_max = math.ceil(lon_max)
lat_max = math.ceil(lat_max)

lon_nodes = np.arange(lon_min, lon_max + point_spacing, point_spacing)
lat_nodes = np.arange(lat_min, lat_max + point_spacing, point_spacing)
nodes = [Point(x, y) for x in lon_nodes for y in lat_nodes]

nodes_gdf = gpd.GeoDataFrame(geometry=nodes, crs=final_census_tracts.crs)

#Select nodes only inside the census tract polygons and reset the index
nodes_in_tracts = gpd.sjoin(nodes_gdf, final_census_tracts, how="inner", predicate="intersects")
nodes_in_tracts = nodes_in_tracts.reset_index(drop=True)
nodes_in_tracts["Node"] = nodes_in_tracts.index + 1

#Circle does not seem circular because the distance change (meters) in 1 degree of latitude is greater than that of 1 degree of longitude.
fig, ax = plt.subplots()
final_census_tracts.plot(ax=ax, color="papayawhip", edgecolor="black", linewidth=0.85)  # Plot polygons
nodes_in_tracts.plot(ax=ax, color="red", markersize=15, edgecolor="white", linewidth=0.2)  # Plot points
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

xmin, xmax = -115, -109
x_ticks = np.arange(xmin, xmax + 1, 1)
ax.set_xticks(x_ticks)

plt.show()

#Linear Programming Optimization Model

#Obtain Initial Dataset
nodes_in_tracts = nodes_in_tracts.to_crs("EPSG:6933")
final_census_tracts = final_census_tracts.to_crs("EPSG:6933")
arizona_gpd = arizona_gpd.to_crs("EPSG:6933")

def node_coverage_area_and_average_risk(point, state_gpd, census_tract_gpd):
    r = 45000
    circle = point.buffer(r)
    intersection = circle.intersection(state_gpd.geometry.iloc[0])
    
    intersecting_polygons = census_tract_gpd[census_tract_gpd.intersects(circle)]
    
    if not intersecting_polygons.empty:
        average_node_risk = intersecting_polygons["Wildfire Risk Value"].mean()
    else:
        average_node_risk = 0
        
    return intersection.area, average_node_risk

nodes_in_tracts[["Node Coverage Area", "Average Node Wildfire Risk"]] = nodes_in_tracts.apply(
    lambda x: node_coverage_area_and_average_risk(x["geometry"], arizona_gpd, final_census_tracts), axis=1, result_type="expand")
nodes_in_tracts.index = nodes_in_tracts.index + 1

nodes_data = {
    node: {"Coverage Area": area, "Average Wildfire Risk": average_risk, "Point Geometry": geometry}
    for node, geometry, area, average_risk in zip(nodes_in_tracts["Node"], nodes_in_tracts["geometry"], nodes_in_tracts["Node Coverage Area"], nodes_in_tracts["Average Node Wildfire Risk"])
}

def arc_distance(node1, node2):
    distance = nodes_data[node1]["Point Geometry"].distance(nodes_data[node2]["Point Geometry"])
    return distance

#Parameters
num_drones = 5
drones = range(1,num_drones+1)
battery_capacity = 100
consumption_per_unit_distance = 0.0005
recharge_cost = 0.025
time_periods = [1,2,3,4]
max_time = 4
max_distance = 85000
max_recharge_amount = 20

final_census_tracts = final_census_tracts.to_crs("EPSG:6933")
tract_ids = final_census_tracts["ID"].tolist()
nodes = list(nodes_data.keys())

arcs = {(i,j): arc_distance(i,j) for i in nodes for j in nodes if i != j and arc_distance(i,j) <= max_distance}

#Optimization Model
m = gp.Model("Wildfire Surveillance")

#Decision Variables
x = m.addVars(nodes, drones, time_periods, vtype=GRB.BINARY, name="x")
y = m.addVars(nodes, nodes, drones, time_periods[1:], vtype=GRB.BINARY, name="y")
b = m.addVars(nodes, drones, time_periods, vtype=GRB.CONTINUOUS, name="b")
r = m.addVars(nodes, drones, time_periods, vtype=GRB.CONTINUOUS, name="r")
s = m.addVars(nodes, drones, vtype=GRB.BINARY, name="s")

#Objective Function
m.setObjective(sum((nodes_data[i]["Average Wildfire Risk"] + (nodes_data[i]["Coverage Area"]/1000000000)) * x[i,d,t] for i in nodes for d in drones for t in time_periods) + sum((arcs.get((i, j),0)/100000) * y[i,j,d,t] for i in nodes for j in nodes for d in drones for t in time_periods[1:]) - sum(recharge_cost * r[i,d,t] for i in nodes for d in drones for t in time_periods), GRB.MAXIMIZE)

#Constraints
#Constraint 1: drones are at full battery at all starting nodes.
for i in nodes:
    for d in drones:
        m.addConstr(b[i, d, time_periods[0]] == s[i, d] * battery_capacity)

#Constraint 2: Each drone must have a starting node.
for d in drones:
    m.addConstr(sum(s[i, d] for i in nodes) == 1)

#Constraint 3: Ensure that each drone is present at the node it starts at during the first time period
for i in nodes:
    for d in drones:
        m.addConstr(x[i, d, time_periods[0]] == s[i, d])

#Constraint 4: One drone can be at most at one node during time t
for d in drones:
    for t in time_periods:
        m.addConstr(sum(x[i,d,t] for i in nodes) <= 1)
        
#Constraint 5: One node can have at most one drone during time t
for i in nodes:
    for t in time_periods:
        m.addConstr(sum(x[i,d,t] for d in drones) <= 1)

#Constraint 6: Each node can only be visited at most once.
for i in nodes:
    m.addConstr(sum(x[i,d,t] for d in drones for t in time_periods) <= 1)

#Constraint 7: Arc determination
for i in nodes:
    for d in drones:
        for t in time_periods[1:]:
            m.addConstr(sum(y[i, j, d, t] for j in nodes if (i, j) in arcs) == x[i, d, t-1])
            m.addConstr(sum(y[j, i, d, t] for j in nodes if (j, i) in arcs) == x[i, d, t])

#Constraint 8: If a drone is at node i during time t, it cannot be at node j during time t+1 if it cannot reach node j based on traveling distance constraint during each time period
for i in nodes:
    for j in nodes:
        if (i, j) not in arcs:
            for t in time_periods[:-1]:
                for d in drones:
                    m.addConstr(x[i, d, t] + x[j, d, t+1] <= 1)

#Constraint 9: No travel occurs between nodes i and j if the distance between these two nodes exceeds the maximum traveling distance allowed.
for i in nodes:
    for j in nodes:
        for d in drones:
            for t in time_periods[1:]:
                if (i, j) not in arcs:
                    m.addConstr(y[i, j, d, t] == 0)
                    
#Constraint 10: Change in battery level each time a drone travels from one node to another node.
for i in nodes:
    for d in drones:
        for t in time_periods:
            if t > 1:
                m.addConstr(b[i, d, t] == (sum(b[j, d, t-1] - (y[j, i, d, t] * arcs.get((j, i),0) * consumption_per_unit_distance) for j in nodes if (j, i) in arcs) + r[i, d, t]) * x[i, d, t])

#Constraint 11: Ensure battery level does not exceed max capacity
for i in nodes:
    for d in drones:
        for t in time_periods:
            m.addConstr(b[i, d, t] <= battery_capacity)

#Constraint 12: Ensure drones do not travel if the battery is too low
for i in nodes:
    for j in nodes:
        for d in drones:
            for t in time_periods[1:]:
                if (i, j) in arcs:
                    m.addConstr(y[i, j, d, t] * arcs.get((i, j),0) * consumption_per_unit_distance <= b[i, d, t-1])

#Constraint 13: Recharging can only happen if a drone is present at the node  
for i in nodes:
    for d in drones:
        for t in time_periods:
            m.addConstr(r[i, d, t] <= x[i, d, t] * max_recharge_amount)

#Constraint 14: Ensure that each drone does exceed the maximum time limit
for d in drones:
    m.addConstr(sum(x[i,d,t] for i in nodes for t in time_periods) <= max_time)

#Constraint 15: Non-negativity constraints for recharging
for i in nodes:
    for d in drones:
        for t in time_periods:
            m.addConstr(r[i, d, t] >= 0)

#Constraint 16: Non-negativity constraints for battery level
for i in nodes:
    for d in drones:
        for t in time_periods:
            m.addConstr(b[i, d, t] >= 0)

m.optimize()

if m.status == GRB.Status.OPTIMAL:
    print('\nOptimal Objective Function Value: %g' % m.objVal)
    for v in m.getVars():
        if v.x > 0:
            print('%s: %g' % (v.varName, v.x))
else:
    print('No solution')