import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from data import generate_job_data

st.set_page_config(page_title="PortPilot.AI Optimiser", layout="wide")
st.title("🚢 PortPilot.AI – HT Job Scheduling & Routing Optimisation")

# --- 1. Data Loading and UI Controls ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("job_dataset.csv")
        # Re-create coords from data file for consistency
        qcs = {row['QC_M']: (row['pickup_x'], row['pickup_y']) for _, row in df[df['JOB_TYPE'] == 'DI'].iterrows()}
        yards = {row['YARD_BLOCK']: (row['pickup_x'], row['pickup_y']) for _, row in df[df['JOB_TYPE'] == 'LO'].iterrows()}
        return df, qcs, yards
    except FileNotFoundError:
        return generate_job_data()

jobs, qc_coords, yard_coords = load_data()
all_locations = {**qc_coords, **yard_coords}

st.sidebar.header("Simulation Parameters")
num_agvs = st.sidebar.slider("Number of HTs (AGVs)", 5, 80, 10)
ht_speed = 10 # seconds per sector as per requirements

st.sidebar.info(f"""
**Objective:** Complete all jobs in the shortest possible time (minimise makespan).
- **HT Fleet:** {num_agvs}
- **HT Speed:** {ht_speed} secs/sector
- **QC Handling:** 2 mins (120s)
- **Yard Handling:** 5 mins (300s)
""")

# --- 2. Build Location and Time Matrix ---
# Create a unified list of unique locations for the distance matrix
location_names = ['Depot'] + list(all_locations.keys())
location_coords = {'Depot': (0, 8)} # Start/End point in the middle of highway
for name, coords in all_locations.items():
    location_coords[name] = coords
    
coord_array = np.array([location_coords[name] for name in location_names])

def calculate_travel_time(from_node, to_node):
    """
    Calculates travel time respecting simplified directional rules from the PDF.
    Distance is Manhattan distance. Time = distance * speed.
    """
    start = coord_array[from_node]
    end = coord_array[to_node]
    
    # Manhattan distance
    distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    # Simplified rule: movements in QC/Highway lanes have directionality
    # A more complex implementation would use a graph pathfinder (e.g., A*)
    # y=5 is QC lane, y=6,7,8,9 are highway lanes
    if start[1] <= 9 and start[1] >= 5:
        # Penalize leftward movement in these lanes
        if end[0] < start[0]:
            distance *= 1.5
            
    return int(distance * ht_speed)

n_locations = len(coord_array)
time_matrix = [[calculate_travel_time(i, j) for j in range(n_locations)] for i in range(n_locations)]

# --- 3. OR-Tools Vehicle Routing Problem with Time Windows (VRPTW) ---
# Map location names to matrix indices
location_to_idx = {name: i for i, name in enumerate(location_names)}
depot_idx = 0

manager = pywrapcp.RoutingIndexManager(n_locations, num_agvs, depot_idx)
routing = pywrapcp.RoutingModel(manager)

# Create the transit callback
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return time_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Time Window dimension
handling_times = {depot_idx: 0}
for _, job in jobs.iterrows():
    pickup_node_name = job['QC_M'] if job['JOB_TYPE'] == 'DI' else job['YARD_BLOCK']
    dropoff_node_name = job['YARD_BLOCK'] if job['JOB_TYPE'] == 'DI' else job['QC_M']
    
    handling_times[location_to_idx[pickup_node_name]] = job['handling_time_pickup']
    handling_times[location_to_idx[dropoff_node_name]] = job['handling_time_dropoff']

def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return handling_times.get(from_node, 0)

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

routing.AddDimension(
    demand_callback_index,
    0,      # slack
    86400,  # vehicle capacity (max time for a day in seconds)
    False,  # start cumul to zero
    'Time')
time_dimension = routing.GetDimensionOrDie('Time')

# Define Pickups & Deliveries with optional yard locations
yard_capacities = {name: 0 for name in yard_coords}
yard_capacity_limit = 700
for index, job in jobs.iterrows():
    if job['JOB_TYPE'] == 'DI':
        pickup_node = location_to_idx[job['QC_M']]
        
        # Collect all possible dropoff nodes for this DI job
        possible_dropoffs = [location_to_idx[job['YARD_BLOCK']]]
        for alt in ['ALT_YARD_BLOCK_1', 'ALT_YARD_BLOCK_2', 'ALT_YARD_BLOCK_3']:
            if pd.notna(job[alt]):
                possible_dropoffs.append(location_to_idx[job[alt]])
        
        # Add disjunction: the solver must visit exactly one of these locations
        routing.AddDisjunction([manager.NodeToIndex(n) for n in possible_dropoffs], 1)
        
        # Tie the pickup to the group of possible dropoffs
        pickup_index = manager.NodeToIndex(pickup_node)
        routing.AddPickupAndDelivery(pickup_index, routing.GetDisjunctionIndices(pickup_index)[0])

    else: # LO job
        pickup_node = location_to_idx[job['YARD_BLOCK']]
        dropoff_node = location_to_idx[job['QC_M']]
        routing.AddPickupAndDelivery(
            manager.NodeToIndex(pickup_node),
            manager.NodeToIndex(dropoff_node)
        )

# Add Yard Capacity Constraint
for yard_name, yard_idx in location_to_idx.items():
    if yard_name in yard_coords:
        capacity_callback_index = routing.RegisterUnaryTransitCallback(
            lambda from_index, yard=yard_name: 1 if jobs[
                (jobs['YARD_BLOCK'] == yard) |
                (jobs['ALT_YARD_BLOCK_1'] == yard) |
                (jobs['ALT_YARD_BLOCK_2'] == yard) |
                (jobs['ALT_YARD_BLOCK_3'] == yard)
            ].loc[jobs.apply(lambda row: manager.IndexToNode(from_index) == location_to_idx.get(row['YARD_BLOCK'] if row['JOB_TYPE'] == 'DI' else row['QC_M']), axis=1)].any() else 0
        )
        routing.AddDimension(capacity_callback_index, 0, yard_capacity_limit, True, f"YardCapacity_{yard_name}")

# Set the main objective: minimize the makespan (total completion time)
time_dimension.SetGlobalSpanCostCoefficient(100)

# --- 4. Solve the VRP ---
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
search_parameters.time_limit.seconds = 15
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

solution = routing.SolveWithParameters(search_parameters)

# --- 5. Extract, Analyze & Visualize Solution ---
if solution:
    st.success("✅ Optimisation complete! Found a solution.")
    
    # Extract routes and metrics
    routes = []
    total_time = 0
    assigned_yard_counts = {name: 0 for name in yard_coords}

    for vehicle_id in range(num_agvs):
        index = routing.Start(vehicle_id)
        path_nodes = []
        route_time = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            path_nodes.append(location_names[node_index])
            
            # Check if this node is a dropoff for a DI job
            for i, job in jobs[jobs['JOB_TYPE'] == 'DI'].iterrows():
                if location_names[node_index] in [job['YARD_BLOCK'], job['ALT_YARD_BLOCK_1'], job['ALT_YARD_BLOCK_2'], job['ALT_YARD_BLOCK_3']]:
                    if solution.Value(routing.NextVar(manager.NodeToIndex(location_to_idx[job['QC_M']]))) == index:
                         assigned_yard_counts[location_names[node_index]] +=1
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
        
        time_var = time_dimension.CumulVar(routing.Start(vehicle_id))
        end_time_var = time_dimension.CumulVar(routing.End(vehicle_id))
        
        path_coords = [location_coords[name] for name in path_nodes]
        routes.append((vehicle_id, path_coords, path_nodes, solution.Value(end_time_var)))

    # --- KPIs ---
    st.subheader("📈 Performance Metrics")
    makespan = max(r[3] for r in routes if r[1])
    total_travel_time = sum(solution.Value(routing.NextVar(i)) - solution.Value(time_dimension.CumulVar(i)) for i in range(routing.Size()))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Completion Time (Makespan)", f"{makespan/60:.1f} min")
    c2.metric("Total HT Travel Time", f"{total_travel_time/60:.1f} min")
    c3.metric("AGV Fleet Size", num_agvs)
    
    # --- Visualization ---
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    # Plot QC and Yard locations
    fig.add_trace(go.Scatter(
        x=[c[0] for c in qc_coords.values()], y=[c[1] for c in qc_coords.values()],
        mode='markers+text', name='Quay Cranes (QC)', text=list(qc_coords.keys()),
        marker=dict(color='blue', size=15, symbol='square'), textposition="bottom center"
    ))
    fig.add_trace(go.Scatter(
        x=[c[0] for c in yard_coords.values()], y=[c[1] for c in yard_coords.values()],
        mode='markers+text', name='Yard Blocks', text=list(yard_coords.keys()),
        marker=dict(color='green', size=15, symbol='circle'), textposition="bottom center"
    ))

    for agv, path, _, r_time in routes:
        if len(path) > 1:
            xs, ys = zip(*path)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                  name=f"AGV {agv+1} (end_time={r_time/60:.1f}m)",
                                  line=dict(color=colors[agv % len(colors)], width=2)))

    fig.update_layout(title="Optimised HT Routes & Schedule",
                      xaxis_title="X Position", yaxis_title="Y Position",
                      height=650, legend_title="Assets & Routes")
    st.plotly_chart(fig, use_container_width=True)

    # --- Validation ---
    st.subheader("📊 Constraint Validation")
    yard_counts_df = pd.DataFrame(list(assigned_yard_counts.items()), columns=['Yard Block', 'Assigned DI Jobs'])
    yard_counts_df['Capacity'] = yard_capacity_limit
    yard_counts_df['Breached'] = yard_counts_df['Assigned DI Jobs'] > yard_counts_df['Capacity']
    
    st.dataframe(yard_counts_df)
    if yard_counts_df['Breached'].any():
        st.error("⚠️ Hard Rule Violation: A yard block has exceeded the 700 DI job capacity.")
    else:
        st.success("✅ Hard Rule Passed: All yard block capacities were respected.")

else:
    st.error("❌ No solution found. Try increasing the time limit or reducing constraints.")
