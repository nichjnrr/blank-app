import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="PortPilot.AI Optimiser", layout="wide")
st.title("🚢 PortPilot.AI – HT Job Scheduling & Routing Optimisation")


def generate_job_data():
    """
    Generates a realistic job dataset based on the port layout and requirements.
    This creates a CSV file that the main app will use.
    """
    NUM_JOBS = 200 # Using a smaller number for faster web demo
    NUM_QCS = 8
    NUM_YARDS = 16 # A1, A2, B1, B2 ... H2

    # Define locations from the map (approximated from the GUI screenshot)
    # QC locations (x, y)
    qc_coords = {f'QC{i+1}': (3 + i*5, 3) for i in range(NUM_QCS)}
    
    # Yard locations (x, y) - A1, A2, B1, B2, etc.
    yard_ids = [f'{chr(65+i)}{j}' for i in range(NUM_QCS) for j in [1, 2]]
    yard_coords = {
        yard_ids[k]: (2 + (k//2)*5 + (k%2)*2, 13) for k in range(NUM_YARDS)
    }

    jobs = []
    for i in range(NUM_JOBS):
        job_type = np.random.choice(['DI', 'LO'])
        qc = f'QC{np.random.randint(1, NUM_QCS + 1)}'
        
        if job_type == 'DI': # Discharge: QC -> Yard
            pickup_loc = qc_coords[qc]
            default_yard = np.random.choice(list(yard_coords.keys()))
            drop_loc = yard_coords[default_yard]
            
            # Select 3 unique alternative yards
            alt_yards = np.random.choice(
                [y for y in yard_ids if y != default_yard], 3, replace=False
            ).tolist()

        else: # Load: Yard -> QC
            default_yard = np.random.choice(list(yard_coords.keys()))
            pickup_loc = yard_coords[default_yard]
            drop_loc = qc_coords[qc]
            alt_yards = [None, None, None] # No alternatives for LO jobs

        jobs.append({
            'JOB_ID': i,
            'JOB_TYPE': job_type,
            'QC_M': qc,
            'YARD_BLOCK': default_yard,
            'pickup_x': pickup_loc[0],
            'pickup_y': pickup_loc[1],
            'drop_x': drop_loc[0],
            'drop_y': drop_loc[1],
            'ALT_YARD_BLOCK_1': alt_yards[0],
            'ALT_YARD_BLOCK_2': alt_yards[1],
            'ALT_YARD_BLOCK_3': alt_yards[2],
            'handling_time_pickup': 120 if job_type == 'LO' else 300, # QC time for DI, Yard for LO
            'handling_time_dropoff': 300 if job_type == 'DI' else 120, # Yard time for DI, QC for LO
        })
        
    df = pd.DataFrame(jobs)
    df.to_csv("job_dataset.csv", index=False)
    return df, qc_coords, yard_coords

import os

# --- 0. Ensure job dataset exists ---
if not os.path.exists("job_dataset.csv"):
    st.info("Generating initial dataset...")
    _, _, _ = generate_job_data()
    st.success("✅ job_dataset.csv generated successfully!")

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
ht_speed = st.sidebar.slider("HT Speed (secs/sector)", 5, 20, 10)


if st.sidebar.button("🔁 Regenerate Job Dataset"):
    jobs, qc_coords, yard_coords = generate_job_data()
    all_locations = {**qc_coords, **yard_coords}
    st.success("✅ Dataset regenerated! Click 'Run Optimisation' to optimise routes.")


st.sidebar.info(f"""
**Objective:** Complete all jobs in the shortest possible time (minimise makespan).
- **HT Fleet:** {num_agvs}
- **HT Speed:** {ht_speed} secs/sector
- **QC Handling:** 2 mins (120s)
- **Yard Handling:** 5 mins (300s)
""")

run_optimisation = st.sidebar.button("🚀 Run Optimisation")

if run_optimisation:
    with st.spinner("⚙️ Optimising HT Routes... Please wait"):

        # -----------------------------
        # 4a. Build Time Matrix
        # -----------------------------
        location_names = ['Depot'] + list(all_locations.keys())
        location_coords = {'Depot': (0,8)}
        for name, coords in all_locations.items():
            location_coords[name] = coords
        coord_array = np.array([location_coords[name] for name in location_names])

        def calculate_travel_time(from_node, to_node):
            start = coord_array[from_node]
            end = coord_array[to_node]
            distance = abs(start[0]-end[0]) + abs(start[1]-end[1])
            if 5 <= start[1] <= 9 and end[0] < start[0]:
                distance *= 1.5
            return int(distance * ht_speed)

        n_locations = len(coord_array)
        time_matrix = [[calculate_travel_time(i,j) for j in range(n_locations)] for i in range(n_locations)]
        location_to_idx = {name:i for i,name in enumerate(location_names)}
        depot_idx = 0

        # -----------------------------
        # 4b. OR-Tools VRP Setup
        # -----------------------------
        manager = pywrapcp.RoutingIndexManager(n_locations, num_agvs, depot_idx)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return time_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        handling_times = {depot_idx:0}
        for _, job in jobs.iterrows():
            pickup_node = job['QC_M'] if job['JOB_TYPE']=='DI' else job['YARD_BLOCK']
            drop_node = job['YARD_BLOCK'] if job['JOB_TYPE']=='DI' else job['QC_M']
            handling_times[location_to_idx[pickup_node]] = job['handling_time_pickup']
            handling_times[location_to_idx[drop_node]] = job['handling_time_dropoff']

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return handling_times.get(from_node,0)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimension(
            demand_callback_index,
            0,
            86400,
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # -----------------------------
        # 4c. Define Pickups & Deliveries
        # -----------------------------
        for _, job in jobs.iterrows():
            if job['JOB_TYPE'] == 'DI':
                pickup_node = location_to_idx[job['QC_M']]
                possible_dropoffs = [location_to_idx[job['YARD_BLOCK']]]
                for alt in ['ALT_YARD_BLOCK_1','ALT_YARD_BLOCK_2','ALT_YARD_BLOCK_3']:
                    if pd.notna(job[alt]):
                        possible_dropoffs.append(location_to_idx[job[alt]])
                pickup_index = manager.NodeToIndex(pickup_node)
                dropoff_indices = [manager.NodeToIndex(n) for n in possible_dropoffs]
                routing.AddDisjunction(dropoff_indices, 1)
                for d_idx in dropoff_indices:
                    routing.AddPickupAndDelivery(pickup_index, d_idx)
            else:
                pickup_node = location_to_idx[job['YARD_BLOCK']]
                dropoff_node = location_to_idx[job['QC_M']]
                routing.AddPickupAndDelivery(manager.NodeToIndex(pickup_node), manager.NodeToIndex(dropoff_node))

        time_dimension.SetGlobalSpanCostCoefficient(100)

        # -----------------------------
        # 4d. Solve VRP
        # -----------------------------
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        search_parameters.time_limit.seconds = 15
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

        solution = routing.SolveWithParameters(search_parameters)

        # -----------------------------
        # 4e. Extract & Visualise Solution
        # -----------------------------
        if solution:
            st.success("✅ Optimisation complete! Found a solution.")

            routes = []
            assigned_yard_counts = {name:0 for name in yard_coords}

            for vehicle_id in range(num_agvs):
                index = routing.Start(vehicle_id)
                path_nodes = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    path_nodes.append(location_names[node_index])
                    index = solution.Value(routing.NextVar(index))
                end_time_var = time_dimension.CumulVar(routing.End(vehicle_id))
                path_coords = [location_coords[name] for name in path_nodes]
                routes.append((vehicle_id, path_coords, path_nodes, solution.Value(end_time_var)))

            # Metrics
            makespan = max(r[3] for r in routes if r[1])
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Completion Time (Makespan)", f"{makespan/60:.1f} min")
            c2.metric("AGV Fleet Size", num_agvs)
            c3.metric("HT Speed", f"{ht_speed} secs/sector")

            # Plot
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            fig.add_trace(go.Scatter(x=[c[0] for c in qc_coords.values()], y=[c[1] for c in qc_coords.values()],
                                     mode='markers+text', text=list(qc_coords.keys()), name='QC',
                                     marker=dict(color='blue', size=15, symbol='square'), textposition="bottom center"))
            fig.add_trace(go.Scatter(x=[c[0] for c in yard_coords.values()], y=[c[1] for c in yard_coords.values()],
                                     mode='markers+text', text=list(yard_coords.keys()), name='Yard',
                                     marker=dict(color='green', size=15, symbol='circle'), textposition="bottom center"))

            for agv, path, _, r_time in routes:
                if len(path) > 1:
                    xs, ys = zip(*path)
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers',
                                             name=f'AGV {agv+1} (end={r_time/60:.1f}m)',
                                             line=dict(color=colors[agv%len(colors)], width=2)))
            fig.update_layout(title="Optimised HT Routes", xaxis_title="X", yaxis_title="Y", height=650)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ No solution found. Try increasing the time limit or reducing constraints.")
