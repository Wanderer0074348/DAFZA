import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import folium
from folium.plugins import MarkerCluster
import random
from sklearn.cluster import KMeans
from streamlit_folium import folium_static
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Drone Delivery System Dashboard",
    page_icon="🚁",
    layout="wide"
)

# Title and introduction
st.title("Drone Delivery System Dashboard")
st.markdown("This dashboard visualizes the drone delivery system's routes, metrics, and performance.")

# Import your existing code (you can also import it as a module)
# For this example, I'll assume we're running the code directly

# Function to load data
@st.cache_data
def load_data():
    try:
        locations_df = pd.read_csv('delivery_locations.csv')
        distance_df = pd.read_csv('distance_traffic_matrix.csv')
        orders_df = pd.read_csv('delivery_orders.csv')
        return locations_df, distance_df, orders_df
    except FileNotFoundError:
        st.error("Data files not found. Please upload the required CSV files.")
        return None, None, None

# File uploader for data files if they don't exist
uploaded_locations = st.sidebar.file_uploader("Upload delivery_locations.csv", type="csv")
uploaded_distance = st.sidebar.file_uploader("Upload distance_traffic_matrix.csv", type="csv")
uploaded_orders = st.sidebar.file_uploader("Upload delivery_orders.csv", type="csv")

if uploaded_locations and uploaded_distance and uploaded_orders:
    locations_df = pd.read_csv(uploaded_locations)
    distance_df = pd.read_csv(uploaded_distance)
    orders_df = pd.read_csv(uploaded_orders)
else:
    # Try to load from files
    locations_df, distance_df, orders_df = load_data()
    
    if locations_df is None:
        st.warning("Please upload the required CSV files to continue.")
        st.stop()

# Define main warehouse (depot) location with a name
MAIN_DEPOT = {
    'depot_id': 'CENTRAL-HUB',
    'name': 'Central Operations Hub',
    'latitude': 25.15, 
    'longitude': 55.25,
    'capacity': 30
}

# Define no-fly zones as circular areas (center lat, center long, radius in km)
no_fly_zones = [
    {'lat': 25.12, 'lon': 55.25, 'radius': 0.8, 'name': 'Airport Zone'},
    {'lat': 25.18, 'lon': 55.28, 'radius': 0.6, 'name': 'Military Facility'},
    {'lat': 25.07, 'lon': 55.31, 'radius': 0.5, 'name': 'Government Building'},
    {'lat': 25.22, 'lon': 55.20, 'radius': 0.7, 'name': 'Residential High-Security Area'}
]

# Create secondary depots with human-readable names (strategically placed outside restricted zones)
secondary_depots = [
    {'depot_id': 'DEPOT-WEST', 'name': 'West District Hub', 'latitude': 25.15, 'longitude': 55.22, 'capacity': 15},
    {'depot_id': 'DEPOT-SOUTH', 'name': 'South Gateway', 'latitude': 25.10, 'longitude': 55.30, 'capacity': 20},
    {'depot_id': 'DEPOT-NORTH', 'name': 'North Command Center', 'latitude': 25.20, 'longitude': 55.33, 'capacity': 15},
    {'depot_id': 'DEPOT-EAST', 'name': 'East Distribution Point', 'latitude': 25.05, 'longitude': 55.25, 'capacity': 10},
    {'depot_id': 'DEPOT-CENTRAL', 'name': 'Central Auxiliary Base', 'latitude': 25.23, 'longitude': 55.28, 'capacity': 12}
]

# Include all your existing functions here (is_in_no_fly_zone, find_nearest_depot, etc.)
# For brevity, I'm not including all of them in this example

# Function to check if a location is within a no-fly zone
def is_in_no_fly_zone(lat, lon, zones):
    for zone in zones:
        # Calculate distance from point to zone center (in km)
        dist = distance.euclidean((lat, lon), (zone['lat'], zone['lon'])) * 111 # Rough conversion to km
        if dist <= zone['radius']:
            return True, zone['name']
    return False, None

# Function to find nearest depot for a location
def find_nearest_depot(lat, lon, depots_df, main_depot):
    min_dist = float('inf')
    nearest_depot = None
    depot_name = None

    # Check main depot
    main_dist = distance.euclidean((lat, lon), (main_depot['latitude'], main_depot['longitude'])) * 111
    if main_dist < min_dist:
        min_dist = main_dist
        nearest_depot = main_depot['depot_id']
        depot_name = main_depot['name']

    # Check secondary depots
    for _, depot in depots_df.iterrows():
        dist = distance.euclidean((lat, lon), (depot['latitude'], depot['longitude'])) * 111
        if dist < min_dist:
            min_dist = dist
            nearest_depot = depot['depot_id']
            depot_name = depot['name']

    return nearest_depot, depot_name, min_dist

# Include all other functions from your original code...
# For brevity, I'm assuming they're included

# Streamlit sidebar for configuration
st.sidebar.title("Configuration")
st.sidebar.header("Drone Specifications")

# Allow users to adjust drone specifications
max_range = st.sidebar.slider("Maximum Drone Range (km)", 10, 40, 20)
max_payload = st.sidebar.slider("Maximum Payload (kg)", 1, 10, 5)
avg_item_weight = st.sidebar.slider("Average Item Weight (kg)", 0.1, 2.0, 0.5)

# Update drone specifications
drone_specs = {
    'max_range': max_range,
    'base_range': max_range + 5,
    'max_payload': max_payload,
    'avg_item_weight': avg_item_weight,
    'weight_reduction_factor': 0.5,
    'base_energy_per_km': 0.1,
    'battery_capacity': 100
}

# Process data and generate routes
if st.sidebar.button("Generate Routes"):
    with st.spinner("Processing data and generating routes..."):
        # Process data (similar to your original code)
        depots_df = pd.DataFrame(secondary_depots)
        depots_df['in_no_fly_zone'], depots_df['zone_name'] = zip(*depots_df.apply(
            lambda row: is_in_no_fly_zone(row['latitude'], row['longitude'], no_fly_zones), axis=1))
        
        # Ensure no depot is in a restricted zone
        if depots_df['in_no_fly_zone'].any():
            st.warning("Some depots are in no-fly zones! Relocating...")
            # Move affected depots slightly outside the no-fly zone
            for idx, depot in depots_df[depots_df['in_no_fly_zone']].iterrows():
                # Find direction vector away from the zone center
                zone_name = depot['zone_name']
                zone = next(z for z in no_fly_zones if z['name'] == zone_name)
                
                # Calculate direction vector (away from zone center)
                direction_x = depot['latitude'] - zone['lat']
                direction_y = depot['longitude'] - zone['lon']
                
                # Normalize and multiply by zone radius + 0.1km safety margin
                magnitude = np.sqrt(direction_x**2 + direction_y**2)
                if magnitude > 0: # Avoid division by zero
                    direction_x = direction_x / magnitude
                    direction_y = direction_y / magnitude
                
                # Move depot outside the zone
                depots_df.at[idx, 'latitude'] = zone['lat'] + direction_x * (zone['radius'] + 0.1) / 111
                depots_df.at[idx, 'longitude'] = zone['lon'] + direction_y * (zone['radius'] + 0.1) / 111
                
                st.write(f"Relocated {depot['depot_id']} outside {zone_name}")
            
            # Re-check no-fly zone status
            depots_df['in_no_fly_zone'], depots_df['zone_name'] = zip(*depots_df.apply(
                lambda row: is_in_no_fly_zone(row['latitude'], row['longitude'], no_fly_zones), axis=1))
        
        # Add no-fly zone status to locations dataframe
        locations_df['in_no_fly_zone'], locations_df['zone_name'] = zip(*locations_df.apply(
            lambda row: is_in_no_fly_zone(row['latitude'], row['longitude'], no_fly_zones), axis=1))
        
        # Generate drone fleet
        drone_fleet_df = generate_drone_fleet(depots_df, MAIN_DEPOT)
        
        # Find nearest depot for each location
        locations_df['nearest_depot'], locations_df['depot_name'], locations_df['depot_distance'] = zip(*locations_df.apply(
            lambda row: find_nearest_depot(row['latitude'], row['longitude'], depots_df, MAIN_DEPOT), axis=1))
        
        # Create locations lookup dictionary
        locations_dict = {}
        for _, row in locations_df.iterrows():
            locations_dict[row['location_id']] = {
                'address': row['address'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'in_no_fly_zone': row['in_no_fly_zone'],
                'zone_name': row['zone_name'],
                'nearest_depot': row['nearest_depot'],
                'depot_name': row['depot_name'],
                'depot_distance': row['depot_distance']
            }
        
        # Add depot locations to locations_dict for routing
        for _, row in depots_df.iterrows():
            locations_dict[row['depot_id']] = {
                'address': row['name'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'in_no_fly_zone': False,
                'zone_name': None,
                'nearest_depot': None,
                'depot_name': None,
                'depot_distance': 0
            }
        
        # Also add the main depot
        locations_dict[MAIN_DEPOT['depot_id']] = {
            'address': MAIN_DEPOT['name'],
            'latitude': MAIN_DEPOT['latitude'],
            'longitude': MAIN_DEPOT['longitude'],
            'in_no_fly_zone': False,
            'zone_name': None,
            'nearest_depot': None,
            'depot_name': None,
            'depot_distance': 0
        }
        
        # Create distance matrix
        # (Include your distance matrix creation code here)
        
        # Process orders and generate routes
        order_data = orders_df.to_dict('records')
        optimize_results, updated_drone_fleet = batch_optimize_routes(
            orders_df, locations_dict, depots_df, distance_matrix, no_fly_zones, drone_specs, drone_fleet_df, MAIN_DEPOT
        )
        
        # Optimize route sequences
        sequence_optimized_routes = optimize_all_route_sequences(optimize_results, locations_dict, distance_matrix)
        
        # Generate detailed routes with coordinates
        detailed_routes = generate_detailed_routes(sequence_optimized_routes, locations_dict)
        
        # Update drone fleet status
        final_drone_fleet = update_drone_fleet_status(detailed_routes, updated_drone_fleet)
        
        # Calculate metrics
        route_metrics = calculate_route_metrics(detailed_routes, final_drone_fleet)
        
        # Store results in session state for access across reruns
        st.session_state.detailed_routes = detailed_routes
        st.session_state.final_drone_fleet = final_drone_fleet
        st.session_state.route_metrics = route_metrics
        st.session_state.locations_dict = locations_dict
        st.session_state.routes_generated = True
        
        st.success("Routes generated successfully!")

# Display results if routes are generated
if 'routes_generated' in st.session_state and st.session_state.routes_generated:
    # Get data from session state
    detailed_routes = st.session_state.detailed_routes
    final_drone_fleet = st.session_state.final_drone_fleet
    route_metrics = st.session_state.route_metrics
    locations_dict = st.session_state.locations_dict
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Route Map", "Detailed Routes", "Drone Fleet"])
    
    with tab1:
        st.header("Delivery System Dashboard")
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", route_metrics['total_orders'])
            st.metric("Assigned Orders", route_metrics['assigned_orders'])
        
        with col2:
            st.metric("Total Distance", f"{route_metrics['total_distance']:.2f} km")
            st.metric("Average Distance", f"{route_metrics['average_distance']:.2f} km")
        
        with col3:
            st.metric("Direct Deliveries", route_metrics['direct_deliveries'])
            st.metric("Multi-hop Deliveries", route_metrics['multi_hop_deliveries'])
        
        with col4:
            st.metric("Restricted Zone Deliveries", route_metrics['restricted_zone_deliveries'])
            st.metric("Drones Used", f"{route_metrics['drones_used']} of {route_metrics['drones_total']}")
        
        # Create a summary visualization of route types
        st.subheader("Distribution of Route Types")
        
        route_type_counts = {
            'direct': route_metrics['direct_deliveries'],
            'multi_hop': route_metrics['multi_hop_deliveries'],
            'depot_transfer': route_metrics['restricted_zone_deliveries'],
            'infeasible': route_metrics['infeasible_deliveries']
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(route_type_counts.keys(), route_type_counts.values(), color=['green', 'blue', 'orange', 'red'])
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        ax.set_title('Distribution of Route Types')
        ax.set_xlabel('Route Type')
        ax.set_ylabel('Count')
        
        st.pyplot(fig)
        
        # Depot utilization
        st.subheader("Depot Utilization")
        
        depot_names = []
        delivery_counts = []
        
        for depot, count in route_metrics['depot_utilization'].items():
            depot_name = next((d['name'] for d in secondary_depots if d['depot_id'] == depot), 
                              MAIN_DEPOT['name'] if depot == MAIN_DEPOT['depot_id'] else depot)
            depot_names.append(depot_name)
            delivery_counts.append(count)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars2 = ax2.bar(depot_names, delivery_counts, color='skyblue')
        
        # Add data labels on top of bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        ax2.set_title('Depot Utilization')
        ax2.set_xlabel('Depot')
        ax2.set_ylabel('Number of Deliveries')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig2)
    
    with tab2:
        st.header("Route Map")
        
        # Create a visual map of all routes
        route_map = create_route_map(detailed_routes, locations_dict, no_fly_zones, MAIN_DEPOT, depots_df)
        
        # Display the map
        folium_static(route_map, width=1000, height=600)
        
        # Add option to filter routes
        st.subheader("Filter Routes")
        
        route_types = ['All'] + list(set(route['route_type'] for route in detailed_routes))
        selected_type = st.selectbox("Select Route Type", route_types)
        
        if selected_type != 'All':
            filtered_routes = [r for r in detailed_routes if r['route_type'] == selected_type]
        else:
            filtered_routes = detailed_routes
        
        # Create filtered map
        filtered_map = create_route_map(filtered_routes, locations_dict, no_fly_zones, MAIN_DEPOT, depots_df)
        
        # Display filtered map
        st.subheader(f"Filtered Map: {selected_type} Routes")
        folium_static(filtered_map, width=1000, height=600)
    
    with tab3:
        st.header("Detailed Routes")
        
        # Convert routes to DataFrame for display
        routes_df = pd.DataFrame([
            {
                'Order ID': r['order_id'],
                'Destination': r['destination_address'],
                'Route Type': r['route_type'],
                'Distance (km)': round(r['total_distance'], 2),
                'Drone': r['drone_name'] if r['assigned_drone'] else 'None',
                'Restricted Zone': 'Yes' if r['is_restricted'] else 'No'
            }
            for r in detailed_routes
        ])
        
        # Add search and filter options
        search_term = st.text_input("Search by Order ID or Destination")
        
        if search_term:
            filtered_df = routes_df[
                routes_df['Order ID'].astype(str).str.contains(search_term, case=False) | 
                routes_df['Destination'].astype(str).str.contains(search_term, case=False)
            ]
        else:
            filtered_df = routes_df
        
        # Display the filtered DataFrame
        st.dataframe(filtered_df, use_container_width=True)
        
        # Route details
        st.subheader("Route Details")
        
        # Select a route to display details
        selected_order = st.selectbox(
            "Select Order ID to view details",
            options=[r['order_id'] for r in detailed_routes]
        )
        
        # Find the selected route
        selected_route = next((r for r in detailed_routes if r['order_id'] == selected_order), None)
        
        if selected_route:
            # Display route details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Order ID:** {selected_route['order_id']}")
                st.write(f"**Destination:** {selected_route['destination_address']}")
                st.write(f"**Route Type:** {selected_route['route_type']}")
                st.write(f"**Total Distance:** {selected_route['total_distance']:.2f} km")
                st.write(f"**Restricted Zone:** {'Yes' if selected_route['is_restricted'] else 'No'}")
            
            with col2:
                st.write(f"**Assigned Drone:** {selected_route['drone_name'] if selected_route['assigned_drone'] else 'None'}")
                st.write(f"**Home Depot:** {selected_route['depot_name'] if selected_route['home_depot'] else 'None'}")
                st.write(f"**Final Delivery:** {selected_route.get('final_delivery', 'N/A')}")
                
                if selected_route['is_restricted']:
                    st.write(f"**Ground Distance:** {selected_route.get('ground_distance', 0):.2f} km")
            
            # Display route path
            if selected_route['route_path']:
                st.subheader("Route Path")
                
                path_df = pd.DataFrame([
                    {
                        'Sequence': i+1,
                        'Stop ID': stop,
                        'Name': locations_dict[stop]['address'] if stop in locations_dict else stop,
                        'Type': 'Depot' if stop.startswith('DEPOT') or stop == MAIN_DEPOT['depot_id'] else 'Delivery Location'
                    }
                    for i, stop in enumerate(selected_route['route_path'])
                ])
                
                st.dataframe(path_df, use_container_width=True)
                
                # Create a map for this specific route
                specific_route_map = folium.Map(
                    location=[MAIN_DEPOT['latitude'], MAIN_DEPOT['longitude']],
                    zoom_start=12,
                    tiles='OpenStreetMap'
                )
                
                # Add route line
                route_points = []
                for stop_id in selected_route['route_path']:
                    if stop_id in locations_dict:
                        lat = locations_dict[stop_id]['latitude']
                        lon = locations_dict[stop_id]['longitude']
                        route_points.append([lat, lon])
                
                folium.PolyLine(
                    locations=route_points,
                    color='blue',
                    weight=3,
                    opacity=0.8,
                    tooltip=f"Route {selected_route['order_id']}"
                ).add_to(specific_route_map)
                
                # Add markers for each stop
                for i, stop_id in enumerate(selected_route['route_path']):
                    if stop_id in locations_dict:
                        lat = locations_dict[stop_id]['latitude']
                        lon = locations_dict[stop_id]['longitude']
                        
                        # Different icons for different types of stops
                        if stop_id == MAIN_DEPOT['depot_id']:
                            icon = folium.Icon(color='black', icon='home', prefix='fa')
                        elif stop_id.startswith('DEPOT'):
                            icon = folium.Icon(color='darkblue', icon='building', prefix='fa')
                        elif i == len(selected_route['route_path']) - 1:  # Destination
                            icon = folium.Icon(color='red', icon='flag-checkered', prefix='fa')
                        else:
                            icon = folium.Icon(color='blue', icon='stop-circle', prefix='fa')
                        
                        folium.Marker(
                            location=[lat, lon],
                            icon=icon,
                            tooltip=f"{i+1}. {locations_dict[stop_id]['address']}"
                        ).add_to(specific_route_map)
                
                # Display the map
                st.subheader(f"Map for Order {selected_route['order_id']}")
                folium_static(specific_route_map, width=800, height=500)
    
    with tab4:
        st.header("Drone Fleet Status")
        
        # Display drone fleet as a DataFrame
        st.dataframe(final_drone_fleet, use_container_width=True)
        
        # Drone status visualization
        st.subheader("Drone Status Distribution")
        
        status_counts = final_drone_fleet['status'].value_counts()
        
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99', '#ffcc99'])
        ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        st.pyplot(fig3)
        
        # Drone location distribution
        st.subheader("Drone Location Distribution")
        
        location_counts = final_drone_fleet['current_location'].value_counts()
        
        # Get readable names for locations
        location_names = []
        for loc in location_counts.index:
            if loc == MAIN_DEPOT['depot_id']:
                location_names.append(MAIN_DEPOT['name'])
            elif loc.startswith('DEPOT'):
                depot_name = next((d['name'] for d in secondary_depots if d['depot_id'] == loc), loc)
                location_names.append(depot_name)
            else:
                location_names.append(loc)
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        bars4 = ax4.bar(location_names, location_counts.values, color='lightgreen')
        
        # Add data labels on top of bars
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        ax4.set_title('Drone Location Distribution')
        ax4.set_xlabel('Location')
        ax4.set_ylabel('Number of Drones')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig4)

    # Add download buttons for CSV exports
    st.sidebar.header("Export Data")
    
    # Function to create a download link
    def get_csv_download_link(df, filename, text):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        return href
    
    # Create DataFrames for export
    routes_export_df = pd.DataFrame([
        {
            'order_id': r['order_id'],
            'destination': r['destination'],
            'destination_address': r['destination_address'],
            'is_restricted': r['is_restricted'],
            'route_type': r['route_type'],
            'total_distance': r['total_distance'],
            'assigned_drone': r['assigned_drone'],
            'drone_name': r['drone_name'],
            'home_depot': r['home_depot'],
            'depot_name': r['depot_name'],
            'route_path': ' -> '.join(r['route_path']) if r['route_path'] else 'No path'
        }
        for r in detailed_routes
    ])
    
    # Add download links
    st.sidebar.markdown(get_csv_download_link(routes_export_df, "optimized_routes.csv", "Download Routes CSV"), unsafe_allow_html=True)
    st.sidebar.markdown(get_csv_download_link(final_drone_fleet, "drone_fleet_status.csv", "Download Drone Fleet CSV"), unsafe_allow_html=True)

else:
    # Display instructions if routes are not yet generated
    st.info("Configure the drone specifications in the sidebar and click 'Generate Routes' to start.")
    
    # Display a sample map with depots and no-fly zones
    st.subheader("Sample Map with Depots and No-Fly Zones")
    
    sample_map = folium.Map(
        location=[MAIN_DEPOT['latitude'], MAIN_DEPOT['longitude']],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add no-fly zones
    for zone in no_fly_zones:
        folium.Circle(
            location=[zone['lat'], zone['lon']],
            radius=zone['radius'] * 1000,  # Convert km to m
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.2,
            tooltip=f"No-Fly Zone: {zone['name']}"
        ).add_to(sample_map)
    
    # Add the main depot
    folium.Marker(
        location=[MAIN_DEPOT['latitude'], MAIN_DEPOT['longitude']],
        icon=folium.Icon(color='black', icon='home', prefix='fa'),
        tooltip=f"Main Depot: {MAIN_DEPOT['name']}"
    ).add_to(sample_map)
    
    # Add secondary depots
    for depot in secondary_depots:
        folium.Marker(
            location=[depot['latitude'], depot['longitude']],
            icon=folium.Icon(color='darkblue', icon='building', prefix='fa'),
            tooltip=f"Depot: {depot['name']}"
        ).add_to(sample_map)
    
    folium_static(sample_map, width=1000, height=600)
