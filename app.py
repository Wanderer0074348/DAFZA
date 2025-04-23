import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import os
from src.DroneDeliveryOptimizer import DroneDeliveryRouter

# Set page configuration
st.set_page_config(
    page_title="DCubed",
    page_icon="✈️",
    layout="wide"
)

# App title and description
st.title("DCubed")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Battery capacity slider
battery_capacity = st.sidebar.slider(
    "Battery Capacity (minutes)", 
    min_value=30, 
    max_value=120, 
    value=50,
    step=10
)

# File uploader for custom data (optional)
st.sidebar.header("Data Files (Optional)")
uploaded_locations = st.sidebar.file_uploader("Upload Locations CSV", type="csv")
uploaded_distances = st.sidebar.file_uploader("Upload Distance Matrix CSV", type="csv")

# Use default files if no files are uploaded
locations_file = 'data/delivery_locations.csv'
distance_matrix_file = 'data/distance_traffic_matrix.csv'

if uploaded_locations is not None:
    # Save the uploaded file temporarily
    with open("data/temp_locations.csv", "wb") as f:
        f.write(uploaded_locations.getbuffer())
    locations_file = "data/temp_locations.csv"

if uploaded_distances is not None:
    # Save the uploaded file temporarily
    with open("data/temp_distances.csv", "wb") as f:
        f.write(uploaded_distances.getbuffer())
    distance_matrix_file = "data/temp_distances.csv"

# Initialize the router
@st.cache_data
def initialize_router(locations_file, distance_matrix_file, battery_capacity):
    return DroneDeliveryRouter(
        locations_file=locations_file,
        distance_matrix_file=distance_matrix_file,
        battery_capacity_min=battery_capacity
    )

# Load the router with caching to improve performance
try:
    router = initialize_router(locations_file, distance_matrix_file, battery_capacity)
    
    # Display no-fly zones information
    st.header("No-Fly Zones")
    no_fly_zones_data = []
    for zone in router.no_fly_zones:
        no_fly_zones_data.append({
            "ID": zone['id'],
            "Center": f"({zone['center_lat']:.4f}, {zone['center_lon']:.4f})",
            "Radius": f"{zone['radius_km']:.2f} km"
        })
    st.table(pd.DataFrame(no_fly_zones_data))
    
    # Load location data for dropdowns
    locations_df = pd.read_csv(locations_file)
    depot_df = pd.DataFrame(router.depots.values())
    
    # Combine all location IDs for selection
    all_locations = list(locations_df['location_id'].unique())
    all_depots = list(depot_df['location_id'].unique())
    
    # Create two columns for sender and delivery location selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sender Location")
        sender_location = st.selectbox(
            "Select sender location",
            options=all_locations,
            index=0
        )
    
    with col2:
        st.subheader("Delivery Location")
        delivery_location = st.selectbox(
            "Select delivery location",
            options=all_locations + all_depots,
            index=min(1, len(all_locations) - 1)
        )
    
    # Button to calculate route
    if st.button("Calculate Optimal Route"):
        with st.spinner("Calculating optimal route..."):
            # Plan delivery route
            route = router.plan_delivery_route(sender_location, delivery_location)
            
            if "error" in route:
                st.error(f"Error: {route['error']}")
            else:
                # Display route details
                st.header("Route Details")
                
                # Create three columns for the different parts of the route
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("First Mile")
                    st.write(f"From: {route['first_mile']['from']}")
                    st.write(f"To: {route['first_mile']['to']}")
                    st.write(f"Distance: {route['first_mile']['distance_km']:.2f} km")
                    # st.write(f"Time: {route['first_mile']['time_min']:.1f} min")
                
                with col2:
                    st.subheader("Drone Route")
                    st.write(f"Path: {' → '.join(route['drone_route']['path'])}")
                    st.write(f"Time: {route['drone_route']['time_min']:.1f} min")
                
                with col3:
                    st.subheader("Last Mile")
                    st.write(f"From: {route['last_mile']['from']}")
                    st.write(f"To: {route['last_mile']['to']}")
                    st.write(f"Distance: {route['last_mile']['distance_km']:.2f} km")
                    # st.write(f"Time: {route['last_mile']['time_min']:.1f} min")
                
                # Display total time
                st.subheader("Total Delivery Time")
                st.write(f"{route['total_time_min']:.1f} minutes")
                
                # Visualize the route
                st.header("Route Visualization")
                
                # Generate and display the map using the streamlit-specific method
                folium_map = router.visualize_route_streamlit(route)
                folium_static(folium_map, width=1000, height=600)
                
                # Also save the map to a file for download
                map_file = "drone_delivery_route.html"
                router.visualize_route(route, output_file=map_file)
                
                # Provide download link for the map
                with open(map_file, "rb") as file:
                    btn = st.download_button(
                        label="Download Map as HTML",
                        data=file,
                        file_name="drone_delivery_route.html",
                        mime="text/html"
                    )
except Exception as e:
    st.error(f"Error initializing the router: {str(e)}")
    st.write("Please check that the data files are in the correct format and location.")

# Footer
st.markdown("---")
st.markdown("Drone Delivery Route Optimizer | Created with Streamlit")
