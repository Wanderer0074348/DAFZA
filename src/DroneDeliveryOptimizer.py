import pandas as pd
import numpy as np
import folium
from folium.plugins import AntPath
import math
from collections import defaultdict
import heapq
import random
from shapely.geometry import Point, LineString

class DroneDeliveryRouter:
    """
    A class to optimize drone delivery routes through a network of depots,
    considering battery constraints and no-fly zones.
    """
    
    def __init__(self, locations_file, distance_matrix_file, battery_capacity_min=50):
        """
        Initialize the router with data files, depot locations, and constraints.
        
        Args:
            locations_file: Path to CSV file containing location data
            distance_matrix_file: Path to CSV file containing distance/time matrix
            battery_capacity_min: Battery capacity in minutes of flight time
        """
        # Load datasets
        self.locations = pd.read_csv(locations_file)
        self.dist_matrix = pd.read_csv(distance_matrix_file)
        
        # Set battery capacity
        self.battery_capacity_min = battery_capacity_min
        
        # Define depot locations
        self.depots = {
            # Existing depots
            'DEPOT1': {'location_id': 'DEPOT1', 'address': 'North Depot', 'latitude': 25.22, 'longitude': 55.28},
            'DEPOT2': {'location_id': 'DEPOT2', 'address': 'South Depot', 'latitude': 25.04, 'longitude': 55.24},
            'DEPOT3': {'location_id': 'DEPOT3', 'address': 'East Depot', 'latitude': 25.14, 'longitude': 55.36},
            # 'DEPOT4': {'location_id': 'DEPOT4', 'address': 'West Depot', 'latitude': 25.13, 'longitude': 55.16},
            'DEPOT5': {'location_id': 'DEPOT5', 'address': 'Central Depot', 'latitude': 25.15, 'longitude': 55.25},
            
            # New depots
            # 'DEPOT6': {'location_id': 'DEPOT6', 'address': 'Northeast Depot', 'latitude': 25.19, 'longitude': 55.33},
            'DEPOT7': {'location_id': 'DEPOT7', 'address': 'Northwest Depot', 'latitude': 25.20, 'longitude': 55.19},
            'DEPOT8': {'location_id': 'DEPOT8', 'address': 'Southeast Depot', 'latitude': 25.08, 'longitude': 55.32},
            'DEPOT9': {'location_id': 'DEPOT9', 'address': 'Southwest Depot', 'latitude': 25.07, 'longitude': 55.18},
            'DEPOT10': {'location_id': 'DEPOT10', 'address': 'Far East Depot', 'latitude': 25.11, 'longitude': 55.37},
            
            # Additional depots for better coverage
            'DEPOT11': {'location_id': 'DEPOT11', 'address': 'North Central Depot', 'latitude': 25.18, 'longitude': 55.25},
            'DEPOT12': {'location_id': 'DEPOT12', 'address': 'South Central Depot', 'latitude': 25.10, 'longitude': 55.25},
            'DEPOT13': {'location_id': 'DEPOT13', 'address': 'East Central Depot', 'latitude': 25.15, 'longitude': 55.30},
            'DEPOT14': {'location_id': 'DEPOT14', 'address': 'West Central Depot', 'latitude': 25.15, 'longitude': 55.20},
            # 'DEPOT15': {'location_id': 'DEPOT15', 'address': 'Far North Depot', 'latitude': 25.24, 'longitude': 55.25}
        }
        
        # Define fixed no-fly zones instead of random ones
        self.no_fly_zones = [
            {
                'id': 'NFZ1',
                'center_lat': 25.17,
                'center_lon': 55.27,
                'radius_km': 2.0
            },
            {
                'id': 'NFZ2',
                'center_lat': 25.21,
                'center_lon': 55.32,
                'radius_km': 1.5
            },
            {
                'id': 'NFZ3',
                'center_lat': 25.10,
                'center_lon': 55.22,
                'radius_km': 1.8
            }
        ]
        
        # Add depots to locations dataframe
        depot_df = pd.DataFrame(self.depots.values())
        self.locations = pd.concat([self.locations, depot_df], ignore_index=True)
        
        # Build the depot network graph
        self.depot_graph = self._build_depot_graph()
        
        # Build the complete graph (including locations and depots)
        self.complete_graph = self._build_complete_graph()
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points in kilometers."""
        R = 6371  # Earth radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _path_intersects_no_fly_zone(self, lat1, lon1, lat2, lon2):
        """
        Check if a direct path between two points intersects any no-fly zone.
        
        Args:
            lat1, lon1: Coordinates of the first point
            lat2, lon2: Coordinates of the second point
            
        Returns:
            True if the path intersects any no-fly zone, False otherwise
        """
        # Create a LineString representing the path
        path = LineString([(lon1, lat1), (lon2, lat2)])
        
        for zone in self.no_fly_zones:
            # Create a Point representing the center of the no-fly zone
            center = Point(zone['center_lon'], zone['center_lat'])
            
            # Calculate the distance from the path to the center
            distance_km = path.distance(center) * 111  # Convert degrees to km (approximate)
            
            # If the distance is less than the radius, the path intersects the no-fly zone
            if distance_km < zone['radius_km']:
                return True
        
        return False
    
    def _build_depot_graph(self):
        """Build a graph of connections between depots with distances."""
        graph = defaultdict(dict)
        
        # Connect all depots to each other
        depot_ids = list(self.depots.keys())
        for i, depot1 in enumerate(depot_ids):
            for depot2 in depot_ids[i+1:]:
                d1 = self.depots[depot1]
                d2 = self.depots[depot2]
                
                # Calculate distance between depots
                distance = self._haversine_distance(
                    d1['latitude'], d1['longitude'],
                    d2['latitude'], d2['longitude']
                )
                
                # Estimate time (assuming 30 km/h drone speed)
                time_min = (distance / 30) * 60
                
                # Check if the path intersects any no-fly zone
                intersects_no_fly_zone = self._path_intersects_no_fly_zone(
                    d1['latitude'], d1['longitude'],
                    d2['latitude'], d2['longitude']
                )
                
                # Only add the connection if it doesn't intersect a no-fly zone
                # and the time is within the battery capacity limit
                if not intersects_no_fly_zone and time_min <= self.battery_capacity_min * 0.8:  # 20% safety margin
                    # Add bidirectional connections
                    graph[depot1][depot2] = time_min
                    graph[depot2][depot1] = time_min
        
        return graph
    
    def _build_complete_graph(self):
        """Build a complete graph including all locations and depots."""
        graph = defaultdict(dict)
        
        # Add connections from the distance matrix
        for _, row in self.dist_matrix.iterrows():
            from_id = row['from_location_id']
            to_id = row['to_location_id']
            time = row['base_time_min']
            
            # Check if the path intersects any no-fly zone
            from_loc = self.locations[self.locations['location_id'] == from_id]
            to_loc = self.locations[self.locations['location_id'] == to_id]
            
            if not from_loc.empty and not to_loc.empty:
                intersects_no_fly_zone = self._path_intersects_no_fly_zone(
                    from_loc.iloc[0]['latitude'], from_loc.iloc[0]['longitude'],
                    to_loc.iloc[0]['latitude'], to_loc.iloc[0]['longitude']
                )
                
                if not intersects_no_fly_zone:
                    graph[from_id][to_id] = time
        
        # Add connections to/from depots
        for depot_id, depot in self.depots.items():
            for _, loc in self.locations.iterrows():
                loc_id = loc['location_id']
                
                # Skip if it's a depot or already in the graph
                if loc_id in self.depots or loc_id == depot_id:
                    continue
                
                # Calculate distance and time
                distance = self._haversine_distance(
                    depot['latitude'], depot['longitude'],
                    loc['latitude'], loc['longitude']
                )
                time_min = (distance / 30) * 60
                
                # Check if the path intersects any no-fly zone
                intersects_no_fly_zone = self._path_intersects_no_fly_zone(
                    depot['latitude'], depot['longitude'],
                    loc['latitude'], loc['longitude']
                )
                
                if not intersects_no_fly_zone and time_min <= self.battery_capacity_min * 0.4:  # Only connect if within 40% of battery capacity
                    # Add bidirectional connections
                    graph[depot_id][loc_id] = time_min
                    graph[loc_id][depot_id] = time_min
        
        # Add depot-to-depot connections from depot_graph
        for depot1 in self.depot_graph:
            for depot2, time in self.depot_graph[depot1].items():
                graph[depot1][depot2] = time
        
        return graph
    
    def find_nearest_depot(self, location_id):
        """
        Find the nearest depot to a given location.
        
        Args:
            location_id: ID of the location
            
        Returns:
            Tuple of (nearest_depot_id, distance_in_minutes)
        """
        if location_id in self.depots:
            return location_id, 0  # It's already a depot
        
        location = self.locations[self.locations['location_id'] == location_id]
        if location.empty:
            return None, float('infinity')
        
        lat = location.iloc[0]['latitude']
        lon = location.iloc[0]['longitude']
        
        nearest_depot = None
        min_distance = float('infinity')
        
        for depot_id, depot in self.depots.items():
            # Check if there's a direct path in the graph
            if location_id in self.complete_graph and depot_id in self.complete_graph[location_id]:
                time = self.complete_graph[location_id][depot_id]
                if time < min_distance:
                    min_distance = time
                    nearest_depot = depot_id
            else:
                # Calculate direct distance if not in graph
                distance = self._haversine_distance(
                    lat, lon, depot['latitude'], depot['longitude']
                )
                time = (distance / 30) * 60  # Assuming 30 km/h speed
                
                # Check if the path intersects any no-fly zone
                intersects_no_fly_zone = self._path_intersects_no_fly_zone(
                    lat, lon, depot['latitude'], depot['longitude']
                )
                
                if not intersects_no_fly_zone and time < min_distance:
                    min_distance = time
                    nearest_depot = depot_id
        
        return nearest_depot, min_distance
    
    def find_shortest_path_between_depots(self, start_depot, end_depot):
        """
        Find the shortest path between two depots using Dijkstra's algorithm.
        
        Args:
            start_depot: Starting depot ID
            end_depot: Ending depot ID
            
        Returns:
            Tuple of (path, total_time)
        """
        if start_depot == end_depot:
            return [start_depot], 0
        
        # Initialize distances with infinity
        distances = {depot: float('infinity') for depot in self.depots}
        distances[start_depot] = 0
        
        # Initialize priority queue and previous nodes
        priority_queue = [(0, start_depot)]
        previous_nodes = {depot: None for depot in self.depots}
        visited = set()
        
        while priority_queue:
            current_distance, current_depot = heapq.heappop(priority_queue)
            
            # If we've reached our target, we can stop
            if current_depot == end_depot:
                break
                
            # Skip if we've already processed this node
            if current_depot in visited:
                continue
                
            visited.add(current_depot)
            
            # Check all neighbors of the current depot
            for neighbor, time in self.depot_graph[current_depot].items():
                if neighbor in visited:
                    continue
                    
                distance = current_distance + time
                
                # If we found a better path, update it
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_depot
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # Reconstruct the path
        path = []
        current_depot = end_depot
        
        # If we couldn't reach the end depot, return None
        if previous_nodes[end_depot] is None:
            return None, float('infinity')
        
        while current_depot:
            path.append(current_depot)
            current_depot = previous_nodes[current_depot]
            
        path.reverse()
        
        return path, distances[end_depot]
    
    def plan_delivery_route(self, sender_location, delivery_location):
        """
        Plan a delivery route from sender to recipient through the depot network,
        considering battery constraints and no-fly zones.
        
        Args:
            sender_location: ID of the sender location
            delivery_location: ID of the delivery location
            
        Returns:
            Dictionary with route information
        """
        # Find nearest depots to sender and delivery locations
        sender_nearest_depot, sender_distance = self.find_nearest_depot(sender_location)
        delivery_nearest_depot, delivery_distance = self.find_nearest_depot(delivery_location)
        
        if sender_nearest_depot is None or delivery_nearest_depot is None:
            return {"error": "Could not find a valid path to a depot from either sender or delivery location"}
        
        # Calculate first and last mile times
        sender_time = sender_distance
        delivery_time = delivery_distance
        
        # Find the shortest path between the depots
        depot_path, depot_time = self.find_shortest_path_between_depots(
            sender_nearest_depot, delivery_nearest_depot
        )
        
        # If no path was found between depots, try alternative depots
        if depot_path is None:
            # Try finding alternative depots
            alternative_sender_depots = []
            alternative_delivery_depots = []
            
            # Find alternative depots for sender
            for depot_id, depot in self.depots.items():
                if depot_id != sender_nearest_depot:
                    loc = self.locations[self.locations['location_id'] == sender_location].iloc[0]
                    distance = self._haversine_distance(
                        loc['latitude'], loc['longitude'],
                        depot['latitude'], depot['longitude']
                    )
                    time = (distance / 30) * 60
                    
                    if not self._path_intersects_no_fly_zone(
                        loc['latitude'], loc['longitude'],
                        depot['latitude'], depot['longitude']
                    ):
                        alternative_sender_depots.append((depot_id, time))
            
            # Find alternative depots for delivery
            for depot_id, depot in self.depots.items():
                if depot_id != delivery_nearest_depot:
                    loc = self.locations[self.locations['location_id'] == delivery_location].iloc[0]
                    distance = self._haversine_distance(
                        loc['latitude'], loc['longitude'],
                        depot['latitude'], depot['longitude']
                    )
                    time = (distance / 30) * 60
                    
                    if not self._path_intersects_no_fly_zone(
                        loc['latitude'], loc['longitude'],
                        depot['latitude'], depot['longitude']
                    ):
                        alternative_delivery_depots.append((depot_id, time))
            
            # Sort alternatives by time
            alternative_sender_depots.sort(key=lambda x: x[1])
            alternative_delivery_depots.sort(key=lambda x: x[1])
            
            # Try combinations of alternative depots
            for alt_sender_depot, alt_sender_time in alternative_sender_depots[:3]:  # Try top 3 alternatives
                for alt_delivery_depot, alt_delivery_time in alternative_delivery_depots[:3]:  # Try top 3 alternatives
                    alt_depot_path, alt_depot_time = self.find_shortest_path_between_depots(
                        alt_sender_depot, alt_delivery_depot
                    )
                    
                    if alt_depot_path is not None:
                        sender_nearest_depot = alt_sender_depot
                        delivery_nearest_depot = alt_delivery_depot
                        sender_time = alt_sender_time
                        delivery_time = alt_delivery_time
                        depot_path = alt_depot_path
                        depot_time = alt_depot_time
                        break
                
                if depot_path is not None:
                    break
            
            # If still no path found, return error
            if depot_path is None:
                return {"error": "Cannot find a valid path between depots"}
        
        # Apply battery constraints to the depot path
        battery_constrained_path = []
        current_battery = self.battery_capacity_min
        total_time = 0
        
        # Start with the first depot
        battery_constrained_path.append(depot_path[0])
        
        for i in range(1, len(depot_path)):
            from_depot = depot_path[i-1]
            to_depot = depot_path[i]
            
            # Get the time between these depots
            segment_time = self.depot_graph[from_depot][to_depot]
            
            # Check if we have enough battery
            if segment_time <= current_battery:
                # We can make it to the next depot
                battery_constrained_path.append(to_depot)
                current_battery -= segment_time
                total_time += segment_time
            else:
                # Need to find an intermediate depot to recharge
                intermediate_path = self._find_intermediate_depot(from_depot, to_depot, current_battery)
                
                if intermediate_path:
                    # Add the intermediate depots to the path
                    for j in range(1, len(intermediate_path)):
                        battery_constrained_path.append(intermediate_path[j])
                        
                        # Calculate time for this segment
                        int_segment_time = self.depot_graph[intermediate_path[j-1]][intermediate_path[j]]
                        total_time += int_segment_time
                        
                        # Recharge at each intermediate depot
                        current_battery = self.battery_capacity_min
                else:
                    # Cannot find a path with current battery constraints
                    return {"error": "Cannot find a valid path with current battery constraints"}
        
        # Construct the complete route
        complete_route = {
            'first_mile': {
                'from': sender_location,
                'to': sender_nearest_depot,
                'distance_km': sender_distance * 30 / 60,  # Convert time to distance
                'time_min': sender_time
            },
            'drone_route': {
                'path': battery_constrained_path,
                'time_min': total_time
            },
            'last_mile': {
                'from': delivery_nearest_depot,
                'to': delivery_location,
                'distance_km': delivery_distance * 30 / 60,  # Convert time to distance
                'time_min': delivery_time
            },
            'total_time_min': sender_time + total_time + delivery_time,
            'battery_capacity_min': self.battery_capacity_min,
            'no_fly_zones': self.no_fly_zones
        }
        
        return complete_route
    
    def _find_intermediate_depot(self, from_depot, to_depot, current_battery):
        """
        Find a path through intermediate depots when direct path exceeds battery capacity.
        
        Args:
            from_depot: Starting depot ID
            to_depot: Target depot ID
            current_battery: Current battery level in minutes
            
        Returns:
            List of depot IDs forming a path, or None if no path is found
        """
        # Initialize distances with infinity
        distances = {depot: float('infinity') for depot in self.depots}
        distances[from_depot] = 0
        
        # Initialize priority queue and previous nodes
        priority_queue = [(0, from_depot, current_battery)]
        previous_nodes = {depot: None for depot in self.depots}
        max_battery = {depot: 0 for depot in self.depots}
        max_battery[from_depot] = current_battery
        visited = set()
        
        while priority_queue:
            current_distance, current_depot, battery_left = heapq.heappop(priority_queue)
            
            # If we've reached our target, we can stop
            if current_depot == to_depot:
                break
                
            # Skip if we've already processed this node with better battery
            if current_depot in visited and battery_left <= max_battery[current_depot]:
                continue
                
            visited.add(current_depot)
            max_battery[current_depot] = max(max_battery[current_depot], battery_left)
            
            # Check all neighbors of the current depot
            for neighbor, time in self.depot_graph[current_depot].items():
                # Skip if we don't have enough battery to reach this neighbor
                if time > battery_left:
                    continue
                
                new_distance = current_distance + time
                new_battery = battery_left - time
                
                # If we found a better path, update it
                if new_distance < distances[neighbor] or new_battery > max_battery[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_depot
                    max_battery[neighbor] = new_battery
                    heapq.heappush(priority_queue, (new_distance, neighbor, new_battery))
        
        # If we couldn't reach the target depot, return None
        if previous_nodes[to_depot] is None:
            return None
        
        # Reconstruct the path
        path = []
        current_depot = to_depot
        
        while current_depot:
            path.append(current_depot)
            current_depot = previous_nodes[current_depot]
            
        path.reverse()
        
        return path
    
    def visualize_route(self, route, output_file='drone_delivery_route.html'):
        """
        Visualize the delivery route on a map, showing all depots, no-fly zones,
        and the complete route.
        
        Args:
            route: Route dictionary from plan_delivery_route
            output_file: Path to save the HTML map
            
        Returns:
            Path to the saved HTML file
        """
        if "error" in route:
            print(f"Error: {route['error']}")
            return None
        
        # Extract locations for the route
        sender_loc = self.locations[self.locations['location_id'] == route['first_mile']['from']]
        delivery_loc = self.locations[self.locations['location_id'] == route['last_mile']['to']]
        
        # Calculate map center
        all_lats = [sender_loc.iloc[0]['latitude']]
        all_lons = [sender_loc.iloc[0]['longitude']]
        
        # Include all depots in center calculation
        for depot_id, depot in self.depots.items():
            all_lats.append(depot['latitude'])
            all_lons.append(depot['longitude'])
        
        all_lats.append(delivery_loc.iloc[0]['latitude'])
        all_lons.append(delivery_loc.iloc[0]['longitude'])
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add sender marker
        folium.Marker(
            [sender_loc.iloc[0]['latitude'], sender_loc.iloc[0]['longitude']],
            popup=f"Sender: {route['first_mile']['from']} - {sender_loc.iloc[0]['address']}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Add delivery location marker
        folium.Marker(
            [delivery_loc.iloc[0]['latitude'], delivery_loc.iloc[0]['longitude']],
            popup=f"Delivery: {route['last_mile']['to']} - {delivery_loc.iloc[0]['address']}",
            icon=folium.Icon(color='red', icon='flag')
        ).add_to(m)
        
        # Add ALL depot markers (including unused ones)
        for depot_id, depot in self.depots.items():
            # Check if this depot is used in the route
            if depot_id in route['drone_route']['path']:
                # Get the index in the route path
                route_index = route['drone_route']['path'].index(depot_id)
                
                # Different icon for first and last depot in the drone route
                if route_index == 0:
                    icon = folium.Icon(color='blue', icon='upload')
                    popup_text = f"Pickup Depot: {depot_id}"
                elif route_index == len(route['drone_route']['path']) - 1:
                    icon = folium.Icon(color='orange', icon='download')
                    popup_text = f"Delivery Depot: {depot_id}"
                else:
                    icon = folium.Icon(color='purple', icon='exchange')
                    popup_text = f"Transit Depot: {depot_id}"
            else:
                # Unused depot - show with a different style
                icon = folium.Icon(color='gray', icon='home')
                popup_text = f"Unused Depot: {depot_id}"
            
            folium.Marker(
                [depot['latitude'], depot['longitude']],
                popup=popup_text,
                icon=icon
            ).add_to(m)
        
        # Add no-fly zones
        for zone in route['no_fly_zones']:
            folium.Circle(
                location=[zone['center_lat'], zone['center_lon']],
                radius=zone['radius_km'] * 1000,  # Convert km to meters
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.2,
                popup=f"No-Fly Zone: {zone['id']}"
            ).add_to(m)
        
        # Add lines between depots in the route
        for i in range(1, len(route['drone_route']['path'])):
            current_depot_id = route['drone_route']['path'][i]
            prev_depot_id = route['drone_route']['path'][i-1]
            
            current_depot = self.depots[current_depot_id]
            prev_depot = self.depots[prev_depot_id]
            
            # Use AntPath for drone routes (animated dashed line)
            AntPath(
                [[prev_depot['latitude'], prev_depot['longitude']], 
                 [current_depot['latitude'], current_depot['longitude']]],
                color='blue',
                weight=3,
                dash_array=[10, 20],
                delay=1000,
                popup=f"Drone flight: {prev_depot_id} to {current_depot_id}"
            ).add_to(m)
        
        # Add first mile line
        first_depot = self.depots[route['drone_route']['path'][0]]
        folium.PolyLine(
            [[sender_loc.iloc[0]['latitude'], sender_loc.iloc[0]['longitude']], 
             [first_depot['latitude'], first_depot['longitude']]],
            color='green',
            weight=2,
            opacity=0.7,
            popup=f"First mile: {route['first_mile']['time_min']:.1f} min"
        ).add_to(m)
        
        # Add last mile line
        last_depot = self.depots[route['drone_route']['path'][-1]]
        folium.PolyLine(
            [[last_depot['latitude'], last_depot['longitude']], 
             [delivery_loc.iloc[0]['latitude'], delivery_loc.iloc[0]['longitude']]],
            color='red',
            weight=2,
            opacity=0.7,
            popup=f"Last mile: {route['last_mile']['time_min']:.1f} min"
        ).add_to(m)
        
        # Add a legend to explain the markers
        legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 180px; height: 200px; 
            border:2px solid grey; z-index:9999; font-size:12px;
            background-color:white; padding: 10px;
            border-radius: 5px;
            ">
            <p><b>Legend</b></p>
            <p><i class="fa fa-play" style="color:green"></i> Sender</p>
            <p><i class="fa fa-flag" style="color:red"></i> Delivery Location</p>
            <p><i class="fa fa-upload" style="color:blue"></i> Pickup Depot</p>
            <p><i class="fa fa-download" style="color:orange"></i> Delivery Depot</p>
            <p><i class="fa fa-exchange" style="color:purple"></i> Transit Depot</p>
            <p><i class="fa fa-home" style="color:gray"></i> Unused Depot</p>
            <p><i class="fa fa-ban" style="color:red"></i> No-Fly Zone</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(output_file)
        return output_file
    
    def visualize_route_streamlit(self, route):
        """
        Visualize the delivery route on a map for Streamlit display.
        
        Args:
            route: Route dictionary from plan_delivery_route
            
        Returns:
            folium.Map object that can be displayed in Streamlit
        """
        if "error" in route:
            print(f"Error: {route['error']}")
            return None
        
        # Extract locations for the route
        sender_loc = self.locations[self.locations['location_id'] == route['first_mile']['from']]
        delivery_loc = self.locations[self.locations['location_id'] == route['last_mile']['to']]
        
        # Calculate map center
        all_lats = [sender_loc.iloc[0]['latitude']]
        all_lons = [sender_loc.iloc[0]['longitude']]
        
        # Include all depots in center calculation
        for depot_id, depot in self.depots.items():
            all_lats.append(depot['latitude'])
            all_lons.append(depot['longitude'])
        
        all_lats.append(delivery_loc.iloc[0]['latitude'])
        all_lons.append(delivery_loc.iloc[0]['longitude'])
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add sender marker
        folium.Marker(
            [sender_loc.iloc[0]['latitude'], sender_loc.iloc[0]['longitude']],
            popup=f"Sender: {route['first_mile']['from']} - {sender_loc.iloc[0]['address']}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Add delivery location marker
        folium.Marker(
            [delivery_loc.iloc[0]['latitude'], delivery_loc.iloc[0]['longitude']],
            popup=f"Delivery: {route['last_mile']['to']} - {delivery_loc.iloc[0]['address']}",
            icon=folium.Icon(color='red', icon='flag')
        ).add_to(m)
        
        # Add ALL depot markers (including unused ones)
        for depot_id, depot in self.depots.items():
            # Check if this depot is used in the route
            if depot_id in route['drone_route']['path']:
                # Get the index in the route path
                route_index = route['drone_route']['path'].index(depot_id)
                
                # Different icon for first and last depot in the drone route
                if route_index == 0:
                    icon = folium.Icon(color='blue', icon='upload')
                    popup_text = f"Pickup Depot: {depot_id}"
                elif route_index == len(route['drone_route']['path']) - 1:
                    icon = folium.Icon(color='orange', icon='download')
                    popup_text = f"Delivery Depot: {depot_id}"
                else:
                    icon = folium.Icon(color='purple', icon='exchange')
                    popup_text = f"Transit Depot: {depot_id}"
            else:
                # Unused depot - show with a different style
                icon = folium.Icon(color='gray', icon='home')
                popup_text = f"Unused Depot: {depot_id}"
            
            folium.Marker(
                [depot['latitude'], depot['longitude']],
                popup=popup_text,
                icon=icon
            ).add_to(m)
        
        # Add no-fly zones
        for zone in route['no_fly_zones']:
            folium.Circle(
                location=[zone['center_lat'], zone['center_lon']],
                radius=zone['radius_km'] * 1000,  # Convert km to meters
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.2,
                popup=f"No-Fly Zone: {zone['id']}"
            ).add_to(m)
        
        # Add lines between depots in the route
        for i in range(1, len(route['drone_route']['path'])):
            current_depot_id = route['drone_route']['path'][i]
            prev_depot_id = route['drone_route']['path'][i-1]
            
            current_depot = self.depots[current_depot_id]
            prev_depot = self.depots[prev_depot_id]
            
            # Use AntPath for drone routes (animated dashed line)
            AntPath(
                [[prev_depot['latitude'], prev_depot['longitude']], 
                 [current_depot['latitude'], current_depot['longitude']]],
                color='blue',
                weight=3,
                dash_array=[10, 20],
                delay=1000,
                popup=f"Drone flight: {prev_depot_id} to {current_depot_id}"
            ).add_to(m)
        
        # Add first mile line
        first_depot = self.depots[route['drone_route']['path'][0]]
        folium.PolyLine(
            [[sender_loc.iloc[0]['latitude'], sender_loc.iloc[0]['longitude']], 
             [first_depot['latitude'], first_depot['longitude']]],
            color='green',
            weight=2,
            opacity=0.7,
            popup=f"First mile: {route['first_mile']['time_min']:.1f} min"
        ).add_to(m)
        
        # Add last mile line
        last_depot = self.depots[route['drone_route']['path'][-1]]
        folium.PolyLine(
            [[last_depot['latitude'], last_depot['longitude']], 
             [delivery_loc.iloc[0]['latitude'], delivery_loc.iloc[0]['longitude']]],
            color='red',
            weight=2,
            opacity=0.7,
            popup=f"Last mile: {route['last_mile']['time_min']:.1f} min"
        ).add_to(m)
        
        # Add a legend to explain the markers
        legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 180px; height: 200px; 
            border:2px solid grey; z-index:9999; font-size:12px;
            background-color:white; padding: 10px;
            border-radius: 5px;
            ">
            <p><b>Legend</b></p>
            <p><i class="fa fa-play" style="color:green"></i> Sender</p>
            <p><i class="fa fa-flag" style="color:red"></i> Delivery Location</p>
            <p><i class="fa fa-upload" style="color:blue"></i> Pickup Depot</p>
            <p><i class="fa fa-download" style="color:orange"></i> Delivery Depot</p>
            <p><i class="fa fa-exchange" style="color:purple"></i> Transit Depot</p>
            <p><i class="fa fa-home" style="color:gray"></i> Unused Depot</p>
            <p><i class="fa fa-ban" style="color:red"></i> No-Fly Zone</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Return the map object for Streamlit display
        return m
