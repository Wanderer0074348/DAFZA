from src.DroneDeliveryOptimizer import DroneDeliveryRouter

def main():
    # Initialize the router with increased battery capacity
    router = DroneDeliveryRouter(
        locations_file='delivery_locations.csv',
        distance_matrix_file='distance_traffic_matrix.csv',
        battery_capacity_min=50
    )
    
    # Print information about the no-fly zones
    print("No-Fly Zones:")
    for zone in router.no_fly_zones:
        print(f"  {zone['id']}: Center ({zone['center_lat']:.4f}, {zone['center_lon']:.4f}), Radius: {zone['radius_km']:.2f} km")
    
    # Plan a delivery route
    sender_location = 'LOC012'  # Example sender location
    delivery_location = 'LOC024'  # Example delivery location
    
    print(f"\nPlanning delivery route from {sender_location} to {delivery_location}...")
    route = router.plan_delivery_route(sender_location, delivery_location)
    
    if "error" in route:
        print(f"Error: {route['error']}")
    else:
        # Print route details
        print(f"\nDelivery from {sender_location} to {delivery_location}:")
        print(f"First mile: {route['first_mile']['from']} → {route['first_mile']['to']} ({route['first_mile']['time_min']:.1f} min)")
        print(f"Drone route: {' → '.join(route['drone_route']['path'])} ({route['drone_route']['time_min']:.1f} min)")
        print(f"Last mile: {route['last_mile']['from']} → {route['last_mile']['to']} ({route['last_mile']['time_min']:.1f} min)")
        print(f"Total delivery time: {route['total_time_min']:.1f} minutes")
        print(f"Battery capacity: {route['battery_capacity_min']} minutes")
        
        # Visualize the route
        map_file = router.visualize_route(route)
        print(f"\nRoute map saved to: {map_file}")


if __name__ == "__main__":
    main()
