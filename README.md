# Drone Delivery System

## Project Overview

This project implements an advanced drone delivery route optimization system that plans efficient delivery paths while accounting for various constraints such as no-fly zones, battery/range limitations, and multi-depot operations. The system calculates optimal routes by combining first-mile delivery, drone flight paths between depots, and last-mile delivery to reach the final destination.

## Key Features

- **Multi-depot Operations**: Routes packages through a network of strategically placed depots
- **No-fly Zone Avoidance**: Automatically detects and routes around restricted airspace
- **Battery Constraint Management**: Plans routes respecting drone battery capacity limits
- **Visualization**: Interactive maps showing delivery routes, depots, and no-fly zones
- **Streamlit Web Interface**: User-friendly interface for route planning and visualization

## System Architecture

The system consists of several key components:

1. **Route Optimization Engine**: Core algorithm that finds optimal paths through depots
2. **Depot Network**: Strategic placement of depots to maximize coverage
3. **Visualization Tools**: Interactive maps showing routes and constraints
4. **Web Interface**: Streamlit application for easy interaction

## Installation

### Prerequisites

- Python 3.8 or higher
- uv package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Wanderer0074348/DAFZA
   cd drone-delivery-system
   ```

2. **Create and activate a virtual environment using uv**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies using uv**:
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage
### Streamlit Web Interface

For a more interactive experience, run the Streamlit application:

```bash
uv run streamlit run app.py
```

This will start a local web server (typically at http://localhost:8501) where you can:

1. Adjust battery capacity constraints
2. Upload custom location and distance data
3. Select specific sender and delivery locations
4. Calculate and visualize optimal routes
5. Download the generated map as an HTML file

## Project Structure

```
drone-delivery-system/
├── app.py                  # Streamlit web interface
├── main.py                 # Command line interface
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
└── src/
    └── DroneDeliveryOptimizer.py  # Core route optimization logic
```

## Configuration Options

The route optimizer is configured with the following parameters:

- **Battery Capacity**: Flight time in minutes (default: 50)
- **No-fly Zones**: Restricted areas defined by center coordinates and radius
- **Depot Locations**: Random placement of depots for routing and delivery management

These parameters can be adjusted in the web interface(some) or by modifying the initialization in `main.py`.

## Dependencies

All python dependencies listed under requirements.txt.

