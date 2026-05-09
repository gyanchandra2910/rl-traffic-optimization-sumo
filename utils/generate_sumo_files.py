"""
SUMO Network and Traffic Demand Generator

This script generates the necessary SUMO files for a 4-way intersection
traffic simulation:
- network.net.xml: The road network
- routes.rou.xml: Vehicle routes/traffic demand
- sumo_config.sumocfg: SUMO configuration file

Usage:
    python utils/generate_sumo_files.py
"""

import os
import subprocess
import sys
import random
import xml.etree.ElementTree as ET


def main():
    """Generate SUMO network, routes, and configuration files."""
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    data_dir = os.path.abspath(data_dir)
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"[INFO] Output directory: {data_dir}")
    
    # Check for SUMO_HOME environment variable
    if 'SUMO_HOME' not in os.environ:
        print("[ERROR] SUMO_HOME environment variable is not set.")
        print("Please set SUMO_HOME to your SUMO installation directory.")
        print("Example: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
        sys.exit(1)
    
    sumo_home = os.environ['SUMO_HOME']
    print(f"[INFO] SUMO_HOME: {sumo_home}")
    
    # Define output file paths
    net_file = os.path.join(data_dir, "network.net.xml")
    route_file = os.path.join(data_dir, "routes.rou.xml")
    config_file = os.path.join(data_dir, "sumo_config.sumocfg")
    trips_file = os.path.join(data_dir, "trips.trips.xml")
    vtypes_file = os.path.join(data_dir, "vtypes.add.xml")
    
    # =========================================================================
    # Step 1: Generate Network using netgenerate
    # =========================================================================
    print("\n" + "="*60)
    print("[STEP 1] Generating 4-way intersection network...")
    print("="*60)
    
    netgenerate_cmd = [
        "netgenerate",
        "--spider",                        # Generate spider/radial network (creates intersection)
        "--spider.arm-number", "4",        # 4 arms (4-way intersection)
        "--spider.circle-number", "1",     # Single ring = one intersection
        "--spider.space-radius", "200",    # Distance from center (200m)
        "--default.lanenumber", "2",       # 2 lanes per direction
        "--lefthand", "true",              # Indian-style left-hand traffic
        "--default-junction-type", "traffic_light",  # Add traffic lights
        "--tls.guess", "true",             # Guess traffic light locations
        "--output-file", net_file,         # Output network file
        "--no-turnarounds",                # Disable U-turns
    ]
    
    print(f"[CMD] {' '.join(netgenerate_cmd)}")
    
    result = subprocess.run(netgenerate_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] netgenerate failed:")
        print(result.stderr)
        sys.exit(1)
    
    print(f"[SUCCESS] Network generated: {net_file}")
    
    # =========================================================================
    # Step 2: Create Vehicle Types (Cars and Buses)
    # =========================================================================
    print("\n" + "="*60)
    print("[STEP 2] Creating vehicle types (cars and buses)...")
    print("="*60)
    
    vtypes_content = """<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">

    <!-- Car: Standard passenger vehicle with realistic physics -->
    <vType id="car" vClass="passenger" guiShape="passenger" length="5.0" accel="2.6" decel="4.5" maxSpeed="13.89" minGap="2.5" sigma="0.5" color="0,1,0"/>
    
    <!-- Bus: Larger, slower vehicle with realistic physics -->
    <vType id="bus" vClass="bus" guiShape="bus" length="12.0" accel="1.2" decel="3.0" maxSpeed="11.11" minGap="3.0" sigma="0.3" color="1,0,0"/>

</additional>
"""
    
    with open(vtypes_file, 'w') as f:
        f.write(vtypes_content)
    
    print(f"[SUCCESS] Vehicle types created: {vtypes_file}")
    
    # =========================================================================
    # Step 3: Generate Traffic Routes using randomTrips.py
    # =========================================================================
    print("\n" + "="*60)
    print("[STEP 2] Generating traffic routes...")
    print("="*60)
    
    # Find randomTrips.py in SUMO tools
    random_trips_path = os.path.join(sumo_home, "tools", "randomTrips.py")
    
    if not os.path.exists(random_trips_path):
        print(f"[ERROR] randomTrips.py not found at: {random_trips_path}")
        sys.exit(1)
    
    print(f"[INFO] Using randomTrips.py from: {random_trips_path}")
    
    # Generate random trips and routes
    random_trips_cmd = [
        sys.executable,                    # Use the current Python interpreter
        random_trips_path,
        "-n", net_file,                    # Network file
        "-r", route_file,                  # Output route file
        "-o", trips_file,                  # Output trips file
        "-e", "3600",                      # End time (1 hour = 3600 seconds)
        "-p", "2.0",                       # Spawn a vehicle every 2 seconds
        "--validate",                      # Validate routes
    ]
    
    print(f"[CMD] {' '.join(random_trips_cmd)}")
    
    result = subprocess.run(random_trips_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] randomTrips.py failed:")
        print(result.stderr)
        sys.exit(1)
    
    print(f"[SUCCESS] Routes generated: {route_file}")
    
    # =========================================================================
    # Step 4: Add Mixed Traffic (90% Cars, 10% Buses)
    # =========================================================================
    print("\n" + "="*60)
    print("[STEP 4] Adding mixed traffic (90% cars, 10% buses)...")
    print("="*60)
    
    # Parse the routes file and assign vehicle types
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    car_count = 0
    bus_count = 0
    
    for vehicle in root.findall('vehicle'):
        # Randomly assign type: 90% car, 10% bus
        if random.random() < 0.1:
            vehicle.set('type', 'bus')
            bus_count += 1
        else:
            vehicle.set('type', 'car')
            car_count += 1
    
    # Save modified routes file
    tree.write(route_file, encoding='UTF-8', xml_declaration=True)
    
    print(f"[SUCCESS] Vehicle types assigned: {car_count} cars, {bus_count} buses")
    
    # =========================================================================
    # Step 5: Create SUMO Configuration File
    # =========================================================================
    print("\n" + "="*60)
    print("[STEP 5] Creating SUMO configuration file...")
    print("="*60)
    
    # Create viewsettings.xml for "real world" GUI scheme
    viewsettings_file = os.path.join(data_dir, "viewsettings.xml")
    viewsettings_content = '<viewsettings><scheme name="real world"/></viewsettings>'
    
    with open(viewsettings_file, 'w') as f:
        f.write(viewsettings_content)
    
    print(f"[SUCCESS] View settings created: {viewsettings_file}")
    
    sumo_config_content = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="vtypes.add.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>

    <gui_only>
        <gui-settings-file value="viewsettings.xml"/>
    </gui_only>

</configuration>
"""
    
    with open(config_file, 'w') as f:
        f.write(sumo_config_content)
    
    print(f"[SUCCESS] Configuration file created: {config_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("[COMPLETE] All SUMO files generated successfully!")
    print("="*60)
    print(f"\nGenerated files in {data_dir}:")
    print(f"  - network.net.xml    : Road network (4-way intersection)")
    print(f"  - routes.rou.xml     : Vehicle routes (mixed: cars + buses)")
    print(f"  - trips.trips.xml    : Trip definitions")
    print(f"  - vtypes.add.xml     : Vehicle type definitions")
    print(f"  - viewsettings.xml   : GUI view settings (real world scheme)")
    print(f"  - sumo_config.sumocfg: SUMO configuration")
    print("\nTo test the simulation, run:")
    print(f'  sumo-gui -c "{config_file}"')
    print()


if __name__ == "__main__":
    main()
