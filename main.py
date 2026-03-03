import yaml

from src.reactor import FT_Reactor

def run_simulation():
    # 1. Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # 2. Initialize the reactor
    reactor = FT_Reactor(config)
    
    # 3. Perform calculations
    volume = reactor.calculate_bed_volume()
    
    # 4. Output the results
    print("========================================")
    print("      FT REACTOR ISOLATED SIMULATION    ")
    print("========================================")
    print(f"Required Catalyst Volume : {volume:.2f} m³")
    print("========================================")

if __name__ == "__main__":
    run_simulation()