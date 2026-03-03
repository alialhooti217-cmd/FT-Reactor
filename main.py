# main.py
import yaml
from src.reactor import FT_Reactor

def run_simulation():
    # 1. Load the fully modular configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # 2. Initialize the dynamic reactor
    reactor = FT_Reactor(config)
    
    # 3. Perform calculations in sequence
    volume_total = reactor.calculate_bed_volume()
    N_parallel, Nt_per_reactor, Ds = reactor.calculate_geometry()
    delta_P = reactor.calculate_pressure_drop()
    
    # 4. Output the results
    print("========================================")
    print("      FT REACTOR ISOLATED SIMULATION    ")
    print("========================================")
    print(f"Total Catalyst Volume    : {volume_total:.2f} m³")
    print(f"Number of Reactors (N)   : {N_parallel}")
    print(f"Tubes per reactor (Nt)   : {Nt_per_reactor:,}")
    print(f"Shell Diameter (Ds)      : {Ds:.2f} m")
    print(f"L/D Ratio                : {reactor.L_over_D:.2f}")
    print(f"Pressure Drop (ΔP)       : {delta_P:.2f} bar")
    print("========================================")

if __name__ == "__main__":
    run_simulation()
