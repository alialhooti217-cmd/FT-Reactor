# main.py
import yaml
from src.reactor import FT_Reactor

def run_simulation():
    # 1. Load the fully modular configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # 2. Define FT Stream Data (From Colab Notebook)
    # Eventually, this could be pulled from an upstream mixing module
    FT_rec_heated = {
        "C1": 268.98, "C3": 45.71, "C6": 2.00, "C11": 0.00, "C18": 0.00,
        "CO": 198.07, "CO2": 189.32, "H2": 17282.51, "H2O": 1.15,
    }
    
    FT_in = {
        "C1": 27.06, "C3": 0.00, "C6": 0.73, "C11": 0.00, "C18": 0.00,
        "CO": 1421.12, "CO2": 203.34, "H2": 20920.52, "H2O": 47.54,
    }
    
    # Combine the streams into a single dictionary
    species = sorted(set(FT_rec_heated) | set(FT_in))
    FT_total_in = {s: FT_rec_heated.get(s, 0.0) + FT_in.get(s, 0.0) for s in species}
    
    # 3. Initialize the dynamic reactor with the feed composition
    reactor = FT_Reactor(config, feed_composition=FT_total_in)
    
    # 4. Perform calculations in sequence
    volume_total = reactor.calculate_bed_volume()
    N_parallel, Nt_per_reactor, Ds = reactor.calculate_geometry()
    delta_P = reactor.calculate_pressure_drop()
    
    # 5. Output the results
    print("========================================")
    print("      FT REACTOR ISOLATED SIMULATION    ")
    print("========================================")
    print(f"Total Inlet Flow         : {sum(FT_total_in.values()):,.2f} kmol/h")
    print(f"Catalyst Sizing Mode     : {reactor.Vcat_mode.upper()}")
    print(f"Total Catalyst Volume    : {volume_total:.2f} m³")
    print(f"Number of Reactors (N)   : {N_parallel}")
    print(f"Tubes per reactor (Nt)   : {Nt_per_reactor:,}")
    print(f"Tube Length Selected     : {reactor.L_tube_selected:.2f} m")
    print(f"Shell Diameter (Ds)      : {Ds:.2f} m")
    print(f"L/D Ratio                : {reactor.L_over_D:.2f}")
    print(f"Pressure Drop (ΔP)       : {delta_P:.3f} bar")
    print("========================================")

if __name__ == "__main__":
    run_simulation()
