class FT_Reactor:
    def __init__(self, config):
        """Initializes the reactor with parameters from the YAML config."""
        self.T_C = config['reactor_conditions']['temperature_C']
        self.P_bar = config['reactor_conditions']['pressure_bar']
        self.flow_in = config['reactor_conditions']['inlet_flow_kmol_h']
        self.ghsv = config['reactor_conditions']['ghsv_h_1']

    def calculate_bed_volume(self):
        """Calculates the required catalyst volume based on GHSV."""
        R = 0.08314  # L*bar/(mol*K)
        T_K = self.T_C + 273.15
        
        # Volumetric flow rate in m3/h
        vol_flow_m3_h = (self.flow_in * 1000 * R * T_K) / (self.P_bar * 1000)
        
        # Required catalyst volume
        cat_volume = vol_flow_m3_h / self.ghsv
        return cat_volume