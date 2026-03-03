# src/reactor.py
import math

class FT_Reactor:
    def __init__(self, config):
        # 1. Pull dynamic decision variables
        self.T_C = config['decision_variables']['temperature_C']
        self.P_bar = config['decision_variables']['pressure_bar']
        self.h2_co_ratio = config['decision_variables']['h2_co_ratio'] 
        
        # 2. Pull fixed parameters
        self.flow_in = config['fixed_parameters']['inlet_flow_kmol_h']
        self.ghsv = config['fixed_parameters']['ghsv_h_1']
        self.eps = config['fixed_parameters']['catalyst_void_fraction']
        self.dp = config['fixed_parameters']['catalyst_dp_m']
        self.L_tube = config['fixed_parameters']['target_l_tube_m']
        
        # 3. Tube dimensions (Input in inches -> Convert to meters automatically)
        self.Di_inch = config['fixed_parameters']['tube_di_inch']
        self.Do_inch = config['fixed_parameters']['tube_do_inch']
        self.Di = self.Di_inch * 0.0254
        self.Do = self.Do_inch * 0.0254
        
        # 4. Pull Constraints
        self.max_Ds = config['constraints']['max_shell_diameter_m']
        
        # 5. Set static thermodynamic constants & calculate density
        self.R = 0.08314
        self.mu_Pa_s = 2.0e-5  
        self.calculate_gas_density() 

    def calculate_gas_density(self):
        """Calculates dynamic syngas inlet density (kg/m³) using the Ideal Gas Law."""
        # Molar masses (kg/kmol)
        MW_H2 = 2.016
        MW_CO = 28.01
        
        # Mole fractions based on H2/CO ratio
        y_H2 = self.h2_co_ratio / (self.h2_co_ratio + 1)
        y_CO = 1 / (self.h2_co_ratio + 1)
        
        # Average mixture molecular weight
        MW_mix = (y_H2 * MW_H2) + (y_CO * MW_CO)
        
        # Ideal Gas Law: rho = (P * MW) / (R * T)
        T_K = self.T_C + 273.15
        self.rho = (self.P_bar * MW_mix) / (self.R * T_K)
        
        return self.rho

    def calculate_bed_volume(self):
        """Calculates total required catalyst volume for the entire plant."""
        T_K = self.T_C + 273.15
        vol_flow_m3_h = (self.flow_in * 1000 * self.R * T_K) / (self.P_bar * 1000)
        self.cat_volume_total = vol_flow_m3_h / self.ghsv
        return self.cat_volume_total

    def calculate_geometry(self):
        """Dynamically finds minimum parallel reactors to meet the 4.0m diameter limit."""
        self.N_parallel = 1
        
        while True:
            # Volume per reactor
            Vcat_per = self.cat_volume_total / self.N_parallel
            Vreact_per = Vcat_per / (1.0 - self.eps)
            
            # Tubes per reactor
            A_tube = math.pi * (self.Di / 2)**2
            V_tube = A_tube * self.L_tube
            self.Nt_per_reactor = math.ceil(Vreact_per / V_tube)
            
            # Shell diameter calculation (using Colab constants)
            Kt = 0.215
            n_exp = 2.207
            clearance_m = 0.097
            
            Db = self.Do * ((self.Nt_per_reactor / Kt) ** (1.0 / n_exp))
            self.Ds = Db + clearance_m
            
            # Check if this geometry is valid against our 4.0m constraint
            if self.Ds <= self.max_Ds:
                break
                
            # If the shell is too big, add another parallel reactor and loop again!
            self.N_parallel += 1 
            
        self.Lshell = self.L_tube * 1.2
        self.L_over_D = self.Lshell / self.Ds
        
        return self.N_parallel, self.Nt_per_reactor, self.Ds

    def calculate_pressure_drop(self):
        """Calculates Ergun pressure drop per reactor based on divided flow."""
        T_K = self.T_C + 273.15
        vol_flow_m3_s_total = ((self.flow_in * 1000 * self.R * T_K) / (self.P_bar * 1000)) / 3600
        
        # Divide flow among parallel reactors
        vol_flow_m3_s_per_reactor = vol_flow_m3_s_total / self.N_parallel
        
        # Velocity per reactor
        cross_area_per_reactor = self.Nt_per_reactor * math.pi * (self.Di / 2)**2
        u = vol_flow_m3_s_per_reactor / cross_area_per_reactor
        
        # Ergun Equation components
        term1 = (150 * self.mu_Pa_s * (1 - self.eps)**2) / (self.dp**2 * self.eps**3) * u
        term2 = (1.75 * self.rho * (1 - self.eps)) / (self.dp * self.eps**3) * u**2
        
        self.delta_P_bar = ((term1 + term2) * self.L_tube) / 100000
        
        return self.delta_P_bar
