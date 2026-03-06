# src/reactor.py
import math

class FT_Reactor:
    def __init__(self, config, feed_composition=None):
        # 1. Pull dynamic decision variables
        self.T_C = config['decision_variables']['temperature_C']
        self.P_bar = config['decision_variables']['pressure_bar']
        self.h2_co_ratio = config['decision_variables'].get('h2_co_ratio', 2.0) 
        
        # 2. Pull fixed parameters
        self.flow_in = config['fixed_parameters']['inlet_flow_kmol_h']
        self.ghsv = config['fixed_parameters']['ghsv_h_1']
        self.eps = config['fixed_parameters']['catalyst_void_fraction']
        self.dp = config['fixed_parameters']['catalyst_dp_m']
        
        # Tube dimensions (Input in inches -> Convert to meters)
        self.Di_inch = config['fixed_parameters']['tube_di_inch']
        self.Do_inch = config['fixed_parameters']['tube_do_inch']
        self.Di = self.Di_inch * 0.0254
        self.Do = self.Do_inch * 0.0254
        
        # 3. Pull Constraints & Search Limits (Defaults provided if not in config yet)
        self.max_Ds = config['constraints']['max_shell_diameter_m']
        self.LD_min = config['constraints'].get('LD_min', 3.0)
        self.LD_max = config['constraints'].get('LD_max', 5.0)
        self.L_tube_min = config['constraints'].get('L_tube_min', 4.0)
        self.L_tube_max = config['constraints'].get('L_tube_max', 16.0)
        self.L_tube_step = config['constraints'].get('L_tube_step', 0.5)
        self.N_max_search = config['constraints'].get('N_max_search', 12)
        
        # 4. Pull Kinetics Parameters (Defaults match your paper)
        # We will add these to config.yaml in the next step
        self.k_rate = config.get('kinetics', {}).get('k_rate', 0.0339)
        self.a_ads = config.get('kinetics', {}).get('a_ads', 1.185)
        self.rho_cat_kg_m3 = config.get('kinetics', {}).get('rho_cat_kg_m3', 7840.0)
        self.Vcat_mode = config.get('kinetics', {}).get('Vcat_mode', 'max')
        self.target_X_CO = config['decision_variables'].get('single_pass_conversion', 0.7702)

        # Feed composition (Fallback to simple ratios if feed_dict isn't passed yet)
        self.feed_composition = feed_composition or {}
        
        # 5. Set static thermodynamic constants
        self.R_Pa = 8.314e3 # Pa·m3/(kmol·K)
        self.mu_Pa_s = 2.0e-5  
        self.Kt = 0.215
        self.n_exp = 2.207
        self.clearance_m = 0.097
        
        # Initialize internal state variables
        self.rho = self.calculate_gas_density()
        self.rCO_kmol_m3_s = self.calculate_kinetic_rate()

    def calculate_gas_density(self):
        """Calculates gas density. Uses exact mixture if feed provided, else uses H2/CO ratio."""
        T_K = self.T_C + 273.15
        
        if self.feed_composition:
            # Full mixture calculation from Colab
            MW_dict = {"H2": 2.016, "CO": 28.01, "CO2": 44.01, "H2O": 18.015,
                       "C1": 16.043, "C3": 44.11, "C6": 86.1779, "C11": 156.31, "C18": 254.49}
            Ftot = sum(self.feed_composition.values())
            MW_mix = sum((F / Ftot) * MW_dict.get(sp, 0.0) for sp, F in self.feed_composition.items()) if Ftot > 0 else 0.0
        else:
            # Fallback simple calculation
            y_H2 = self.h2_co_ratio / (self.h2_co_ratio + 1)
            y_CO = 1 / (self.h2_co_ratio + 1)
            MW_mix = (y_H2 * 2.016) + (y_CO * 28.01)
            
        P_Pa = self.P_bar * 1e5
        self.rho = (P_Pa * MW_mix) / (self.R_Pa * T_K)
        return self.rho

    def calculate_kinetic_rate(self):
        """Calculates volumetric CO consumption rate (kmol/m3_cat/s) based on inlet partial pressures."""
        if not self.feed_composition:
            # Approximation if feed dict isn't provided
            xCO = 1 / (self.h2_co_ratio + 1)
            xH2 = self.h2_co_ratio / (self.h2_co_ratio + 1)
        else:
            Ftot = sum(self.feed_composition.values())
            xCO = self.feed_composition.get("CO", 0.0) / Ftot
            xH2 = self.feed_composition.get("H2", 0.0) / Ftot

        Ptot_MPa = self.P_bar * 0.1
        PCO = xCO * Ptot_MPa
        PH2 = xH2 * Ptot_MPa

        # Rate law: r_CO = k * P_H2^2 * P_CO / (1 + a*P_CO*P_H2^2)
        denom = (1.0 + self.a_ads * PCO * (PH2 ** 2))
        rCO_mol_kg_s_inlet = self.k_rate * (PH2 ** 2) * PCO / denom
        
        # Convert to volumetric rate (kmol/m3/s)
        self.rCO_kmol_m3_s = max((rCO_mol_kg_s_inlet * self.rho_cat_kg_m3) / 1000.0, 1e-9)
        return self.rCO_kmol_m3_s

    def calculate_bed_volume(self):
        """Calculates catalyst volume required based on chosen mode (GHSV, Kinetics, or MAX)."""
        T_K = self.T_C + 273.15
        
        # 1. Volume via GHSV
        vol_flow_m3_h = (self.flow_in * self.R_Pa * T_K) / (self.P_bar * 1e5)
        self.Vcat_ghsv = vol_flow_m3_h / self.ghsv
        
        # 2. Volume via Kinetics (Vcat = FCO0 * X_CO / rCO_vol)
        if self.feed_composition:
            FCO_in_kmol_h = self.feed_composition.get("CO", 0.0)
        else:
            FCO_in_kmol_h = self.flow_in * (1 / (self.h2_co_ratio + 1))
            
        FCO0_kmol_s = FCO_in_kmol_h / 3600.0
        self.Vcat_kin = (FCO0_kmol_s * self.target_X_CO) / self.rCO_kmol_m3_s

        # 3. Determine used volume
        mode = self.Vcat_mode.lower()
        if mode == "kinetics":
            self.cat_volume_total = self.Vcat_kin
        elif mode == "ghsv":
            self.cat_volume_total = self.Vcat_ghsv
        else:
            self.cat_volume_total = max(self.Vcat_ghsv, self.Vcat_kin)
            
        return self.cat_volume_total

    def calculate_geometry(self):
        """Searches for minimum N and shortest L_tube meeting Ds and L/D constraints."""
        best_compromise = None
        
        # Helper inner function for calculating tubes and diameters
        def evaluate_design(N, L):
            Vcat_per = self.cat_volume_total / N
            Vreact_per = Vcat_per / (1.0 - self.eps)
            
            Vtube = (math.pi * self.Di**2 / 4.0) * L
            Nt = max(1, int(math.ceil(Vreact_per / Vtube)))
            
            Db = self.Do * ((Nt / self.Kt) ** (1.0 / self.n_exp))
            Ds = Db + self.clearance_m
            Lshell = L * 1.2
            LD = Lshell / Ds if Ds > 0 else float("inf")
            
            return Nt, Ds, LD

        # Grid Search
        for N in range(1, self.N_max_search + 1):
            
            # Generate L_tube search array
            steps = int(math.floor((self.L_tube_max - self.L_tube_min) / self.L_tube_step + 1e-12))
            L_range = [self.L_tube_min + i * self.L_tube_step for i in range(steps + 1)]
            
            for L in L_range:
                Nt, Ds, LD = evaluate_design(N, L)
                
                ok_D = (Ds <= self.max_Ds)
                ok_LD = (self.LD_min <= LD <= self.LD_max)
                
                # Penalty function
                pen_D = (Ds - self.max_Ds)**2 if Ds > self.max_Ds else 0.0
                pen_LD = (self.LD_min - LD)**2 if LD < self.LD_min else ((LD - self.LD_max)**2 if LD > self.LD_max else 0.0)
                total_penalty = pen_D + pen_LD
                
                cand = {'N': N, 'L_tube': L, 'Nt': Nt, 'Ds': Ds, 'L_over_D': LD, 'penalty': total_penalty}
                
                if ok_D and ok_LD:
                    # First valid solution found (since we loop N ascending, this gives min N)
                    self._apply_geometry(cand)
                    return self.N_parallel, self.Nt_per_reactor, self.Ds
                
                if best_compromise is None or total_penalty < best_compromise['penalty']:
                    best_compromise = cand
                    
        # If no strict solution was found, apply the best compromise
        self._apply_geometry(best_compromise)
        return self.N_parallel, self.Nt_per_reactor, self.Ds

    def _apply_geometry(self, cand):
        """Saves chosen geometric candidate variables to class instance."""
        self.N_parallel = cand['N']
        self.L_tube_selected = cand['L_tube']
        self.Nt_per_reactor = cand['Nt']
        self.Ds = cand['Ds']
        self.L_over_D = cand['L_over_D']

    def calculate_pressure_drop(self):
        """Calculates Ergun pressure drop per reactor."""
        T_K = self.T_C + 273.15
        vol_flow_m3_s_total = (self.flow_in * self.R_Pa * T_K) / (self.P_bar * 1e5) / 3600.0
        vol_flow_m3_s_per_reactor = vol_flow_m3_s_total / self.N_parallel
        
        cross_area_per_reactor = self.Nt_per_reactor * math.pi * (self.Di / 2)**2
        u = vol_flow_m3_s_per_reactor / cross_area_per_reactor
        
        term1 = (150 * self.mu_Pa_s * (1 - self.eps)**2) / (self.dp**2 * self.eps**3) * u
        term2 = (1.75 * self.rho * (1 - self.eps)) / (self.dp * self.eps**3) * u**2
        
        self.delta_P_bar = ((term1 + term2) * self.L_tube_selected) / 1e5
        
        return self.delta_P_bar
