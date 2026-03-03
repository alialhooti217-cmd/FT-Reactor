def calculate_asf_distribution(alpha, conversion):
    """
    Calculate the Modified ASF distribution of hydrocarbon products.
    
    Input: 
        alpha (float): Chain growth probability (0 < alpha < 1)
        conversion (float): Single-pass conversion (0 < conversion < 1)
    
    Output: 
        Dictionary of molar fractions for C1, C3, C6, C11, C18
    """
    # ASF distribution: w_n = (1-alpha) * alpha^(n-1)
    # where n is the carbon number
    
    carbon_numbers = {'C1': 1, 'C3': 3, 'C6': 6, 'C11': 11, 'C18': 18}
    
    # Calculate theoretical molar fractions based on ASF equation
    asf_fractions = {}
    for product, n in carbon_numbers.items():
        asf_fractions[product] = (1 - alpha) * (alpha ** (n - 1))
    
    # Normalize to sum to 1
    total = sum(asf_fractions.values())
    normalized_fractions = {k: v / total for k, v in asf_fractions.items()}
    
    # Apply conversion factor
    result = {k: v * conversion for k, v in normalized_fractions.items()}
    
    return result