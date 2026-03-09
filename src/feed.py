def build_total_feed(config):
    
    feed_cfg = config["feed"]
    total_flow = feed_cfg["total_flow_kmol_h"]
    composition = feed_cfg["composition"]

    frac_sum = sum(composition.values())

    # Normalize fractions if they are slightly off due to rounding
    if abs(frac_sum - 1.0) > 1e-6:
        composition = {k: v / frac_sum for k, v in composition.items()}

    feed_stream = {}

    for comp, frac in composition.items():
        feed_stream[comp] = total_flow * frac

    return feed_stream