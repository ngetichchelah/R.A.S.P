def add_severe_accident_features(X):
    """Add features that help detect severe accidents"""
    X = X.copy()

    # High-speed scenarios
    X['extreme_speed'] = (X['speed_limit'] > 60).astype(int)
    X['high_speed_rural'] = ((X['speed_limit'] > 50) & 
                            (X['urban_or_rural_area'] == 2)).astype(int)

    # Dangerous conditions
    X['night_rural'] = ((X['is_night'] == 1) & 
                       (X['urban_or_rural_area'] == 2)).astype(int)
    X['bad_weather_high_speed'] = ((X['weather_conditions'] > 1) & 
                                  (X['speed_limit'] > 50)).astype(int)

    # # Complex scenarios
    # X['complex_junction_high_speed'] = ((X['junction_control'] > 2) & 
    #                                    (X['speed_limit'] > 40)).astype(int)

    return X
