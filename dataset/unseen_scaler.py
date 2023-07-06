class UnseenScaler:
    """
    Normally scaler use the max and min from seen data. However, there are cases where
    we know the max and min beforehand even when they do not present in the data. This class
    addresses that need.
    """
    def __init__(self):
        pass