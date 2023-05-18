R = 8.3144598  # Universal Gas Constant
Ta = 273.15  # Celcius to Kelvin

# Starch Gelatinization
Tu = 315.4 - Ta  # Un-gelatinezed temperature C - threshold temperature
Tg = 336.5 - Ta  # Gelatinezed temperature C - threshold temperature

# Enzimes Denaturation
Kalpha0 = 3.86 * 10 ** 34  # Alpha Amylase Pre-exponential coefficient min^-1
Edaplha = 2.377 * 10 ** 5  # Alpha Amylase Activation energy J/mol
Halpha = 9.72 * 10 ** -5  # Dissolution coefficient

Kbeta0 = 9.46 * 10 ** 67  # Beta Amylase Pre-exponential coefficient min^-1
Edbeta = 4.439 * 10 ** 5  # Beta Amylase Activation energy J/mol
H_beta = 7.57 * 10 ** -5  # Dissolution coefficient

# Starch Conversion
Agmlt0 = 6.42 * 10 ** 9  # Maltotriose Pre-exponential coefficient min^-1
Agdex0 = 3.77 * 10 ** 10  # Dextrin Pre-exponential coefficient min^-1
Ealpha = 1.03 * 10 ** 5  # Alpha Amylase Activation - Activation energy J/mol

Bgl0 = 1.62 * 10 ** 40  # Glucose Pre-exponential coefficient min^-1
Bmal0 = 1.05 * 10 ** 42  # Maltose Pre-exponential coefficient min^-1
Bldex0 = 1.09 * 10 ** 41  # Limit Dextrins Pre-exponential coefficient min^-1
Ebeta = 2.93 * 10 ** 5  # Beta Amylase Activation - Activation energy J/mol

Km = 2.8  # Michaelis-Menten Maltose Constant
