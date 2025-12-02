module Problems
using StaticArrays
function prob_Lima_N2()
    L = (
        U=-87211375.744478,
        V=1.0,
        N=[1e4]
    )

    G = (
        U=-87211375.744478,
        V=1.0,
        N=[1e4]
    )

    T = (
        U=-87211375.744478,
        V=1.0,
        N=[1e4]
    )
    V_spec = 19.9
    T_spec = 293.15
    P_spec = 101325.0
    Stab = (c=[19487.12093974618], T=280.0)
    (; L, G, T, Stab)
end


function prob_castier()
    L = (U=-544956.2614300611, T=297.997716, V=1500.361229e-6, N=[0.335680, 35.684022])
    G = (U=-211544.60739702464, T=297.997716, V=5130.638771e-6, N=[9.664320, 54.315978])
    
    T = (U=-756500.8688270857, T=297.997716, V=52869.0e-6, N=[4.0, 6.0])
    Stab = (c=[0.000409, 34.5372] .* 1e3, T=151.8)
    T_spec = 300.0
    P_spec = 5e6
    (; L, G, T, Stab, T_spec, P_spec)
end

function prob_castier2()

    T_spec = 373.15
    P_spec = 1000.0 #1e-2 * 1e5
    # P_spec = 0.100292E-1
    # V_spec = 0.00029701177437835627
    V_spec= 0.0004714
    Stab = (;c=[7.155786053308659e-6, 0.009066854549791037, 0.024116066958271946, 0.0034854250292711436, 0.0007629123861000657], T = T_spec)
    L = (U=-756500.8688270857, T=373.15, V=V_spec-eps(Float64), N=[9.999016054947432e-9, 0.09999999790422619, 0.5999999939900136, 0.19999999901646356,
    0.09999999975100068])
    G = (U=-756500.8688270857, T=373.15, V=eps(Float64), N=[9.839450525683898e-04, 2.0957738183779995e-01, 6.009986397437217e-08, 9.835364467303265e-04,
    2.4899932116184686e-10])
    T = (U=-756500.8688270857, T=T_spec, V=V_spec, N=[1e-8, 0.1, 0.6, 0.2, 0.1])
    (; L, G,T, T_spec, P_spec, Stab)
end

function prob_castier21()

    T_spec = 310.0
    P_spec = 20 * 1e6
    T = (U=-756500.8688270857, T=373.15, V=4.714e-4, N=[1e-2, 1e-20, 1e-20, 1e-20, 1e-20])
    (; L = T, G = T, T, T_spec, P_spec)
end

function prob_Goncalves()
    T_spec = 298.15
    P_spec = 2e6
    V_spec = 30.0

end

function prob_Arendsen()
    T_spec = 293.15
    P_spec = 5.07e5
    L = (U=-544956.2614300611, T=297.997716, V=1502.361229e-6, N=[0.335680, 35.684022])
    G = (U=-211544.60739702464, T=297.997716, V=51366.638771e-6, N=[9.664320, 54.315978])
    T = (U=-756500.8688270857, T=T_spec, V=1.0, N=[0.5, 0.5])
    # Stab = (c=[0.000409, 34.5372] .* 1e3, T=151.8)
    
    (; L, G, T, T_spec, P_spec)
end

function prob_castier3()
    
    T_spec = 403.15
    P_spec = 4.494 * 1e6
    V_spec = 0.34828
    T = (U=-756500.8688270857, T=T_spec, V=V_spec, N=[10.8, 360.8, 146.5, 233, 233, 15.9, 500])
    (; T, T_spec, P_spec)
end

function prob_tmp()
    
    P_spec = 285.68e5
    T_spec = 273.15 + 40.5
    H_spec = -7138.587593537726
    S_spec = -57.70356197258864
    V_spec = 5.75046203e-5
    L = (U=-544956.2614300611, T=297.997716, V=1502.361229e-6,   N=[0.335680, 35.684022])
    G = (U=-211544.60739702464, T=297.997716, V=51366.638771e-6, N=[9.664320, 54.315978])
    T = (U=-8781.37958690101, H = H_spec, T=297.997716, V=5.75046203e-5, N=[0.726, 0.264])
    Stab = (c=[0.000409, 34.5372] .* 1e3, T=151.8)
    

    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
end

function prob_1()
    P_spec = 2.500170880449619e6
    H_spec = -624319.3345485891
    S_spec = -4335.499558241875
    V_spec = 52869.0e-6
    L = (U=-544956.2614300611, T=297.997716, V=1502.361229e-6,   N=SA[0.335680, 35.684022])
    G = (U=-211544.60739702464, T=297.997716, V=51366.638771e-6, N=SA[9.664320, 54.315978])
    T = (U=-756500.8688270857, H = H_spec, T=297.997716, V=52869.0e-6, N=SA[10.0, 90.0])
    Stab = (c=SA[0.000409, 34.5372] .* 1e3, T=151.8)
    T_spec = 297.9977170044044

    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
end

function prob_2()
    H_spec = -1.50073598e6
    S_spec = -7390.32683706072
    V_spec = 4268.1e-6
    L = (U=-1510985.75, T=298.000861, V=4165.67e-6, N=[0.93070, 98.941685])
    T = (U=-1511407.6, T=298.000861, V=4268.1e-6, N=[0.95, 99.05])
    G = (U=-421.85, T=298.000861, V=102.426e-6, N=[0.0193, 0.1083])
    T_spec = 298.000875588247
    P_spec = 2.5003186454949975e6
    Stab = (c=[0.146112, 0.736148] .* 1e3, T=291.91)
    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)

end

function prob_3()
    H_spec = -130428.4133
    S_spec = -2613.988022727672
    V_spec=80258.1e-6
    L = (U=-566.777015, T=297.996887,    V=1.562506e-6, N=[0.000349, 0.037113])
    G = (U=-330516.922985, T=297.996887, V=80256.5374e-6, N=[15.099651, 84.862887])
    T = (U=-331083.7, T=297.996887, V=80258.1e-6, N=[15.1, 84.9])
    Stab = (c=[0.222, 23.792] .* 1e3, T=297.84)
    T_spec = 297.996887
    P_spec = 2.500125055243379e6
    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
end

function prob_4()
    H_spec = -535905.43397
    S_spec = -4579.402679214386
    V_spec = 9926.7e-6
    L = (U=-245807.965175, T=361.997885, V=3512.626019e-6, N=[3.551418, 33.609473])
    G = (U=-390660.034825, T=361.997885, V=6414.083981e-6, N=[6.448582, 56.390527])
    T = (U=-636468.0, T=361.997885, V=9926.7e-6, N=[10.0, 90.0])
    Stab = (c=[1.0113, 10.0567] .* 1e3, T=361.8)
    T_spec = 361.997885
    P_spec = 1.013051326494734e7
    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
end

function prob_5()
    H_spec = -1.59364418e7
    S_spec = -54937.804162587236
    V_spec = 479845.0e-6
    L = (
        U=-15892619.468615,  # Internal energy for Phase 1 (Liquid)
        T=299.999735,  # Temperature for Phase 1
        V=78647.609580e-6,  # Volume for Phase 1 in m³
        N=[6.595664, 292.574168, 122.030404, 214.470841, 219.174563, 15.574400]  # Mole numbers for Phase 1 (N_C1 to N_C6)
    )

    G = (
        U=-3798866.931385,  # Internal energy for Phase 2 (Gas)
        T=299.999735,  # Temperature for Phase 2
        V=401197.309204e-6,  # Volume for Phase 2 in m³
        N=[4.203436, 68.225832, 24.416050, 18.529159, 13.885437, 0.325600]  # Mole numbers for Phase 2 (N_C1 to N_C6)
    )

    T = (
        U=-16272506.4,  # Total internal energy (sum of Phase 1 and Phase 2)
        T=299.999735,  # Total system temperature
        V=479845.0e-6,  # Total volume (sum of Phase 1 and Phase 2 volumes)
        N=[10.9, 360.8, 146.5, 233.0, 233.0, 15.9]  # Total mole numbers (sum of Phase 1 and Phase 2 mole numbers)
    )
    Stab = (c=[0.0422, 5.648, 1.494, 1.758, 5.504, 0.6127] .* 1e3, T=122.97)
    T_spec = 299.999735
    P_spec = 700360.6123844441
    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
end

function prob_5_Goncalves()
    H_spec = -1.59364418e7
    S_spec = -54937.804162587236
    V_spec = 4.4232
    T_spec = 298.15
    P_spec = 700360.6123844441
    L = (
        U=-15892619.468615,  # Internal energy for Phase 1 (Liquid)
        T=299.999735,  # Temperature for Phase 1
        V=78647.609580e-6,  # Volume for Phase 1 in m³
        N=[6.595664, 292.574168, 122.030404, 214.470841, 219.174563, 15.574400]  # Mole numbers for Phase 1 (N_C1 to N_C6)
    )

    G = (
        U=-3798866.931385,  # Internal energy for Phase 2 (Gas)
        T=299.999735,  # Temperature for Phase 2
        V=401197.309204e-6,  # Volume for Phase 2 in m³
        N=[4.203436, 68.225832, 24.416050, 18.529159, 13.885437, 0.325600]  # Mole numbers for Phase 2 (N_C1 to N_C6)
    )

    T = (
        U=-16272506.4,  # Total internal energy (sum of Phase 1 and Phase 2)
        T=T_spec,  # Total system temperature
        V=V_spec,  # Total volume (sum of Phase 1 and Phase 2 volumes)
        N=[10.8, 360.8, 146.5, 233.0, 233.0, 15.9]  # Total mole numbers (sum of Phase 1 and Phase 2 mole numbers)
    )
    Stab = (c=[18.343460113669014, 1729.4104976931485, 816.6034962176252, 3038.8938672860386, 4216.813137005761, 892.1568523114585], T=298.15)
    
    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
end

# Problem 6 Data
function prob_6()
    H_spec = 1.24900732e6
    S_spec = -9052.420340826524
    V_spec = 289380.3e-6
    L = (
        U=-150012.775415,  # Internal energy for Phase 1 (Liquid)
        T=394.998501,  # Temperature for Phase 1
        V=16232.875672e-6,  # Volume for Phase 1 in m³
        N=[0.735307, 27.089302, 11.174436, 19.334487, 19.881086, 1.508810]  # Mole numbers for Phase 1 (N_C1 to N_C6)
    )

    G = (
        U=174870.975415,  # Internal energy for Phase 2 (Gas)
        T=394.998501,  # Temperature for Phase 2
        V=273147.424328e-6,  # Volume for Phase 2 in m³
        N=[10.066493, 333.710698, 135.325654, 213.665513, 213.118914, 14.391190]  # Mole numbers for Phase 2 (N_C1 to N_C6)
    )

    T = (
        U=24858.200000,  # Total internal energy
        T=394.998501,  # Total temperature
        V=289380.3e-6,  # Total volume
        N=[10.801800, 360.800000, 146.500090, 233.000000, 233.000000, 15.900000]  # Total mole numbers
    )
    Stab = (c=[0.0464123, 1.73853, 0.718791, 1.26159, 1.3047, 0.101004] .* 1e3, T=394.54)
    T_spec = 394.9984980974212
    P_spec = 4.230243484663291e6
    (; L, G, T, Stab, T_spec, P_spec, H_spec, S_spec, V_spec)
    
end

# Problem 7 Data, free water flash
function prob_6_2()
    L = (
        U=-16186287.424419,  # Internal energy for Phase 1 (Liquid)
        T=145.637,  # Temperature for Phase 1
        V=401635.14e-6,  # Volume for Phase 1 in m³
        N=[10.799, 360.77, 146.490874, 232.986135, 232.967170, 15.896099]  # Mole numbers for Phase 1 (N_C1 to N_C6)
    )

    G = (
        U=-3145.27,  # Internal energy for Phase 2 (Gas)
        T=145.637,  # Temperature for Phase 2
        V=6.12856e-6,  # Volume for Phase 2 in m³
        N=[0.000273, 0.029, 0.009126, 0.0138, 0.0328, 0.0039]  # Mole numbers for Phase 2 (N_C1 to N_C6)
    )

    T = (
        U=-16189432.703751,  # Total internal energy
        T=145.637,  # Total temperature
        V=401641.275881e-6,  # Total volume
        N=[10.801800, 360.800000, 146.500090, 233.000000, 233.000000, 15.900000]  # Total mole numbers
    )
    Stab = (c=[44.6058, 4818.95, 1489.11, 2262.41, 5356.83, 636.588], T=145.637)
    (; L, G, T, Stab)
end


function prob_7_2phase()
    # Phase 1 (Solid)
    G = (
        T=130.2919699330592,             # U [J]
        U=-616129.2004606545,             # T [K] (assuming same temperature as other phases)
        V=0.00019624833984375,          # V [m³]
        N=[0, 0, 0, 0, 0, 0, 10.17]  # N [mol]
    )

    # Phase 2 (Solid)
    L = (
        T=130.2919699330592,             # U [J]
        U=-1.6114e7,             # T [K] (assuming same temperature as other phases)
        V=0.40172,       # V [m³]
        N=[10.8, 360.8, 146.5, 233.0, 233.0, 15.9, 0.03]  # N [mol]
    )

    T = (
        T=130.2919699330592,
        U=-17008802.6,
        V=401916.6e-6,
        N=[10.8, 360.8, 146.5, 233.0, 233.0, 15.9, 14.0]
    )
    Stab = (c=[1.79366e-43, 8.96831e-44, 2.24208e-44, 4.48416e-44, 8.75812e-47, 1.79366e-43, 51857.3], T=130.29)
    # Stab = (c = [0.00027473,  0.00230545,  0.000656492,  0.000249601,  0.000184547,  2.08648e-6,  7.9766e-5] .* 1e3, T = 130.29)
    (; L, G, T, Stab)

end

# Wrong initial guess
# Problem 7 Data
function prob_7()
    # Problem 7

    # Phase 1 (Liquid)
    L = (
        U=-616129.2004606545,           # U [J]
        T=130.29,              # T [K]
        V=0.00019624833984375,             # V [m³]
        N=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.1769]  # N [mol]
    )

    # Phase 2 (Gas)
    G = (
        U=-16414184.03,      # U [J]
        T=130.29,              # T [K]
        V=0.40176,         # V [m³]
        N=[10.8, 360.8, 146.5, 233.0, 233.0, 15.9, 5.85847]  # N [mol]
    )

    # Phase 3 (Solid)
    S = (
        U=-303290.363609,          # U [J]
        T=130.29,              # T [K]
        V=320889.039078e-6,        # V [m³]
        N=[3.552183, 54.622840, 19.454565, 14.441443, 10.737644, 0.248680, 0.498216]  # N [mol]
    )

    # Total (T)
    T = (
        T=130.29,
        U=-17008802.6,
        V=401916.6e-6,
        N=[10.8, 360.8, 146.5, 233.0, 233.0, 15.9, 14.0]
    )

    Stab = (c=[1.79366e-43, 8.96831e-44, 2.24208e-44, 4.48416e-44, 8.75812e-47, 1.79366e-43, 51857.3], T=130.29)
    # Stab = (c = [0.00027473,  0.00230545,  0.000656492,  0.000249601,  0.000184547,  2.08648e-6,  7.9766e-5] .* 1e3, T = 130.29)
    (; L, G, T, Stab)

end


# Problem 8 Data
function prob_8()
    # Problem 8

    # Phase 1 (Liquid)
    G = (
        U=-4556984.999158,         # U [J]
        T=300.024831,              # T [K]
        V=2120.250219e-6,          # V [m³]
        N=[0.000032, 0.000173, 0.000014, 0.000000, 0.000001, 0.000000, 99.985323]  # N [mol]
    )

    # Phase 2 (Gas)
    L = (
        U=-18469.300842,           # U [J]
        T=300.024831,              # T [K]
        V=89.649781e-6,            # V [m³]
        N=[0.010768, 0.360627, 0.146486, 0.233000, 0.232999, 0.015900, 0.014677]  # N [mol]
    )

    # Total (T)
    T = (
        U=-4575454.3,
        T=286.3364653660059,
        V=2209.9e-6,
        N=[0.0108, 0.3608, 0.1465, 0.233, 0.233, 0.0159, 100.0]
    )
    Stab = (c = [20.3981,  823.915,  735.224,  6843.38,  3399.32,  1079.96,  4.25218], T = 286.33)

    (; L, G, T, Stab)
end
# Problem 9 Data

function prob_9()
    # Problem 9

    # Phase 1 (Liquid)
    L = (
        U=-4248079.288176,         # U [J]
        T=349.47020372260494,              # T [K]
        V=2558.556768e-6,          # V [m³]
        N=[0.000813, 0.013817, 0.002294, 0.000395, 0.000684, 0.000005, 111.866010]  # N [mol]
    )

    # Phase 2 (Gas)
    G = (
        U=-3197022.030237,         # U [J]
        T=349.47020372260494,              # T [K]
        V=99659.564416e-6,         # V [m³]
        N=[5.516386, 209.103028, 86.413985, 150.396122, 154.757385, 11.577650, 59.314485]  # N [mol]
    )

    # Phase 3 (Solid)
    S = (
        U=357048.818413,           # U [J]
        T=349.47020372260494,              # T [K]
        V=163613.178816e-6,        # V [m³]
        N=[5.282801, 151.683155, 60.083721, 82.603483, 78.241932, 4.322345, 28.819505]  # N [mol]
    )

    T = (
        U=-7088052.5,         # Volume
        V=265831.3e-6,         # Moles
        T=349.47020372260494,
        N=[10.8, 360.8, 146.5, 233.0, 233.0, 15.9, 200.0]
    )
    Stab = (c=[0.000880672, 0.00387829, 0.000295245, 4.9211e-6, 1.01076e-5, 6.61121e-9, 47111.0], T=349.47)
    (; L, G, S, T, Stab)
end

# From Mikyska paper, take the solutionand add phase 2 and 3
function prob_9_2()
    L = (
        U=-3197022.030237,
        V=99659.564416e-6,
        N=[5.516386, 209.103028, 86.413985, 150.396122, 154.757385, 11.577650, 59.314485]
    )
    G = (
        U=357048.818413,
        V=163613.178816e-6,
        N=[5.282801, 151.683155, 60.083721, 82.603483, 78.241932, 4.322345, 28.819505]
    )

    # [10.7992, 360.787, 146.49, 232.99966810631008, 232.9994017291831, 15.89, 67.15457]
    T = (
        U=-2.839973211824e6,
        V=0.263272743232,
        N=[10.7992,  360.786,  146.498,  233.0,  232.999,  15.9,  88.134]
    )
    Stab = (c=[57.0157,  2433.09, 1011.72,  1924.82,  2098.66,  182.664,  1032.83], T=379.756)
    (; L, G, T, Stab)

end

function prob_CO2()
    L = (
        U=-87211375.744478,
        V=1.0,
        N=[1e4]
    )

    G = (
        U=-87211375.744478,
        V=1.0,
        N=[1e4]
    )

    T = (
        U=-87211375.744478,
        V=1.0,
        N=[1e4]
    )
    Stab = (c=[19487.12093974618], T=280.0)
    (; L, G, T, Stab)
end

function prob_CO2_tmp()
# -1.7223587411664468e8, 1.0, [15348.165463116053]
    L = (
        U=-1.7223587411664468e8,  # Internal energy for Phase 1 (Liquid)
        V=1.0,                    # Volume for Phase 1 in m³
        N=[15348.165463116053]    # Mole numbers for Phase 1
    )

    G = (
        U=-1.7223587411664468e8,  # Internal energy for Phase 2 (Gas)
        V=1.0,                    # Volume for Phase 2 in m³
        N=[15348.165463116053]    # Mole numbers for Phase 2
    )

    T = (
        U=-1.7223587411664468e8,  # Total internal energy
        V=1.0,                    # Total volume
        N=[15348.165463116053]    # Total mole numbers
    )
    Stab = (c=[19487.12093974618], T=280.0)  # Stability parameters
    
    (; L, G, T, Stab)

end

end