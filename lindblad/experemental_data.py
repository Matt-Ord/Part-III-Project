import numpy as np

temperature = np.array(
    [
        255,
        245,
        235,
        230,
        225,
        220,
        210,
        205,
        200,
        195,
        190,
        185,
        180,
        175,
        165,
        160,
        155,
        145,
        145,
        135,
        135,
        130,
        125,
        112,
    ]
)

# jumprates are in /s
absuppererrorvalue = 1e10 * np.array(
    [
        2.4888,
        1.7503,
        1.5505,
        1.5416,
        1.2597,
        1.1288,
        1.1095,
        0.7155,
        0.6715,
        0.5150,
        0.5240,
        0.3972,
        0.4381,
        0.3046,
        0.3793,
        0.3208,
        0.3321,
        0.2562,
        0.3153,
        0.3498,
        0.3949,
        0.4695,
        0.3706,
        0.5120,
    ]
)
jumprate = 1e10 * np.array(
    [
        2.4041,
        1.6810,
        1.4720,
        1.4636,
        1.1890,
        1.0533,
        1.0000,
        0.6952,
        0.6487,
        0.4861,
        0.4833,
        0.3685,
        0.3859,
        0.2942,
        0.3208,
        0.3028,
        0.2976,
        0.2336,
        0.2518,
        0.2976,
        0.3379,
        0.3904,
        0.2606,
        0.4407,
    ]
)
abslowererrorvalue = 1e10 * np.array(
    [
        2.3223,
        1.5959,
        1.3895,
        1.3895,
        1.0904,
        0.9828,
        0.8962,
        0.6715,
        0.6123,
        0.4642,
        0.4381,
        0.3418,
        0.3399,
        0.2842,
        0.2683,
        0.2859,
        0.2698,
        0.1854,
        0.2230,
        0.2432,
        0.2892,
        0.3046,
        0.1524,
        0.3728,
    ]
)
