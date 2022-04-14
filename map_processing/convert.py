
N_PIX = 230
W_PIX = 312

N_LAT = 1.4363977
W_LON = 103.6374396

DEL_LAT = -7760.504806
DEL_LON = 7768.810023

def latlong2yx(lat, lon):
    return int(N_PIX + (lat-N_LAT)*DEL_LAT), int(W_PIX + (lon-W_LON)*DEL_LON)


def yx2latlong(y, x):
    return (y-N_PIX)/DEL_LAT + N_LAT, (x-W_PIX)/DEL_LON + W_LON