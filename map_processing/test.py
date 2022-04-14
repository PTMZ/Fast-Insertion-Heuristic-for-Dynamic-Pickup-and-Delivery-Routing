import pickle
import numpy as np

# Load final lat_long + time_matrix
with open('final_points.pkl', 'rb') as f:
    final_points = pickle.load(f)

with open('final_matrix.pkl', 'rb') as f:
    final_time = pickle.load(f)

print(final_points[846])
print(final_points[522])
print(final_time[846][522]/60)
#print(final_time[326][30]/60)



