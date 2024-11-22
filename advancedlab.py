import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# define all our functions first

# function that reads an exported csv file from logger pro and returns time, position, and velocity data
def measurements(num):
    df = pd.read_csv('advancedlabtest.csv')

    # offsets to line up each run
    offsets = [0.05, 0.2, 0.5, 0.65, 0.2, 0.45, 0.55, 0.75, 0.8, 0.25, 0.55, 0.6, 0.65, 0.7, 0.4]

    time = []
    position = []
    velocity = []
    for index, rows in df.iterrows():
        print(rows[f'Run {num}: Time (s)'], rows[f'Run {num}: Position (m)'])
        time.append(rows[f'Run {num}: Time (s)'])
        position.append(rows[f'Run {num}: Position (m)'])   
        velocity.append(rows[f'Run {num}: Velocity (m/s)']) 
    aligned_time = np.array(time) - offsets[num-1]   
    return aligned_time, position, velocity 

# function that calculates the luminosity of a video over time
def calculate_luminosity(video_path, center, radius):
    cap = cv2.VideoCapture(video_path)
    luminosities = []
    frame_count = 0
    
    # mask to only look at the part of the video where the bulb is
    mask = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        masked_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)
        
        # average luminosity of the masked frame
        luminosity = np.mean(masked_frame[mask == 255])
        luminosities.append(luminosity)
        
        frame_count += 1
        
    cap.release()
    return luminosities

# offsets to line up each run
offsets = [1.36, 3.07, 1.45, 0.8, 0.57, 1.37, 0.765, 0.84, 1.1, 0.467, 0.506, 0.76, 0.84, 0.64, 0.43]

# generate luminosity plots
def plot_luminosity_over_time(luminosities, frame_rate, num):
    time = np.arange(0, len(luminosities)) / frame_rate
    aligned_time = time - offsets[num-1]    
    return aligned_time, luminosities

# have some global time grid that we interpolate all our data to
def interpolate(x, y, x_new):
    y_new = np.interp(x_new, x, y)
    return y_new

# mask dimensions for the vid
center = (630, 280)  
radius = 50 
frame_rate = 30

# ids of the videos to analyze
min = 13
max = 15

#####################

common_time_grid = np.linspace(-1, 5, 1000)  

lums = []
for i in tqdm(range(min, max+1)):
    video_path = f'/Users/aavikwadivkar/Documents/FocusedPhysics/advancedlabclips/advancedlab{i}.mp4' 
    luminosities = calculate_luminosity(video_path, center, radius)
    time, lum = plot_luminosity_over_time(luminosities, frame_rate, i)
    lums.append(interpolate(time, lum, common_time_grid))

measure = []

for i in tqdm(range(min, max+1)):
    measure.append(interpolate(measurements(i)[0], measurements(i)[2], common_time_grid))

f, axarr = plt.subplots(3)

axarr[0].set_title('Velocity, Luminosity, and their Ratio Over Time for 250 g mass')


avgvel = np.mean(measure, axis=0)
axarr[0].set_ylabel('Velocity (m/s)')
axarr[0].ymin = 0
axarr[0].plot(common_time_grid, avgvel, label=f'Average Trial Velocity')


avglum = np.mean(lums, axis=0)
axarr[1].set_ylabel('Luminosity')
axarr[1].plot(common_time_grid, avglum, label=f'Average Trial Luminosity')

ratio = avgvel/avglum
axarr[2].set_xlabel('Time (s)')
axarr[2].set_ylabel('Ratio')
axarr[2].plot(common_time_grid, ratio, label='Ratio')

for i in range(3):
    axarr[i].legend()
    axarr[i].grid()
    axarr[i].set_xlim(-0.5, 5)

axarr[2].set_ylim(0, 5)

# plt.savefig('/Users/aavikwadivkar/Documents/FocusedPhysics/tripplot250.png')
plt.show()