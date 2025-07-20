import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, timezone
import os
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import  tqdm, tqdm_notebook


def append_csv_files(folder_path, output_file):
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    for filename in os.listdir(folder_path):  # Iterate through each file in the specified folder
        if filename.endswith('.csv'):  # Check if the file has a .csv extension
            file_path = os.path.join(folder_path, filename)  # Construct the full file path
            df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame

            if combined_df.empty:  # If the combined DataFrame is empty, initialize it with the first DataFrame
                combined_df = df
            else:
                if list(combined_df.columns) == list(df.columns):  # Check if the column names are the same
                    combined_df = pd.concat([combined_df, df], ignore_index=True)  # Append the rows

    combined_df.to_csv(output_file, index=False) 

#examples: 
#folder_path = 'C:\\Users\\kmh\\Documents\\DATA\\2024-5\\RFID'  # Replace with your folder path
#output_file = 'RFID_24Dec11_04.csv'
#append_csv_files(folder_path, output_file) 

def append_force_text_files(folder_path, output_file):
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Open the output file in write mode
    with open(output_file, 'w') as f_out:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, delimiter=',')  # Read the text file into a DataFrame

                if combined_df.empty:
                    combined_df = df
                    combined_df.to_csv(f_out, index=False, sep=',')  # Write the header and first chunk
                else:
                    df.to_csv(f_out, index=False, sep=',', header=False)  # Append without writing the header

# Function to convert seconds since 1904-01-01 to datetime format for force data
def convert_time_force(seconds):
    base_time = datetime(1904, 1, 1)
    return (base_time + timedelta(seconds=seconds)).strftime('%Y-%m-%d %H:%M:%S:%f')


# Function to convert microseconds since 1970-01-01 to datetime format for rfid data
def convert_time_rfid(microseconds):
    if pd.isna(microseconds):
        return np.nan
    base_time = datetime(1970, 1, 1)
    return (base_time + timedelta(microseconds=microseconds)).strftime('%Y-%m-%d %H:%M:%S:%f')


# Function to convert microseconds since 1970-01-01 to datetime format for IR data
def convert_time_IR(milliseconds):
    base_time = datetime(1970, 1, 1)
    return (base_time + timedelta(milliseconds=milliseconds)).strftime('%Y-%m-%d %H:%M:%S:%f')



# Define the sequence to filter

# the df should have the following columns and format:

#Sensor               int32
#Time               float64
#datetime    datetime64[ns]

# Function to find and filter the sequence in the 'Sensor' column with possible repeated elements
# e.g. sequence = [1, 2, 3, 4, 5]
def find_and_filter_sequence(df, seq):
    seq_len = len(seq)
    filtered_indices = []
    takeoff_event = []
    event_number = 1
    i = 0
    
    while i < len(df):
        if df['Sensor'].iloc[i] == seq[0]:
            match = True
            seq_index = 0
            for j in range(i, len(df)):
                if df['Sensor'].iloc[j] == seq[seq_index]:
                    seq_index += 1
                    if seq_index == seq_len:
                        # Include all subsequent repeated elements of the last sequence element
                        while j + 1 < len(df) and df['Sensor'].iloc[j + 1] == seq[-1]:
                            j += 1
                        filtered_indices.extend(range(i, j + 1))
                        takeoff_event.extend([event_number] * (j - i + 1))
                        event_number += 1
                        i = j
                        break
                elif df['Sensor'].iloc[j] != seq[seq_index - 1]:
                    match = False
                    break
        i += 1
    
    filtered_df = df.iloc[filtered_indices].copy()
    filtered_df['takeoff_event'] = takeoff_event
    
    return filtered_df

# use example
#filtered_IR_Sensor = find_and_filter_sequence(IR_unique, sequence)
#filtered_IR_Sensor: full dataset with 'Sensor' column filtered

# Display the filtered DataFrame A
#print(filtered_IR_Sensor.head(10))
#print(filtered_IR_Sensor.shape)

# the input df should have the following format, as the product of def find_and_filter_sequence()

#index                     int64
#Sensor                    int32
#Time                    float64
#datetime         datetime64[ns]
#takeoff_event             int64
#dtype: object

def isolate_takeoff(df):

    df = df.drop_duplicates().sort_values(by = 'datetime').reset_index() #drop duplicates that has same timestamp 

    i = 0
    keep_indices = []

    while i < len(df) - 1:
        first_sensor_reading = df['Sensor'].iloc[i]
        
        # Always keep the first reading of a new sensor value
        if i == 0 or df['Sensor'].iloc[i] != df['Sensor'].iloc[i - 1]:
            keep_indices.append(i)
        
        if df['Sensor'].iloc[i + 1] == first_sensor_reading:
            time_diff = (df['datetime'].iloc[i + 1] - df['datetime'].iloc[i]).total_seconds()
            
            if time_diff < 0.5:
                keep_indices.append(i + 1)
            else:
                # Skip all subsequent rows with the same sensor value
                while i < len(df) - 1 and df['Sensor'].iloc[i + 1] == first_sensor_reading:
                    i += 1
        else:
            keep_indices.append(i)
        
        i += 1

    # Add the last index if it wasn't added
    if i == len(df) - 1 and (df['datetime'].iloc[i] - df['datetime'].iloc[i - 1]).total_seconds() < 0.5:
        keep_indices.append(i)

    # Drop duplicates in keep_indices
    keep_indices = list(dict.fromkeys(keep_indices))

    return df.iloc[keep_indices]


# use example
# Apply the function and show the result
#final_filtered_df = isolate_takeoff(filtered_df)

#RFID_match should have the following format:
#  
#id                              object
#status                          object
#epoch_time_converted    datetime64[ns]
#dtype: object
#<class 'pandas.core.frame.DataFrame'>

#takeoff_match should have the following format:
# 
#takeoff_event	datetime	RFID
#0	1	2024-12-11 08:00:48.289	<NA>
#1	2	2024-12-11 08:02:31.780	<NA>
#2	3	2024-12-11 08:02:36.599	<NA>


def process_row(i, RFID_match, takeoff_match):
    print(f"Processing row: {i}")
    if RFID_match['status'].iloc[i] == "Arrive" and RFID_match['status'].iloc[i + 1] in ["Displace", "Depart"]:
        arrival_time = RFID_match['epoch_time_converted'].iloc[i]
        depart_time = RFID_match['epoch_time_converted'].iloc[i + 1]
        arrival_RFID = RFID_match['id'].iloc[i]
        #print(f"Arrive found at row {i} with RFID {arrival_RFID} from {arrival_time} to {depart_time}")

        # Vectorized operation to assign RFID
        mask = (takeoff_match['datetime'] > arrival_time) & (takeoff_match['datetime'] < depart_time)
        takeoff_match.loc[mask, 'RFID'] = arrival_RFID
        print(f"Assigned RFID {arrival_RFID} to {mask.sum()} rows in takeoff_match")

    elif RFID_match['status'].iloc[i] == "Displace" and RFID_match['status'].iloc[i + 1] == "Depart":
        arrival_time = RFID_match['epoch_time_converted'].iloc[i]
        depart_time = RFID_match['epoch_time_converted'].iloc[i + 1]
        arrival_RFID = RFID_match['id'].iloc[i]
        #print(f"Displace found at row {i} with RFID {arrival_RFID} from {arrival_time} to {depart_time}")

        # Vectorized operation to assign RFID
        mask = (takeoff_match['datetime'] > arrival_time) & (takeoff_match['datetime'] < depart_time)
        takeoff_match.loc[mask, 'RFID'] = arrival_RFID
        print(f"Assigned RFID {arrival_RFID} to {mask.sum()} rows in takeoff_match")

# Use example: Process rows sequentially
#for i in tqdm(range(len(RFID_match) - 1)):
#    process_row(i, RFID_match, takeoff_match) #just 6.4 rows

#the input df should be in this format
#index              int64
#Sensor             int32
#Time             float64
#datetime          object
#takeoff_event      int64
#RFID              object
#dtype: object

#returns results_df with firstbroken sensors in the format of: 
#RFID	firstbroken1	firstbroken2	firstbroken3	firstbroken4	firstbroken5
#0	3B0018C6F9	1.733904e+12	NaN	NaN	NaN	NaN
#1	<NA>	NaN	1.733904e+12	NaN	NaN	NaN
#2	<NA>	NaN	NaN	1.733904e+12	NaN	NaN
#3	<NA>	NaN	NaN	NaN	1.733904e+12	NaN
#4	<NA>	NaN	NaN	NaN	NaN	1.733904e+12


def firstbroken(df):
    # Initialize the results list
    results_list = []
    # Initialize start_index
    start_index = 0

    # Iterate over the DataFrame
    for i in range(len(df)):
        first_read = df['Time'].iloc[start_index]
        current_sensor = df['Sensor'].iloc[start_index]

        #rfid assignment
        rfid = df['RFID'].iloc[start_index] if current_sensor == 1 else pd.NA

        # Debugging print statements 
        print(f"Iteration {i}:")
        print(f"  start_index: {start_index}")
        print(f"  first_read: {first_read}")
        print(f"  current_sensor: {current_sensor}")
        print(f"  rfid: {rfid}")


        for j in range(start_index, len(df)):
            if df['Sensor'].iloc[j] != current_sensor:
                start_index = j
                print(f"  Sensor changed at index {j}, new start_index: {start_index}")
                break

        results_list.append({
            f'RFID': rfid,
            f'firstbroken{current_sensor}': first_read,
        })

    # Convert results_list to DataFrame
    results_df = pd.DataFrame(results_list)

    print(results_df)

    

def irfid_allign(results_df):
    # Initialize an empty DataFrame with specified columns
    irfid_alligned = pd.DataFrame(columns=['RFID', 'firstbroken1', 'firstbroken2', 'firstbroken3', 'firstbroken4', 'firstbroken5'])

    # Iterate through results_df
    for i in range(len(results_df) - 4):
        if pd.notna(results_df['RFID'].iloc[i]):
            rfid = results_df['RFID'].iloc[i]
            firstbroken1 = results_df['firstbroken1'].iloc[i]
            firstbroken2 = results_df['firstbroken2'].iloc[i+1]
            firstbroken3 = results_df['firstbroken3'].iloc[i+2]
            firstbroken4 = results_df['firstbroken4'].iloc[i+3]
            firstbroken5 = results_df['firstbroken5'].iloc[i+4]
            firstbroken_values = [firstbroken1, firstbroken2, firstbroken3, firstbroken4, firstbroken5]

            #print(f"Checking row {i}: RFID={rfid}, firstbroken_values={firstbroken_values}")

            if pd.notna(firstbroken_values).all():
                row = [rfid] + firstbroken_values
                irfid_alligned.loc[len(irfid_alligned)] = row

    print(irfid_alligned)


#output dataset: 
#RFID	firstbroken1	firstbroken2	firstbroken3	firstbroken4	firstbroken5
#0	3B0018C6F9	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12
#1	3B00185E23	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12

#the input irfid_new looks like this:
#RFID	firstbroken1	firstbroken2	firstbroken3	firstbroken4	firstbroken5	datetime	takeoff_event
#0	3B0018C6F9	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	2024-12-11 08:00:48.289	1
#1	3B00185E23	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	2024-12-11 08:02:36.599	3
#2	3B00185CB8	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	1.733904e+12	2024-12-11 08:05:55.755	6


def speed_calculation(irfid_new):
    irfid_calc = pd.DataFrame()
    irfid_calc = irfid_new
    #seconds
    irfid_calc['firstbroken1'] = irfid_calc['firstbroken1']/1000
    irfid_calc['firstbroken2'] = irfid_calc['firstbroken2']/1000
    irfid_calc['firstbroken3'] = irfid_calc['firstbroken3']/1000
    irfid_calc['firstbroken4'] = irfid_calc['firstbroken4']/1000
    irfid_calc['firstbroken5'] = irfid_calc['firstbroken5']/1000

    #time
    irfid_calc['t12'] = irfid_calc['firstbroken2'] - irfid_calc['firstbroken1']
    irfid_calc['t23'] = irfid_calc['firstbroken3'] - irfid_calc['firstbroken2']
    irfid_calc['t34'] = irfid_calc['firstbroken4'] - irfid_calc['firstbroken3']
    irfid_calc['t45'] = irfid_calc['firstbroken5'] - irfid_calc['firstbroken4']

    #speed
    irfid_calc['v1'] = 0.135/irfid_calc['t12']
    irfid_calc['v2'] = 0.135/irfid_calc['t23']
    irfid_calc['v3'] = 0.135/irfid_calc['t34']
    irfid_calc['v4'] = 0.135/irfid_calc['t45']
    irfid_calc['avg.v'] = 0.540/(irfid_calc['firstbroken5'] - irfid_calc['firstbroken1'])

    #acceleration
    irfid_calc['acc1'] = 2*(irfid_calc['v2'] - irfid_calc['v1'])/(irfid_calc['t12'] + irfid_calc['t23'])
    irfid_calc['acc2'] = 2*(irfid_calc['v3'] - irfid_calc['v2'])/(irfid_calc['t23'] + irfid_calc['t34'])
    irfid_calc['acc3'] = 2*(irfid_calc['v4'] - irfid_calc['v3'])/(irfid_calc['t34'] + irfid_calc['t45'])
    irfid_calc['avg.acc'] = irfid_calc['v4'] - irfid_calc['v1']/(irfid_calc['t23']+irfid_calc['t34']+1/2*(irfid_calc['t12']+irfid_calc['t45']))

    #filtering
    irfid_calc[(irfid_calc['t12'] <0.25 )& (irfid_calc['t23'] <0.25)] #flying bird should exit the section less than quarter of a second.
    print(irfid_calc.head(3))
    print(irfid_calc.shape)


#the output irfid_calc looks like this: 
#RFID  firstbroken1  firstbroken2  firstbroken3  firstbroken4  \
#0  3B0018C6F9      1.733904      1.733904      1.733904      1.733904   
#1  3B00185E23      1.733904      1.733904      1.733904      1.733904   

#   firstbroken5                 datetime  takeoff_event           t12  \
#0      1.733904  2024-12-11 08:00:48.289              1  2.830001e-10   
#1      1.733904  2024-12-11 08:02:36.599              3  1.340001e-10   

#            t23  ...           t45            v1            v2            v3  \
#0  1.310001e-10  ...  6.699996e-11  4.770317e+08  1.030534e+09  2.327593e+09   
#1  6.899992e-11  ...  7.399992e-11  1.007462e+09  1.956524e+09  2.288130e+09   

#             v4         avg.v          acc1          acc2          acc3  \
#0  2.014927e+09  1.001855e+09  2.673921e+18  1.372550e+19 -5.002674e+18   
#1  1.824326e+09  1.607142e+09  9.350367e+18  5.181344e+18 -6.974491e+18   
  
#        avg.acc  
#0 -1.310527e+18  
#1 -4.342505e+18  

def assign(RFID_match_keys, F_match_keys, F_df, start_index=0):  
    current_F_valid = pd.DataFrame()
    F_match_keys_in = pd.DataFrame()

    for i in range(len(RFID_match_keys) - 1):
        if RFID_match_keys.loc[i, 'status'] == "Arrive" and RFID_match_keys.loc[i + 1, 'status'] in ["Displace", "Depart"]:
            arrival_time = RFID_match_keys.loc[i, 'timestamp']  
            depart_time = RFID_match_keys.loc[i + 1, 'timestamp'] 
            arrival_RFID = RFID_match_keys.loc[i, 'rfid']

            # Adjust the mask to start from start_index
            mask = (F_match_keys['timestamp'] >= arrival_time) & (F_match_keys['timestamp'] <= depart_time) & (F_match_keys['index'] >= start_index)
                
            if not F_match_keys.loc[mask].empty:
                matching_event = F_match_keys.loc[mask, 'event'].iloc[0]
                matching_start_index = F_match_keys.loc[mask, 'index'].iloc[0]
                matching_end_index = F_match_keys.loc[mask, 'index'].iloc[-1]

                #directly stores assigns RFID to the raw F_df file in current_F_valid
                current_F_valid = F_df.loc[matching_start_index:(matching_end_index+999), :]
                current_F_valid['RFID'] = arrival_RFID
                current_F_valid['event'] = matching_event

                F_match_keys.loc[mask, 'RFID'] = arrival_RFID
                F_match_keys_in = F_match_keys[F_match_keys['RFID'].notna()]

                return F_match_keys_in, current_F_valid, matching_end_index

    return F_match_keys_in, current_F_valid, None


def zero_centering(input_df, columns=['Fx','Fy','Fz','Tx','Ty','Tz'], endframe = 50): #from ZF_ontogeny>src>helpers
    # uses mean of values from the number of frames from the end 0-center. 
    # returns new dateframe with 0-centered columns (specify which or defaults to all F/T) 
    df_list = [x for _, x in input_df.groupby(['event2','day'])] #used to be FlightID 
    
    df_0centered_list = []
    for df in tqdm(df_list):

        lastframe = df['Frame'].shape[0]+1

        after_takeoff = df[(df['Frame'] >  lastframe - endframe) & (df['Frame'] < lastframe)]
        df_0centered = df.copy()
        for col in columns:
            df_0centered[col] = [i-after_takeoff[col].mean() for i in df[col]]
            
        df_0centered_list.append(df_0centered)
        
    output_df = pd.concat(df_0centered_list, ignore_index=True)
        
    return output_df



def halve_frame(df):

    # Get unique days 
    unique_days = df['day'].unique()
    all_days_halved = pd.DataFrame()
    for day in unique_days:
        # Get unique events
        unique_events = day['event2'].unique()

        #Initialize an empty DataFrame to store the results
        all_event_data_halved = pd.DataFrame()


        for event in unique_events:
            event_data = day[day['event2'] == event].copy()
            #calculate the halfway point 
            half_frames = len(event_data)//2
            
            # Select the first half of the DataFrame
            event_data_halved = event_data.iloc[:half_frames, :]
            
            # Append the halved data to the result DataFrame
            all_event_data_halved = pd.concat([all_event_data_halved, event_data_halved], ignore_index=True)
        return all_event_data_halved
        all_days_halved = pd.concat([all_days_halved, all_event_data_halved], ignore_index = True)
    return all_days_halved

def halve_frame2(df):
    # Initialize an empty DataFrame to store the final results
    all_days_halved = pd.DataFrame()
    
    # Iterate through unique days
    unique_days = df['day'].unique()
    for day in unique_days:
        # Filter the DataFrame for the current day
        day_data = df[df['day'] == day]
        
        # Get unique events within the current day
        unique_events = day_data['event2'].unique()
        
        # Initialize an empty DataFrame to store halved data for the current day
        all_event_data_halved = pd.DataFrame()
        
        for event in unique_events:
            # Filter the DataFrame for the current event
            event_data = day_data[day_data['event2'] == event].copy()
            
            # Calculate the halfway point
            half_frames = len(event_data) // 2
            
            # Select the first half of the DataFrame
            event_data_halved = event_data.iloc[:half_frames, :]
            
            # Append the halved data to the result DataFrame
            all_event_data_halved = pd.concat([all_event_data_halved, event_data_halved], ignore_index=True)
        
        # Append the halved data for the current day to the final DataFrame
        all_days_halved = pd.concat([all_days_halved, all_event_data_halved], ignore_index=True)
    
    # Return the final halved DataFrame
    return all_days_halved


def describe_FTs(df, start_frame=0, stop_frame=-1, zero_centered=True):
    
    # if data isn't zero-centered, then center it first
    if not zero_centered:
        df = zero_centering(df)
    
    # if no stop frame specified (=-1 by default) then use all the frames (up to the last)
    if stop_frame == -1:
        stop_frame = df['Frame'].max()
        
    # keep only the "region of interest" b/w start frame & stop frame for each flight
    df_truncated = df[(df['Frame'] > start_frame) & (df['Frame'] < stop_frame)]
    
    # get summary/description of all 6 axes for the region of interest
    df_grouped = df_truncated.groupby(['RFID', 'event2']).agg({'Fx': ['mean', 'std', 'min', 'max'], 
                                                                                   'Fy': ['mean', 'std', 'min', 'max'], 
                                                                                   'Fz': ['mean', 'std', 'min', 'max'], 
                                                                                   'Tx': ['mean', 'std', 'min', 'max'], 
                                                                                   'Ty': ['mean', 'std', 'min', 'max'], 
                                                                                   'Tz': ['mean', 'std', 'min', 'max'],})
    return df_grouped

def butter_filt(original_signal, order, cutoff):
    # Design low-pass Butterworth filter
    sos = signal.butter(order/2, cutoff, 'lp', fs=1000, output='sos') #why is this 960?
    filtered_signal = signal.sosfiltfilt(sos, original_signal)
    return filtered_signal

def apply_butterfilt(FT_df, order, cutoff, columns=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'], calibration=False):
    
    # split data by individual flights/landings/signals
    if calibration:
        grouped_df = FT_df.groupby(['Direction', 'Distance'], dropna=False, as_index=False)
    
    else:
        grouped_df = FT_df.groupby(['RFID', 'event2','day'], dropna=False, as_index=False)

    # Make new dataframe for filtered data    
    butter_df = pd.DataFrame()

    # apply filter to all F/T components of each individual signal
    for name, signal_df in tqdm(grouped_df, total=grouped_df.size().shape[0]):
        for col in columns:
            col_filt = col+'_filt'
            signal_df[col_filt] = butter_filt(signal_df[col], order=order, cutoff=cutoff)

        butter_df = pd.concat([butter_df, signal_df], ignore_index=True)

    
    return butter_df

#noisy needs to have 'Fx_filt' column 
def detect_plateaus(noisy, threshold=0.00000011, min_length=300):
    plateaus = []
    start = None

    
    Fx = noisy['Fx_filt'].values

    for i in range(1, len(noisy)): #for every 200 frames from the beginning, the mean and variance is less than threshold. If yes, add to the plateau 

        if abs(Fx[i] - Fx[i-1]) < threshold:
            if start is None:
                start = i-1
        else:
            if start is not None and (i - start) >= min_length:
                plateaus.append((start, i-1))
                start = None

    if start is not None and (len(Fx) - start) >= min_length:
        plateaus.append((start, len(Fx)-1))

    return plateaus


def detect_bw_Fx(noisy, var_threshold=0.00005, frame_length=50):
    plateaus = []
    Fx = noisy['Fx_filt'].values
    Fx_firsthalf = Fx[:len(Fx)//2] #apparently called array slicing 
    #what if I split the sample in half and if the plateau is in the first half, that's bodyweight. 
    #It is already zero-centred. 
    # Iterate through the data in chunks of frame_length
    for start in range(0, len(Fx_firsthalf), frame_length):
        end = min(start + frame_length, len(Fx_firsthalf))
        segment = Fx_firsthalf[start:end]
        
        # Calculate the mean difference and variance of the current segment
        variance = np.var(segment)
        mean = np.mean(segment)
        
        # Check if both the mean difference and variance are below their respective thresholds
        if variance <= var_threshold and mean > 0.075:
            plateaus.append((start, end-1))
    
    # Calculate the overall mean of the values corresponding to the plateaus
    if plateaus:
        plateau_values = [Fx_firsthalf[start:end] for start, end in plateaus]

        # Flatten the list of arrays into a single array
        all_plateau_values = np.concatenate(plateau_values)
        # Calculate the overall mean
        bodyweight = np.mean(all_plateau_values)
            

    return plateaus, bodyweight

def detect_bw_Ftot(df, var_threshold=0.00005, frame_length=50):
    plateaus = []
    Ftotal = df['Ftotal_filt'].values
    Ftotal_firsthalf = Ftotal[:len(Ftotal)//2] #apparently called array slicing 
    #what if I split the sample in half and if the plateau is in the first half, that's bodyweight. 
    #It is already zero-centred. 
    # Iterate through the data in chunks of frame_length
    for start in range(0, len(Ftotal_firsthalf), frame_length):
        end = min(start + frame_length, len(Ftotal_firsthalf))
        segment = Ftotal_firsthalf[start:end]
        
        # Calculate the mean difference and variance of the current segment
        variance = np.var(segment)
        mean = np.mean(segment)
        
        # Check if both the mean difference and variance are below their respective thresholds
        if variance <= var_threshold and mean > 0.075:
            plateaus.append((start, end-1))
    
    #initialize bodyweight with NaN
    bodyweight = np.nan
    
    # Calculate the overall mean of the values corresponding to the plateaus
    if plateaus:
        plateau_values = [Ftotal_firsthalf[start:end] for start, end in plateaus]

        # Flatten the list of arrays into a single array
        all_plateau_values = np.concatenate(plateau_values)
        # Calculate the overall mean
        bodyweight = np.mean(all_plateau_values)
            

    return plateaus, bodyweight


#basically the same but returns single body weight 
def return_bw_Ftot(df, var_threshold=0.00005, frame_length=50):
    plateaus = []
    Ftotal = df['Ftotal_filt'].values
    Ftotal_firsthalf = Ftotal[:len(Ftotal)//2] #apparently called array slicing 
    #what if I split the sample in half and if the plateau is in the first half, that's bodyweight. 
    #It is already zero-centred. 
    # Iterate through the data in chunks of frame_length
    for start in range(0, len(Ftotal_firsthalf), frame_length):
        end = min(start + frame_length, len(Ftotal_firsthalf))
        segment = Ftotal_firsthalf[start:end]
        
        # Calculate the mean difference and variance of the current segment
        variance = np.var(segment)
        mean = np.mean(segment)
        
        # Check if both the mean difference and variance are below their respective thresholds
        if variance <= var_threshold and mean > 0.075:
            plateaus.append((start, end-1))
    
    # Calculate the overall mean of the values corresponding to the plateaus
    if plateaus:
        plateau_values = [Ftotal_firsthalf[start:end] for start, end in plateaus]

        # Flatten the list of arrays into a single array
        all_plateau_values = np.concatenate(plateau_values)
        # Calculate the overall mean
        bodyweight = np.mean(all_plateau_values)
    
    else:
        bodyweight = np.nan

    return bodyweight

def compute_Ftotal(df, use_filt=True):
    new_col = 'Ftotal'
    if use_filt:
        filt = '_filt'
    else: 
        filt = ''
    df[new_col+filt] = np.sqrt(df['Fx'+filt]**2+df['Fy'+filt]**2+df['Fz'+filt]**2)
    
    return df


def get_force(butter_df):
    # Compute Ftotal first
    butter_df = compute_Ftotal(butter_df)

    # DataFrame to return
    force_outcome = pd.DataFrame(columns=['RFID', 'datetime', 'event_no', 'bodyweight', 'max_Ftot', 'ninety9_perc_Ftot', 'time_max_Ftot',  'Fx_maxFtot',
'Fy_maxFtot', 'max_Fx', 'time_max_Fx', 'max_Fy', 'time_max_Fy', 'max_Fz',
 'time_max_Fz'])

    # Group the DataFrame by 'event2' and 'day'
    grouped = butter_df.groupby(['event2', 'day'])

    # Iterate through each group with a progress bar
    for (event2, day), event_data in tqdm(grouped, desc="Processing groups"):
        # Extract required values
        rfid = event_data['RFID'].iloc[0]
        datetime = event_data['datetime'].iloc[0]
        event_no = f"{event2}-{day}"  # Combining event and day for a unique identifier
        bodyweight = return_bw_Ftot(event_data, var_threshold=0.00001, frame_length=200)

        #Find max_Ftot, the peak force at take-off. 
        max_Ftot = event_data['Ftotal_filt'].max()
        ninety9_perc_Ftot = np.percentile(event_data['Ftotal_filt'], 99) #June11
        max_Fx = event_data['Fx_filt'].max()
        max_Fy = event_data['Fy_filt'].max()
        max_Fz = event_data['Fz_filt'].max()
        
        #find the index of max_Ftot

        idx_max_Ftot = event_data['Ftotal_filt'].idxmax()

        # extract Fx, Fy, and Time at max_Ftot

        Fx_maxFtot = event_data.loc[idx_max_Ftot, 'Fx_filt']
        Fy_maxFtot = event_data.loc[idx_max_Ftot, 'Fy_filt']
        Fz_maxFtot = event_data.loc[idx_max_Ftot, 'Fz_filt'] #added June11-2025
        time_max_Ftot = event_data.loc[idx_max_Ftot, 'Time']

        # Find the index of max_Fx, max_Fy, and max_Fz
        
        idx_max_Fx = event_data['Fx_filt'].idxmax()
        idx_max_Fy = event_data['Fy_filt'].idxmax()
        idx_max_Fz = event_data['Fz_filt'].idxmax()

        #extract corresponding Time values 
        time_max_Fx = event_data.loc[idx_max_Fx, 'Time']
        time_max_Fy = event_data.loc[idx_max_Fy, 'Time']
        time_max_Fz = event_data.loc[idx_max_Fz, 'Time']
        
        
        #angle and time difference between time_max_Ftot and time_max_Fx/time_max_Fy/time_max_Fz can be calculated later. 
        

        # Create a new row as a dictionary (faster than creating a DataFrame for each row)
        new_row = {
            'RFID': rfid,
            'datetime': datetime,
            'event_no': event_no,
            'bodyweight': bodyweight,
            'max_Ftot': max_Ftot,
            'ninety9_perc_Ftot':ninety9_perc_Ftot,
            'time_max_Ftot': time_max_Ftot,
            'Fx_maxFtot': Fx_maxFtot,
            'Fy_maxFtot': Fy_maxFtot,
            'Fz_maxFtot': Fz_maxFtot, #added June11-2025
            'max_Fx': max_Fx,
            'time_max_Fx': time_max_Fx,
            'max_Fy': max_Fy,
            'time_max_Fy': time_max_Fy,
            'max_Fz': max_Fz, 
            'time_max_Fz': time_max_Fz

        }

        # Append the new row to the force_outcome DataFrame
        force_outcome = pd.concat([force_outcome, pd.DataFrame([new_row])], ignore_index=True)

    return force_outcome

def match_keys_preparation(F_df, RFID_match):
    # Initialize DataFrames
    F_df_uniq_ts = pd.DataFrame()
    RFID_match_uniq_ts = pd.DataFrame()

    # F_df preparation
    F_df_uniq_ts['hr'] = F_df['datetime'].dt.hour
    F_df_uniq_ts['min'] = F_df['datetime'].dt.minute
    F_df_uniq_ts['s'] = F_df['datetime'].dt.second

    # RFID_match preparation
    RFID_match_uniq_ts['hr'] = RFID_match['rfid_datetime'].dt.hour
    #used to be epoch_time_converted. changed to 'rfid_datetime' as 'epoch_time_converted'
    #in RFID logged file are sometimes unreliable. 'rfid_datetime' and 'datetime' is 1hr ahead
    #of local time. 
    RFID_match_uniq_ts['min'] = RFID_match['rfid_datetime'].dt.minute
    RFID_match_uniq_ts['s'] = RFID_match['rfid_datetime'].dt.second
    RFID_match_uniq_ts['rfid'] = RFID_match['id']
    RFID_match_uniq_ts['status'] = RFID_match['status']

    # Ensure valid DataFrames
    if F_df_uniq_ts.empty or RFID_match_uniq_ts.empty:
        print("Input DataFrames are empty or not properly populated.")
        return None, None

    # Unique combinations of hr, min, and s
    F_match_keys = F_df_uniq_ts[['hr', 'min', 's']].drop_duplicates()
    RFID_match_keys = RFID_match_uniq_ts[['hr', 'min', 's', 'rfid', 'status']].drop_duplicates()

    # Create a timestamp string column
    F_match_keys['timestamp'] = F_match_keys.apply(lambda row: f"{row['hr']:02}:{row['min']:02}:{row['s']:02}", axis=1)
    RFID_match_keys['timestamp'] = RFID_match_keys.apply(lambda row: f"{row['hr']:02}:{row['min']:02}:{row['s']:02}", axis=1)

    # Convert the string column to time format
    try:
        F_match_keys['timestamp'] = pd.to_datetime(F_match_keys['timestamp'], format='%H:%M:%S').dt.time
        RFID_match_keys['timestamp'] = pd.to_datetime(RFID_match_keys['timestamp'], format='%H:%M:%S').dt.time
    except ValueError as e:
        print(f"Error in timestamp conversion: {e}")
        return None, None

    # Initialize event counter and event list
    event_counter = 1
    events = [event_counter]

    # Iterate over rows to mark consecutive events
    for i in range(len(F_match_keys) - 1):
        current_time = pd.to_datetime(F_match_keys['timestamp'].iloc[i].strftime("%H:%M:%S"), format='%H:%M:%S')
        next_time = pd.to_datetime(F_match_keys['timestamp'].iloc[i + 1].strftime("%H:%M:%S"), format='%H:%M:%S')
        if next_time == current_time + timedelta(seconds=1):
            events.append(event_counter)
        else:
            event_counter += 1
            events.append(event_counter)

    # Add event column to F_match_keys
    F_match_keys['event'] = events

    # Reset index and make it a column named 'index'
    F_match_keys.reset_index(inplace=True)
    F_match_keys.rename(columns={'index': 'index'}, inplace=True)

    # Add RFID column as a match key
    F_match_keys['RFID'] = pd.NA

    return F_match_keys, RFID_match_keys


def match_keys_preparation_0216(F_df, RFID_match):

    F_df_uniq_ts = pd.DataFrame()
    RFID_match_uniq_ts = pd.DataFrame()

    #F_df preparation
    F_df_uniq_ts['hr'] = F_df['datetime'].dt.hour
    F_df_uniq_ts['min'] = F_df['datetime'].dt.minute
    F_df_uniq_ts['s'] = F_df['datetime'].dt.second

    #RFID_match preparation
    RFID_match_uniq_ts['hr'] = RFID_match['time'].dt.hour
    RFID_match_uniq_ts['min'] = RFID_match['time'].dt.minute
    RFID_match_uniq_ts['s'] = RFID_match['time'].dt.second
    RFID_match_uniq_ts['rfid'] = RFID_match['id']
    RFID_match_uniq_ts['status'] = RFID_match['status']

    #unique combinations of hr,min, and s 
    F_match_keys = F_df_uniq_ts[['hr', 'min','s']].drop_duplicates()
    RFID_match_keys = RFID_match_uniq_ts[['hr', 'min','s','rfid', 'status']].drop_duplicates()
    #print(f'RFID_match_keys: {RFID_match_keys.head(5)}')
    # Create a timestamp column from hr, min, s columns
    # need to go through the pain of making it a datetime format in case 59 + 1 becomes 00 in seconds

    F_match_keys['timestamp'] = F_match_keys.apply(lambda row: f"{row['hr']:02}:{row['min']:02}:{row['s']:02}", axis=1)
    RFID_match_keys['timestamp'] = RFID_match_keys.apply(lambda row: f"{row['hr']:02}:{row['min']:02}:{row['s']:02}", axis=1)
    

    # Convert the string 'timestamp' into full datetime objects
    F_match_keys['timestamp'] = pd.to_datetime(F_match_keys['timestamp'].astype(str), format='%H:%M:%S')
    RFID_match_keys['timestamp'] = pd.to_datetime(RFID_match_keys['timestamp'].astype(str), format='%H:%M:%S')

    # Initialize event counter and event list
    event_counter = 1
    events = [event_counter]

    # Iterate over rows to mark consecutive events
    for i in range(len(F_match_keys) - 1):
        current_time = F_match_keys['timestamp'].iloc[i]
        next_time = F_match_keys['timestamp'].iloc[i + 1]
        if next_time == current_time + timedelta(seconds=1):
            events.append(event_counter)
        else:
            event_counter += 1
            events.append(event_counter)

    # Add the event column to F_match_keys
    F_match_keys['event'] = events

    # Reset index and make it a column named 'index'
    F_match_keys.reset_index(inplace=True)
    F_match_keys.rename(columns={'index': 'index'}, inplace=True)
    
    #add RFID column as a matchkey
    F_match_keys['RFID'] =pd.NA

    return F_match_keys, RFID_match_keys


def match_visits_irfid(treatment_visits, irfid_filt, day0):
    """
    Matches rows between two DataFrames based on 'id' and time difference within 4 seconds.

    Parameters:
    treatment_visits (DataFrame): Contains RFID
    irfid_filt (DataFrame): Contains RFID filtered data with acc1-acc3.
    day0 (bool): If True, subtracts 1 hour from 'datetime' in treatment_visits.

    Returns:
    DataFrame: The resulting DataFrame with matched rows and combined columns by RFID.
    """
    # Ensure 'datetime' columns are in datetime format
    treatment_visits['datetime'] = pd.to_datetime(treatment_visits['datetime'])
    irfid_filt['datetime'] = pd.to_datetime(irfid_filt['datetime'])

    if day0:
        #-1 hr from visits RFID (turn off when not dealing with experimental day RFIDs)
        treatment_visits['datetime'] = treatment_visits['datetime'] - timedelta(hours =1 )
    
    # Rename columns for clarity
    treatment_visits = treatment_visits.rename(columns={'datetime': 'visit_datetime'})
    irfid_filt = irfid_filt.rename(columns={'datetime': 'irfid_datetime'})

    # Initialize a list to store matching rows
    matching_rows = []
    
    # Iterate through each row in 'treatment_visits'
    for _, fv_row in treatment_visits.iterrows():
        # Filter 'irfid_filt' for matching 'id' and time difference within 4 seconds
        matches = irfid_filt[
            (irfid_filt['id'] == fv_row['id']) &
            (abs((irfid_filt['irfid_datetime'] - fv_row['visit_datetime']).dt.total_seconds()) <= 4)
        ]
        
        # Append all columns from both rows to the result
        for _, match_row in matches.iterrows():
            combined_row = fv_row.to_dict()  # Add all columns from treatment_visit row
            combined_row.update(match_row.to_dict())  # Add all columns from irfid_filt row
            matching_rows.append(combined_row)
    
    # Convert the list of matching rows to a DataFrame
    matching_result = pd.DataFrame(matching_rows)
    
    return matching_result

def full_merge_Frfid(df, full_rfids):
    new = pd.merge(df, full_rfids, on = 'id', how = 'left')
    new.rename(columns={'force_datetime': 'datetime'}, inplace=True)
    new['datetime'] = pd.to_datetime(new['datetime'])
    print("Merged DataFrame shape:", new.shape)
    print("First 3 rows of the merged DataFrame:")
    print(new.head(3)) 
    return new

def full_merge_irfid(df, full_rfids):
    new = pd.merge(df, full_rfids, on='id', how='left')
    new.rename(columns={'irfid_datetime': 'datetime'}, inplace=True)
    new['datetime'] = pd.to_datetime(new['datetime'])
    print("Merged DataFrame shape:", new.shape)
    print("First 3 rows of the merged DataFrame:")
    print(new.head(3)) 

    return new

# Correct renaming of the column
# Convert the 'datetime' column to datetime format
# Print the first 3 rows and the shape of the DataFrame

def full_merge_irfid(df, full_rfids):
    new = pd.merge(df, full_rfids, on='id', how='left')
    new.rename(columns={'irfid_datetime': 'datetime'}, inplace=True)
    new['datetime'] = pd.to_datetime(new['datetime'])
    print("Merged DataFrame shape:", new.shape)
    print("First 3 rows of the merged DataFrame:")
    print(new.head(3)) 

    return new

# Correct renaming of the column
# Convert the 'datetime' column to datetime format
# Print the first 3 rows and the shape of the DataFrame

def match_visits_force(df, force_outcome, day0):
    # Ensure 'datetime' columns are in datetime format
    force_outcome['datetime'] = pd.to_datetime(force_outcome['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'])  # Rename and convert

    
    # Rename columns in force_outcome for consistency
    force_outcome = force_outcome.rename(columns={'RFID': 'id'})

    if day0:
        #-1 hr from visits RFID (turn off when not dealing with experimental day RFIDs)
        df['datetime'] = df['datetime'] - timedelta(hours =1 )
    
    #-1 hour to match the irfid time as SCARE.csv does not have epoch_time
    #df['datetime'] = df['datetime'] - pd.to_timedelta(1, unit='h')

    # Rename columns for clarity
    force_outcome = force_outcome.rename(columns={'datetime': 'force_datetime'})
    df = df.rename(columns={'datetime': 'visit_datetime'})

    print(f"force_outcome head: {force_outcome.head(3)}")
    print(f"visits head: {df.head(3)}")

    # Initialize a list to store matching rows
    matching_rows = []

    # Iterate through each row in 'df'
    for _, fv_row in df.iterrows():
        # Filter 'force_outcome' for matching 'id' and time difference within 3 seconds
        matches = force_outcome[
            (force_outcome['id'] == fv_row['id']) &
            (abs((force_outcome['force_datetime'] - fv_row['visit_datetime']).dt.total_seconds()) <= 4)
        ]

        # Append all columns from both rows to the result
        for _, match_row in matches.iterrows():
            combined_row = fv_row.to_dict()  # Add all columns from df row
            combined_row.update(match_row.to_dict())  # Add all columns from force_outcome row
            matching_rows.append(combined_row)

    # Convert the list of matching rows to a DataFrame
    F_df = pd.DataFrame(matching_rows)

    return F_df


def scare_ordering(df):
    """
    Adds two columns to the DataFrame:
    1. 'exp_order': The order of each day.
    2. 'prev_scares': Cumulative scares ('scare_outcome' == 1) before the current row on the same day and for each individual.

    Parameters:
    df (DataFrame): The input DataFrame containing 'datetime', 'scare_outcome', and 'id' columns.
    'datetime' column needs to be in 'datetime' format.

    Returns:
    DataFrame: The modified DataFrame with the additional columns.
    """
    # Ensure 'scare_outcome' is numeric
    df['scare_outcome'] = pd.to_numeric(df['scare_outcome'], errors='coerce')
    print(f"first line: {df['scare_outcome'].head(7)}")

    # Ensure 'id' is a string
    df['id'] = df['id'].astype(str)

    # Extract date from the 'datetime' column
    df['day'] = df['datetime'].dt.date

    # Initialize the 'exp_order' and 'prev_scares' column
    df['exp_order'] = pd.NA
    df['prev_scares'] = 0

    print(f"second line: {df[['scare_outcome', 'exp_order','prev_scares']].head(7)}")

    # Get the unique days in sorted order (to ensure consistent order)
    unique_days = sorted(df['day'].unique())

    # Assign 'exp_order' based on the index of the day in unique_days
    for idx, day in enumerate(unique_days, start=1):  # start=1 to make the order start at 1
        df.loc[df['day'] == day, 'exp_order'] = idx

    # Calculate cumulative scares before the current row ('prev_scares') within each day and for each individual
    df['prev_scares'] = (
        df.groupby(['day', 'id'])['scare_outcome']
        .apply(lambda x: x.shift().cumsum())
        .reset_index(level=['day', 'id'], drop=True)  # Align the result with the original DataFrame index
    )

    print(f"third line: {df[['scare_outcome', 'exp_order','prev_scares']].head(7)}")

    return df

def scare_control(df):

    #fill na with 0 
    df['scare_outcome'] = df['scare_outcome'].fillna(0)
    print(f"first line: {df['scare_outcome'].head(7)}")
    #make it numeric
    df['scare_outcome'] = pd.to_numeric(df['scare_outcome'],errors = 'coerce')
    print(f"second line: {df['scare_outcome'].head(7)}")

    #if there is any -1 on 'scare_outcome' column, change it to 1. 
    df['scare_outcome'] = df['scare_outcome'].replace(-1.0,1.0)
    print(f"third line: {df['scare_outcome'].head(7)}")
    #last na check
    df['scare_outcome']= df['scare_outcome'].fillna(0)
    print(f"fourth line: {df['scare_outcome'].head(7)}")

    return df