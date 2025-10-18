import pandas as pd
import numpy as np

def generate_job_data():
    """
    Generates a realistic job dataset based on the port layout and requirements.
    This creates a CSV file that the main app will use.
    """
    NUM_JOBS = 200 # Using a smaller number for faster web demo
    NUM_QCS = 8
    NUM_YARDS = 16 # A1, A2, B1, B2 ... H2

    # Define locations from the map (approximated from the GUI screenshot)
    # QC locations (x, y)
    qc_coords = {f'QC{i+1}': (3 + i*5, 3) for i in range(NUM_QCS)}
    
    # Yard locations (x, y) - A1, A2, B1, B2, etc.
    yard_ids = [f'{chr(65+i)}{j}' for i in range(NUM_QCS) for j in [1, 2]]
    yard_coords = {
        yard_ids[k]: (2 + (k//2)*5 + (k%2)*2, 13) for k in range(NUM_YARDS)
    }

    jobs = []
    for i in range(NUM_JOBS):
        job_type = np.random.choice(['DI', 'LO'])
        qc = f'QC{np.random.randint(1, NUM_QCS + 1)}'
        
        if job_type == 'DI': # Discharge: QC -> Yard
            pickup_loc = qc_coords[qc]
            default_yard = np.random.choice(list(yard_coords.keys()))
            drop_loc = yard_coords[default_yard]
            
            # Select 3 unique alternative yards
            alt_yards = np.random.choice(
                [y for y in yard_ids if y != default_yard], 3, replace=False
            ).tolist()

        else: # Load: Yard -> QC
            default_yard = np.random.choice(list(yard_coords.keys()))
            pickup_loc = yard_coords[default_yard]
            drop_loc = qc_coords[qc]
            alt_yards = [None, None, None] # No alternatives for LO jobs

        jobs.append({
            'JOB_ID': i,
            'JOB_TYPE': job_type,
            'QC_M': qc,
            'YARD_BLOCK': default_yard,
            'pickup_x': pickup_loc[0],
            'pickup_y': pickup_loc[1],
            'drop_x': drop_loc[0],
            'drop_y': drop_loc[1],
            'ALT_YARD_BLOCK_1': alt_yards[0],
            'ALT_YARD_BLOCK_2': alt_yards[1],
            'ALT_YARD_BLOCK_3': alt_yards[2],
            'handling_time_pickup': 120 if job_type == 'LO' else 300, # QC time for DI, Yard for LO
            'handling_time_dropoff': 300 if job_type == 'DI' else 120, # Yard time for DI, QC for LO
        })
        
    df = pd.DataFrame(jobs)
    df.to_csv("job_dataset.csv", index=False)
    return df, qc_coords, yard_coords

if __name__ == '__main__':
    generate_job_data()
    print("job_dataset.csv generated successfully.")
