import pyxdf

# Load the recorded .xdf file
filepath = 'C:\\Users\\dorus\\Desktop\\lsl tes blink\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'

streams, fileheader = pyxdf.load_xdf(filepath)

# Print stream names and data
for stream in streams:
    print(f"Stream name: {stream['info']['name'][0]}")
    print(f"Stream data: {stream['time_series']}")