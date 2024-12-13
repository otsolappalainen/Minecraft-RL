import tkinter as tk
from tkinter import filedialog
import ffmpeg
import os

def get_video_info(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    
    # Get basic info
    width = int(video_info['width'])
    height = int(video_info['height'])
    bitrate = int(probe['format']['bit_rate']) if 'bit_rate' in probe['format'] else 0
    filesize = os.path.getsize(filename) / (1024*1024)  # Convert to MB
    
    return {
        'width': width,
        'height': height,
        'bitrate': bitrate,
        'filesize': filesize
    }

def main():
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    filepath = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("MP4 files", "*.mp4")]
    )
    
    if not filepath:
        print("No file selected. Exiting...")
        return

    # Get current video info
    info = get_video_info(filepath)
    
    print("\nCurrent video properties:")
    print(f"Resolution: {info['width']}x{info['height']}")
    print(f"Bitrate: {info['bitrate']/1000:.0f} kbps")
    print(f"File size: {info['filesize']:.1f} MB")

    # Ask for changes
    print("\nEnter new values (or press Enter to keep current):")
    new_width = input(f"New width ({info['width']}): ") or info['width']
    new_height = input(f"New height ({info['height']}): ") or info['height']
    new_bitrate = input(f"New bitrate in kbps ({info['bitrate']/1000:.0f}): ") or info['bitrate']/1000

    # Prepare output filename
    output_path = os.path.splitext(filepath)[0] + "_converted.mp4"

    # Build ffmpeg command
    stream = ffmpeg.input(filepath)
    stream = ffmpeg.output(stream, output_path,
                         video_bitrate=f"{int(float(new_bitrate))}k",
                         s=f"{new_width}x{new_height}")
    
    print("\nProcessing... Please wait.")
    
    try:
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"\nConversion complete! Saved as: {output_path}")
    except ffmpeg.Error as e:
        print("An error occurred:", e.stderr.decode())

if __name__ == "__main__":
    main()