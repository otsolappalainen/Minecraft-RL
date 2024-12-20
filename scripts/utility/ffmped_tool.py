import tkinter as tk
from tkinter import ttk, filedialog, messagebox  # Add messagebox
import ffmpeg
import os
import cv2
from PIL import Image, ImageTk
import subprocess
import re
import threading


def get_video_info(filepath):
        """Get video metadata using ffprobe"""
        try:
            probe = ffmpeg.probe(filepath)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            return {
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'bitrate': int(video_info.get('bit_rate', '2000000')),
                'duration': float(video_info.get('duration', '0'))
            }
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read video info: {str(e)}")
            return {
                'width': 1920,
                'height': 1080,
                'bitrate': 2000000,
                'duration': 0
            }

class VideoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Editor")
        self.filepath = None
        self.video = None
        self.total_frames = 0
        self.current_frame = 0
        
        # Create UI
        self.create_widgets()
        

    
    def create_widgets(self):
        # File selection
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(file_frame, text="Select Video", command=self.select_file).pack()
        
        # Video settings
        self.settings_frame = ttk.LabelFrame(self.root, text="Video Settings", padding=10)
        self.settings_frame.pack(fill="x", padx=5, pady=5)
        
        # Frame selection
        self.frame_frame = ttk.LabelFrame(self.root, text="Frame Selection", padding=10)
        self.frame_frame.pack(fill="x", padx=5, pady=5)
        
        # Preview frame with side-by-side previews
        self.preview_frame = ttk.LabelFrame(self.root, text="Preview", padding=10)
        self.preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create frame for side-by-side previews
        preview_container = ttk.Frame(self.preview_frame)
        preview_container.pack(fill="both", expand=True)
        
        # Start frame preview
        start_preview = ttk.Frame(preview_container)
        start_preview.pack(side="left", fill="both", expand=True)
        ttk.Label(start_preview, text="Start Frame").pack()
        self.start_preview_label = ttk.Label(start_preview)
        self.start_preview_label.pack()
        
        # End frame preview
        end_preview = ttk.Frame(preview_container)
        end_preview.pack(side="right", fill="both", expand=True)
        ttk.Label(end_preview, text="End Frame").pack()
        self.end_preview_label = ttk.Label(end_preview)
        self.end_preview_label.pack()
        
        # Progress frame
        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.pack(fill="x", padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x")
        self.status_label = ttk.Label(self.progress_frame, text="")
        self.status_label.pack()
        
        # Export button
        ttk.Button(self.root, text="Export", command=self.export_video).pack(pady=10)
        
    def select_file(self):
        self.filepath = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if not self.filepath:
            return
            
        # Get video info using global function
        info = get_video_info(self.filepath)
        self.setup_video_settings(info)
        self.setup_frame_controls()
        
    def setup_video_settings(self, info):
        # Clear previous widgets
        for widget in self.settings_frame.winfo_children():
            widget.destroy()
            
        # Resolution
        res_frame = ttk.Frame(self.settings_frame)
        res_frame.pack(fill="x")
        
        ttk.Label(res_frame, text="Width:").pack(side="left")
        self.width_var = tk.StringVar(value=str(info['width']))
        ttk.Entry(res_frame, textvariable=self.width_var).pack(side="left")
        
        ttk.Label(res_frame, text="Height:").pack(side="left")
        self.height_var = tk.StringVar(value=str(info['height']))
        ttk.Entry(res_frame, textvariable=self.height_var).pack(side="left")
        
        # Bitrate
        bit_frame = ttk.Frame(self.settings_frame)
        bit_frame.pack(fill="x")
        
        ttk.Label(bit_frame, text="Bitrate (kbps):").pack(side="left")
        self.bitrate_var = tk.StringVar(value=str(int(info['bitrate']/1000)))
        ttk.Entry(bit_frame, textvariable=self.bitrate_var).pack(side="left")
        
    def setup_frame_controls(self):
        # Clear previous widgets
        for widget in self.frame_frame.winfo_children():
            widget.destroy()
            
        # Open video
        self.video = cv2.VideoCapture(self.filepath)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Start frame
        ttk.Label(self.frame_frame, text="Start Frame:").pack()
        self.start_var = tk.IntVar(value=0)
        self.start_slider = ttk.Scale(
            self.frame_frame, 
            from_=0, 
            to=self.total_frames,
            variable=self.start_var,
            orient="horizontal",
            command=self.update_preview
        )
        self.start_slider.pack(fill="x")
        
        # End frame
        ttk.Label(self.frame_frame, text="End Frame:").pack()
        self.end_var = tk.IntVar(value=self.total_frames)
        self.end_slider = ttk.Scale(
            self.frame_frame,
            from_=0,
            to=self.total_frames,
            variable=self.end_var,
            orient="horizontal",
            command=self.update_preview
        )
        self.end_slider.pack(fill="x")
        
        self.update_preview()
        
    def update_preview(self, *args):
        if not self.video:
            return
            
        # Update start frame preview
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.start_var.get())
        ret, start_frame = self.video.read()
        if ret:
            self._update_preview_image(start_frame, self.start_preview_label)
            
        # Update end frame preview    
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.end_var.get())
        ret, end_frame = self.video.read()
        if ret:
            self._update_preview_image(end_frame, self.end_preview_label)
            
    def _update_preview_image(self, frame, label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        
        # Resize to fit preview (smaller for side-by-side)
        preview_width = 320  # Reduced from 480 for side-by-side
        ratio = preview_width / frame.width
        preview_height = int(frame.height * ratio)
        frame = frame.resize((preview_width, preview_height))
        
        photo = ImageTk.PhotoImage(frame)
        label.configure(image=photo)
        label.image = photo  # Keep reference
        
    def export_video(self):
        if not self.filepath:
            return
            
        output_path = os.path.splitext(self.filepath)[0] + "_trimmed.mp4"
        
        # Prepare ffmpeg command
        duration = (self.end_var.get() - self.start_var.get()) / self.video.get(cv2.CAP_PROP_FPS)
        cmd = [
            'ffmpeg', '-y',
            '-i', self.filepath,
            '-ss', str(self.start_var.get() / self.video.get(cv2.CAP_PROP_FPS)),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-b:v', f"{self.bitrate_var.get()}k",
            '-vf', f'scale={self.width_var.get()}:{self.height_var.get()}',
            output_path
        ]
        
        def run_export():
            try:
                # Reset progress
                self.progress_bar['value'] = 0
                self.status_label['text'] = "Starting export..."
                
                # Run ffmpeg with progress monitoring
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor progress
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                        
                    # Try to parse progress
                    time_match = re.search(r'time=(\d+):(\d+):(\d+)', line)
                    if time_match:
                        hrs, mins, secs = map(int, time_match.groups())
                        current_time = hrs * 3600 + mins * 60 + secs
                        progress = min(100, int(100 * current_time / duration))
                        
                        # Update UI in main thread
                        self.root.after(0, self._update_progress, progress)
                
                # Check result
                if process.wait() == 0:
                    self.root.after(0, self._export_complete, output_path)
                else:
                    error = process.stderr.read()
                    self.root.after(0, self._export_error, error)
                    
            except Exception as e:
                self.root.after(0, self._export_error, str(e))
        
        # Run export in background
        threading.Thread(target=run_export, daemon=True).start()
    
    def _update_progress(self, progress):
        self.progress_bar['value'] = progress
        self.status_label['text'] = f"Exporting: {progress}%"
        
    def _export_complete(self, output_path):
        self.progress_bar['value'] = 100
        self.status_label['text'] = "Export complete!"
        messagebox.showinfo("Success", f"Video exported to: {output_path}")
        
    def _export_error(self, error):
        self.status_label['text'] = "Export failed!"
        messagebox.showerror("Error", str(error))
            
    def __del__(self):
        if self.video:
            self.video.release()

def main():
    root = tk.Tk()
    app = VideoEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()