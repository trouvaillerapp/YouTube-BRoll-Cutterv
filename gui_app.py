#!/usr/bin/env python3
"""
GUI Application for YouTube B-Roll Cutter
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
import json
import webbrowser

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broll_cutter import YouTubeBRollCutter

class BRollCutterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 YouTube B-Roll Cutter")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Application state
        self.cutter = None
        self.is_processing = False
        self.progress_queue = queue.Queue()
        
        # Setup UI
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        
        # Start progress monitor
        self.monitor_progress()
        
    def setup_styles(self):
        """Setup custom styles for the UI"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure("Action.TButton", padding=(10, 5))
        style.configure("Primary.TButton", padding=(15, 8))
        
    def create_widgets(self):
        """Create all UI widgets"""
        
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # Title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="🎬 YouTube B-Roll Cutter",
            font=("Arial", 16, "bold")
        )
        
        # URL input section
        self.url_frame = ttk.LabelFrame(self.main_frame, text="📹 Video URLs", padding="10")
        
        self.url_text = scrolledtext.ScrolledText(
            self.url_frame, 
            height=4, 
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.url_text.insert('1.0', "Enter YouTube URLs here (one per line):\nhttps://youtube.com/watch?v=...")
        
        # Settings section
        self.settings_frame = ttk.LabelFrame(self.main_frame, text="⚙️ Settings", padding="10")
        
        # Output directory
        ttk.Label(self.settings_frame, text="Output Directory:").grid(row=0, column=0, sticky="w", pady=2)
        self.output_var = tk.StringVar(value="./extracted_clips")
        self.output_entry = ttk.Entry(self.settings_frame, textvariable=self.output_var, width=40)
        self.browse_button = ttk.Button(self.settings_frame, text="Browse", command=self.browse_output_dir)
        
        # Quality selection
        ttk.Label(self.settings_frame, text="Quality:").grid(row=1, column=0, sticky="w", pady=2)
        self.quality_var = tk.StringVar(value="720p")
        self.quality_combo = ttk.Combobox(
            self.settings_frame, 
            textvariable=self.quality_var,
            values=["480p", "720p", "1080p", "4k"],
            state="readonly",
            width=10
        )
        
        # Clip duration
        ttk.Label(self.settings_frame, text="Clip Duration (seconds):").grid(row=2, column=0, sticky="w", pady=2)
        self.duration_var = tk.DoubleVar(value=8.0)
        self.duration_spin = ttk.Spinbox(
            self.settings_frame,
            from_=3.0, to=30.0, increment=1.0,
            textvariable=self.duration_var,
            width=10
        )
        
        # Max clips per video
        ttk.Label(self.settings_frame, text="Max Clips per Video:").grid(row=3, column=0, sticky="w", pady=2)
        self.max_clips_var = tk.IntVar(value=5)
        self.max_clips_spin = ttk.Spinbox(
            self.settings_frame,
            from_=1, to=20, increment=1,
            textvariable=self.max_clips_var,
            width=10
        )
        
        # Advanced options
        self.watermark_var = tk.BooleanVar(value=True)
        self.watermark_check = ttk.Checkbutton(
            self.settings_frame,
            text="Remove watermarks",
            variable=self.watermark_var
        )
        
        self.enhance_var = tk.BooleanVar(value=False)
        self.enhance_check = ttk.Checkbutton(
            self.settings_frame,
            text="Enhance video quality",
            variable=self.enhance_var
        )
        
        # Scene detection settings
        ttk.Label(self.settings_frame, text="Scene Detection Sensitivity:").grid(row=6, column=0, sticky="w", pady=2)
        self.sensitivity_var = tk.DoubleVar(value=0.3)
        self.sensitivity_scale = ttk.Scale(
            self.settings_frame,
            from_=0.1, to=1.0,
            orient="horizontal",
            variable=self.sensitivity_var,
            length=200
        )
        self.sensitivity_label = ttk.Label(self.settings_frame, text="0.3")
        
        # Update sensitivity label
        self.sensitivity_var.trace_add("write", self.update_sensitivity_label)
        
        # Action buttons
        self.action_frame = ttk.Frame(self.main_frame)
        
        self.preview_button = ttk.Button(
            self.action_frame,
            text="👁️ Preview Scenes",
            command=self.preview_scenes,
            style="Action.TButton"
        )
        
        self.process_button = ttk.Button(
            self.action_frame,
            text="🎬 Extract B-Roll",
            command=self.start_processing,
            style="Primary.TButton"
        )
        
        self.stop_button = ttk.Button(
            self.action_frame,
            text="⏹️ Stop",
            command=self.stop_processing,
            style="Action.TButton",
            state="disabled"
        )
        
        # Progress section
        self.progress_frame = ttk.LabelFrame(self.main_frame, text="📊 Progress", padding="10")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.progress_frame, textvariable=self.status_var)
        
        # Results section
        self.results_frame = ttk.LabelFrame(self.main_frame, text="📁 Results", padding="10")
        
        self.results_text = scrolledtext.ScrolledText(
            self.results_frame,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        
        self.open_folder_button = ttk.Button(
            self.results_frame,
            text="📂 Open Output Folder",
            command=self.open_output_folder
        )
        
        self.clear_results_button = ttk.Button(
            self.results_frame,
            text="🗑️ Clear Results",
            command=self.clear_results
        )
        
        # Menu bar
        self.create_menu()
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Settings", command=self.save_settings)
        file_menu.add_command(label="Load Settings", command=self.load_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Usage Guide", command=self.show_usage_guide)
    
    def setup_layout(self):
        """Setup widget layout"""
        
        # Main frame
        self.main_frame.pack(fill="both", expand=True)
        
        # Title
        self.title_label.pack(pady=(0, 10))
        
        # URL input
        self.url_frame.pack(fill="x", pady=(0, 10))
        self.url_text.pack(fill="x")
        
        # Settings
        self.settings_frame.pack(fill="x", pady=(0, 10))
        
        # Settings layout
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        self.browse_button.grid(row=0, column=2, padx=(5, 0))
        
        self.quality_combo.grid(row=1, column=1, sticky="w", padx=(5, 0))
        self.duration_spin.grid(row=2, column=1, sticky="w", padx=(5, 0))
        self.max_clips_spin.grid(row=3, column=1, sticky="w", padx=(5, 0))
        
        self.watermark_check.grid(row=4, column=0, columnspan=3, sticky="w", pady=5)
        self.enhance_check.grid(row=5, column=0, columnspan=3, sticky="w")
        
        self.sensitivity_scale.grid(row=6, column=1, sticky="ew", padx=(5, 5))
        self.sensitivity_label.grid(row=6, column=2, padx=(5, 0))
        
        # Configure grid weights
        self.settings_frame.columnconfigure(1, weight=1)
        
        # Action buttons
        self.action_frame.pack(fill="x", pady=(0, 10))
        self.preview_button.pack(side="left", padx=(0, 10))
        self.process_button.pack(side="left", padx=(0, 10))
        self.stop_button.pack(side="left")
        
        # Progress
        self.progress_frame.pack(fill="x", pady=(0, 10))
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.status_label.pack()
        
        # Results
        self.results_frame.pack(fill="both", expand=True)
        self.results_text.pack(fill="both", expand=True, pady=(0, 10))
        
        result_buttons_frame = ttk.Frame(self.results_frame)
        result_buttons_frame.pack(fill="x")
        self.open_folder_button.pack(side="left", padx=(0, 10))
        self.clear_results_button.pack(side="left")
    
    def update_sensitivity_label(self, *args):
        """Update sensitivity label"""
        value = self.sensitivity_var.get()
        self.sensitivity_label.config(text=f"{value:.1f}")
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_var.get()
        )
        if directory:
            self.output_var.set(directory)
    
    def get_urls(self):
        """Get URLs from text widget"""
        content = self.url_text.get("1.0", tk.END).strip()
        urls = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line and line.startswith('http'):
                urls.append(line)
        
        return urls
    
    def progress_callback(self, info):
        """Progress callback for processing"""
        self.progress_queue.put(info)
    
    def monitor_progress(self):
        """Monitor progress queue and update UI"""
        try:
            while True:
                info = self.progress_queue.get_nowait()
                
                # Update progress bar
                progress = info.get('progress', 0)
                self.progress_var.set(progress)
                
                # Update status
                message = info.get('message', '')
                stage = info.get('stage', '')
                if stage:
                    message = f"[{stage}] {message}"
                
                self.status_var.set(message)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.monitor_progress)
    
    def log_result(self, message):
        """Log result to results text widget"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def preview_scenes(self):
        """Preview scenes for the first URL"""
        urls = self.get_urls()
        if not urls:
            messagebox.showwarning("No URLs", "Please enter at least one YouTube URL")
            return
        
        def preview_thread():
            try:
                self.status_var.set("Initializing preview...")
                
                cutter = YouTubeBRollCutter(
                    scene_threshold=self.sensitivity_var.get(),
                    progress_callback=self.progress_callback
                )
                
                url = urls[0]
                self.log_result(f"🔍 Previewing scenes for: {url}")
                
                # Create preview directory
                preview_dir = Path(self.output_var.get()) / "scene_previews"
                preview_dir.mkdir(parents=True, exist_ok=True)
                
                scenes = cutter.preview_scenes(url, str(preview_dir))
                
                self.log_result(f"📋 Detected {len(scenes)} scenes:")
                for i, scene in enumerate(scenes):
                    duration = scene['end_time'] - scene['start_time']
                    self.log_result(f"   Scene {i+1}: {duration:.1f}s - {scene['scene_type']} (confidence: {scene['confidence']:.2f})")
                
                self.log_result(f"💡 Preview images saved to: {preview_dir}")
                self.status_var.set("Preview completed")
                
            except Exception as e:
                self.log_result(f"❌ Preview failed: {str(e)}")
                self.status_var.set("Preview failed")
        
        # Run in separate thread
        thread = threading.Thread(target=preview_thread, daemon=True)
        thread.start()
    
    def start_processing(self):
        """Start B-roll extraction"""
        urls = self.get_urls()
        if not urls:
            messagebox.showwarning("No URLs", "Please enter at least one YouTube URL")
            return
        
        # Disable buttons
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.is_processing = True
        
        def processing_thread():
            try:
                # Initialize cutter with current settings
                self.cutter = YouTubeBRollCutter(
                    output_dir=self.output_var.get(),
                    clip_duration=self.duration_var.get(),
                    quality=self.quality_var.get(),
                    scene_threshold=self.sensitivity_var.get(),
                    remove_watermarks=self.watermark_var.get(),
                    enhance_video=self.enhance_var.get(),
                    max_clips_per_video=self.max_clips_var.get(),
                    progress_callback=self.progress_callback
                )
                
                self.log_result(f"🚀 Starting B-roll extraction for {len(urls)} video(s)")
                
                total_clips = 0
                
                for i, url in enumerate(urls):
                    if not self.is_processing:
                        break
                    
                    self.log_result(f"\n📹 Processing video {i+1}/{len(urls)}: {url}")
                    
                    try:
                        clips = self.cutter.extract_broll(url)
                        
                        if clips:
                            self.log_result(f"✅ Extracted {len(clips)} clips:")
                            for j, clip_path in enumerate(clips):
                                self.log_result(f"   {j+1}. {Path(clip_path).name}")
                            total_clips += len(clips)
                        else:
                            self.log_result("⚠️  No clips extracted from this video")
                    
                    except Exception as e:
                        self.log_result(f"❌ Error processing video: {str(e)}")
                
                # Show final statistics
                if self.cutter:
                    stats = self.cutter.get_processing_stats()
                    self.log_result(f"\n📊 Processing completed!")
                    self.log_result(f"   Videos processed: {stats['videos_processed']}")
                    self.log_result(f"   Total clips extracted: {stats['clips_extracted']}")
                    self.log_result(f"   Total duration: {stats['total_clips_duration']:.1f}s")
                
                self.status_var.set("Processing completed")
                
            except Exception as e:
                self.log_result(f"❌ Processing failed: {str(e)}")
                self.status_var.set("Processing failed")
            
            finally:
                # Re-enable buttons
                self.process_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.is_processing = False
                self.progress_var.set(0)
        
        # Run in separate thread
        thread = threading.Thread(target=processing_thread, daemon=True)
        thread.start()
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        self.status_var.set("Stopping...")
        self.log_result("⏹️  Processing stopped by user")
    
    def open_output_folder(self):
        """Open output folder in file manager"""
        output_dir = Path(self.output_var.get())
        if output_dir.exists():
            if sys.platform == "darwin":  # macOS
                os.system(f"open '{output_dir}'")
            elif sys.platform == "win32":  # Windows
                os.system(f"explorer '{output_dir}'")
            else:  # Linux
                os.system(f"xdg-open '{output_dir}'")
        else:
            messagebox.showwarning("Folder Not Found", f"Output folder does not exist: {output_dir}")
    
    def clear_results(self):
        """Clear results text"""
        self.results_text.delete("1.0", tk.END)
    
    def save_settings(self):
        """Save current settings to file"""
        settings = {
            'output_dir': self.output_var.get(),
            'quality': self.quality_var.get(),
            'clip_duration': self.duration_var.get(),
            'max_clips': self.max_clips_var.get(),
            'remove_watermarks': self.watermark_var.get(),
            'enhance_video': self.enhance_var.get(),
            'scene_sensitivity': self.sensitivity_var.get()
        }
        
        file_path = filedialog.asksaveasfilename(
            title="Save Settings",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=2)
            messagebox.showinfo("Settings Saved", f"Settings saved to {file_path}")
    
    def load_settings(self):
        """Load settings from file"""
        file_path = filedialog.askopenfilename(
            title="Load Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                # Apply settings
                self.output_var.set(settings.get('output_dir', './extracted_clips'))
                self.quality_var.set(settings.get('quality', '720p'))
                self.duration_var.set(settings.get('clip_duration', 8.0))
                self.max_clips_var.set(settings.get('max_clips', 5))
                self.watermark_var.set(settings.get('remove_watermarks', True))
                self.enhance_var.set(settings.get('enhance_video', False))
                self.sensitivity_var.set(settings.get('scene_sensitivity', 0.3))
                
                messagebox.showinfo("Settings Loaded", f"Settings loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load settings: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """🎬 YouTube B-Roll Cutter v1.0.0

An AI-powered tool for extracting high-quality B-roll clips from YouTube videos.

Features:
• Intelligent scene detection
• Automatic watermark removal  
• Video quality enhancement
• Batch processing
• Custom clip durations
• Multiple output formats

Built with Python, OpenCV, and FFmpeg.

© 2024 - Open Source (MIT License)"""
        
        messagebox.showinfo("About", about_text)
    
    def show_usage_guide(self):
        """Show usage guide"""
        guide_text = """📖 Usage Guide

1. Enter YouTube URLs (one per line) in the URL box
2. Configure settings:
   - Output Directory: Where clips will be saved
   - Quality: Video quality (720p recommended)
   - Clip Duration: Length of each extracted clip
   - Max Clips: Maximum clips per video
3. Enable options:
   - Remove watermarks: Automatically detect and remove watermarks
   - Enhance video: Apply color correction and sharpening
4. Adjust scene sensitivity (0.1 = more scenes, 1.0 = fewer scenes)
5. Click "Preview Scenes" to see detected scenes first
6. Click "Extract B-Roll" to start processing
7. Monitor progress and check results
8. Open output folder to view extracted clips

Tips:
• Use preview mode to test settings before processing
• Higher quality settings take longer but produce better results
• Save/load settings for different types of content
• Check the results panel for detailed processing information"""
        
        # Create a new window for the guide
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Usage Guide")
        guide_window.geometry("600x500")
        
        text_widget = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", guide_text)
        text_widget.config(state="disabled")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = BRollCutterGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")

if __name__ == "__main__":
    main()