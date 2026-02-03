"""
Garmin FIT File Analyzer - v2.1 (Stable)
Features: Matplotlib Trends, Fixed Info Modal, Tabbed Interface.
"""

import customtkinter as ctk
from tkinter import filedialog
import threading
import sys
import os
import csv
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- GRAPHING IMPORTS (Crucial for the graph to work) ---
import matplotlib
matplotlib.use("TkAgg") # Force TkAgg backend for cross-platform stability
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# Ensure imports work for both dev and compiled app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from garmin_analyzer.analyzer import FitAnalyzer

# Enforce Dark Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class InfoModal(ctk.CTkToplevel):
    """Modern 'How-To' Modal Dialog."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("How to Use")
        self.geometry("420x380") # Slightly taller/wider for safety
        self.resizable(False, False)
        self.attributes("-topmost", True)
        
        # Center title
        lbl = ctk.CTkLabel(
            self, 
            text="Quick Start Guide", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        lbl.pack(pady=(25, 15))
        
        # Steps with wraplength to prevent cutoff
        steps = (
            "1. Log in to Garmin Connect on your web browser.\n\n"
            "2. Download your activities as 'Original' (.FIT) files.\n\n"
            "3. Move those .fit files into a single folder on your computer.\n\n"
            "4. Click 'Select Folder' here to analyze them!"
        )
        
        step_lbl = ctk.CTkLabel(
            self, 
            text=steps, 
            justify="left", 
            font=ctk.CTkFont(size=14), 
            text_color="#E0E0E0",
            wraplength=350 # Fixes the text cutoff issue
        )
        step_lbl.pack(pady=10, padx=30)
        
        # Close Button
        btn = ctk.CTkButton(
            self, 
            text="Got it!", 
            width=120,
            height=35,
            fg_color="#2FA572",
            hover_color="#25855A", 
            command=self.destroy
        )
        btn.pack(pady=25)

class GarminAnalyzerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Garmin FIT Analyzer")
        self.geometry("1100x800")
        
        # Layout: Sidebar + Main
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.analysis_results: List[Dict[str, Any]] = []
        self.last_folder_path: Optional[str] = None
        
        self._create_widgets()

    def _create_widgets(self):
        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        self.logo = ctk.CTkLabel(self.sidebar, text="RUN ANALYZER", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=(50, 40))
        
        self.btn_select = ctk.CTkButton(
            self.sidebar, text="📂  Select Folder", height=45, 
            font=ctk.CTkFont(size=15, weight="bold"), command=self.select_folder
        )
        self.btn_select.grid(row=1, column=0, padx=30, pady=10)
        
        # Info Button
        self.btn_info = ctk.CTkButton(
            self.sidebar, text="ⓘ  How it works", height=30,
            fg_color="transparent", border_width=1, text_color="gray70",
            command=self.open_info
        )
        self.btn_info.grid(row=2, column=0, padx=30, pady=10)

        # Version
        self.lbl_ver = ctk.CTkLabel(self.sidebar, text="v2.1", text_color="gray40")
        self.lbl_ver.grid(row=5, column=0, padx=20, pady=20, sticky="s")

        # --- MAIN AREA (Tabs) ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.lbl_status = ctk.CTkLabel(self.header, text="Dashboard", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_status.pack(side="left")

        # Tabs: Report vs Visuals
        self.tabs = ctk.CTkTabview(self.main_frame)
        self.tabs.grid(row=1, column=0, sticky="nsew")
        self.tab_report = self.tabs.add("📝 Text Report")
        self.tab_visuals = self.tabs.add("📈 Trend Graph")
        
        # Tab 1: Text Output
        self.tab_report.grid_columnconfigure(0, weight=1)
        self.tab_report.grid_rowconfigure(0, weight=1)
        self.txt_output = ctk.CTkTextbox(self.tab_report, font=("Consolas", 14), fg_color="#181818")
        self.txt_output.grid(row=0, column=0, sticky="nsew")
        self._emit("👋 Welcome!\n\nClick 'Select Folder' to analyze your runs.\nClick 'How it works' if you are new here.")

        # Tab 2: Visuals (Empty initially)
        self.tab_visuals.grid_columnconfigure(0, weight=1)
        self.tab_visuals.grid_rowconfigure(0, weight=1)
        self.lbl_no_data = ctk.CTkLabel(self.tab_visuals, text="No data loaded yet.", text_color="gray")
        self.lbl_no_data.grid(row=0, column=0)
        self.canvas = None

        # --- ACTION BAR ---
        self.actions = ctk.CTkFrame(self.main_frame, fg_color="transparent", height=50)
        self.actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        self.btn_copy = ctk.CTkButton(
            self.actions, text="📋 Copy for LLM", width=160, height=35,
            fg_color="#2FA572", font=ctk.CTkFont(weight="bold"), command=self.copy_all
        )
        self.btn_copy.pack(side="right")

    # --- LOGIC ---
    def open_info(self):
        InfoModal(self)

    def select_folder(self):
        folder = filedialog.askdirectory(mustexist=True)
        if folder:
            self.txt_output.delete("1.0", "end")
            self._emit(f"📂 Analyzing: {folder}...\n")
            self.lbl_status.configure(text="Analyzing...")
            self.btn_select.configure(state="disabled")
            
            # Run analysis in background
            thread = threading.Thread(target=self._run_analysis, args=(folder,))
            thread.daemon = True
            thread.start()

    def _run_analysis(self, folder):
        analyzer = FitAnalyzer(output_callback=lambda t: self.after(0, lambda: self._emit(t)))
        try:
            results = analyzer.analyze_folder(folder)
            self.analysis_results = results
            self.after(0, lambda: self._on_complete(len(results)))
        except Exception as e:
            self.after(0, lambda: self._emit(f"Error: {e}"))
            self.after(0, lambda: self._on_complete(0))

    def _on_complete(self, count):
        self.btn_select.configure(state="normal")
        self.lbl_status.configure(text=f"Dashboard ({count} runs)")
        if count > 0:
            self._update_graph() # This triggers the graph draw!
            self.show_toast(f"Success! Processed {count} runs.")

    def _emit(self, text):
        self.txt_output.insert("end", text + "\n")
        self.txt_output.see("end")

    def _update_graph(self):
        """Draws the Matplotlib graph in the Visuals tab."""
        # Clear previous canvas if it exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        # Hide the "No data" label
        if self.lbl_no_data.winfo_exists():
            self.lbl_no_data.destroy()
        
        if not self.analysis_results:
            return

        # Prepare Data
        dates = [r['date'] for r in self.analysis_results]
        decoupling = [r.get('decoupling', 0) for r in self.analysis_results]
        
        # Setup Figure (Dark Theme)
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#242424') # Match App BG
        ax.set_facecolor('#242424')
        
        # Plot Logic: Dot vs Line
        if len(dates) == 1:
            # Single Dot - Add Annotation
            ax.scatter(dates, decoupling, color='#3B8ED0', s=100, zorder=5)
            ax.text(dates[0], decoupling[0] + 0.5, "Only 1 Run", ha='center', color='white')
            ax.set_title("Aerobic Decoupling (Single Run)", color='white', pad=15)
        else:
            # Trend Line
            ax.plot(dates, decoupling, marker='o', linestyle='-', color='#3B8ED0', linewidth=2, markersize=6)
            ax.set_title("Aerobic Decoupling Trend", color='white', pad=15)

        # Formatting
        ax.axhline(y=5.0, color='#2FA572', linestyle='--', alpha=0.5, label='Good (<5%)') # Threshold
        ax.set_ylabel("Decoupling (%)", color='gray')
        ax.grid(True, color='#404040', linestyle='--', alpha=0.5)
        ax.tick_params(colors='gray')
        
        # Date Formatting
        if len(dates) > 1:
            fig.autofmt_xdate()
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.tab_visuals)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill="both", padx=10, pady=10)

    def show_toast(self, msg):
        toast = ctk.CTkFrame(self, fg_color="#2FA572", height=40, corner_radius=20)
        toast.place(relx=0.5, rely=0.9, anchor="center")
        ctk.CTkLabel(toast, text=msg, text_color="white", padx=20, pady=5).pack()
        self.after(3000, toast.destroy)

    def copy_all(self):
        text = self.txt_output.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.show_toast("Report copied!")

def main():
    app = GarminAnalyzerGUI()
    app.mainloop()

if __name__ == "__main__":
    main()