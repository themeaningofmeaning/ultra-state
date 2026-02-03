import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .analyzer import FitAnalyzer

# Set theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class InfoModal(ctk.CTkToplevel):
    """Pop-up window to explain the graph metrics."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("How to Read This Graph")
        self.geometry("600x650") 
        self.attributes("-topmost", True) 

        # Title
        title = ctk.CTkLabel(self, text="What do the dots mean?", font=("Arial", 22, "bold"), text_color="#2CC985")
        title.pack(pady=(20, 10))

        # Content - Using User's Custom Text
        text = """
1. 🟢 Green: High Quality (Fast & Stable)
   • High Efficiency + Low Decoupling.
   • You were fast and your heart rate was stable.
   • Verdict: Race Ready.

2. 🟡 Yellow: Maintenance Quality (Slow & Stable for RECOVERY / BASE)
   • Lower Efficiency + Low Decoupling.
   • You ran slower/easier, and your heart rate stayed calm.
   • Verdict: Good aerobic maintenance run.

3. 🔴 Red: "Expensive" Quality (Fast but Unstable)
   • High Efficiency + High Decoupling.
   • You ran fast, but your heart rate drifted up significantly (>5%).
   • Verdict: Good speed, but lacks endurance (or dehydrated).

4. ⚫ Black: The Bonk (Slow & Struggling)
   • Low Efficiency + High Decoupling.
   • You were slow AND your heart rate skyrocketed.
   • Verdict: Fatigue, illness, or bad day. Rest up.

--------------------------------------------------

👣 CADENCE (Steps Per Minute)
   • 170-180 spm (Green Band): The efficient goal zone.
   • < 160 spm: Braking forces are higher. Try to shorten stride.
"""
        textbox = ctk.CTkTextbox(self, font=("Consolas", 14), width=500, height=500)
        textbox.pack(padx=20, pady=10)
        textbox.insert("0.0", text)
        textbox.configure(state="disabled") 
        
        btn = ctk.CTkButton(self, text="Got it!", command=self.destroy, fg_color="#2CC985", text_color="black")
        btn.pack(pady=20)

class GarminAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Garmin FIT Analyzer v2.2")
        self.geometry("1100x850")

        self.run_data = []
        self.df = None

        # --- LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="Garmin\nAnalyzer 🏃‍♂️", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Select Folder", command=self.select_folder)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)

        self.btn_csv = ctk.CTkButton(self.sidebar, text="💾 Export CSV", command=self.save_csv, state="disabled")
        self.btn_csv.grid(row=2, column=0, padx=20, pady=10)

        self.btn_copy = ctk.CTkButton(self.sidebar, text="📋 Copy for LLM", command=self.copy_to_clipboard, state="disabled", fg_color="#2CC985", text_color="black")
        self.btn_copy.grid(row=3, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray")
        self.status_label.grid(row=6, column=0, padx=20, pady=20)
        
        # --- LOADING HUD (Real Progress Bar) ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="#1F1F1F", border_width=2, border_color="#2CC985", corner_radius=15, width=350, height=80)
        self.progress_frame.grid_propagate(False) 
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="ANALYZING FIT FILES...", font=("Arial", 14, "bold"), text_color="white")
        self.progress_label.place(relx=0.5, rely=0.3, anchor="center")
        
        self.progress = ctk.CTkProgressBar(self.progress_frame, mode="determinate", width=250, height=12, progress_color="#2CC985")
        self.progress.set(0)
        self.progress.place(relx=0.5, rely=0.7, anchor="center")

        # 2. Main Area
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Tab 1: Report
        self.tab_report = self.tabview.add("📄 Report")
        self.textbox = ctk.CTkTextbox(self.tab_report, font=("Consolas", 14))
        self.textbox.pack(fill="both", expand=True)

        # --- WELCOME MESSAGE (CENTERED) ---
        self.textbox.tag_config("center", justify="center")
        welcome_text = "\n\n\n👋 Welcome to Garmin Analyzer!\n\n1. Click '📂 Select Folder' on the left.\n2. Choose your folder of .FIT files.\n3. Watch the magic happen."
        self.textbox.insert("0.0", welcome_text, "center")

        # Tab 2: Graphs
        self.tab_graph = self.tabview.add("📈 Trend Analysis")
        
        # Controls
        self.graph_controls = ctk.CTkFrame(self.tab_graph, fg_color="transparent", height=30)
        self.graph_controls.pack(fill="x", padx=5, pady=5)
        self.btn_info = ctk.CTkButton(self.graph_controls, text="❓ What do the dots mean?", command=self.open_guide, width=180, height=24, fg_color="#444", hover_color="#555")
        self.btn_info.pack(side="right")

        self.graph_frame = ctk.CTkFrame(self.tab_graph)
        self.graph_frame.pack(fill="both", expand=True)

    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.btn_load.configure(state="disabled")
            self.progress_frame.place(relx=0.6, rely=0.5, anchor="center")
            self.progress_frame.lift()
            self.progress.set(0) # Reset to 0
            threading.Thread(target=self.run_analysis, args=(folder_selected,), daemon=True).start()

    def run_analysis(self, folder):
        analyzer = FitAnalyzer(output_callback=self.update_log, progress_callback=self.update_progress)
        results = analyzer.analyze_folder(folder)
        self.after(0, lambda: self.display_results(results))

    def update_progress(self, current, total):
        if total > 0:
            val = current / total
            self.progress.set(val)
            self.progress_label.configure(text=f"ANALYZING... ({current}/{total})")

    def update_log(self, text):
        print(text)

    def display_results(self, results):
        self.progress_frame.place_forget()

        self.run_data = results
        self.df = pd.DataFrame(results)
        
        avg_ef = 0
        if not self.df.empty:
            self.df['date_obj'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date_obj')
            avg_ef = self.df['efficiency_factor'].mean() # Calculate folder average

        # Report
        report_text = ""
        for data in results:
            # Pass the average EF to the formatter for comparison
            report_text += self.format_run_data(data, avg_ef)
            report_text += "\n" + "="*40 + "\n"

        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", report_text)
        
        # Graphs
        self.plot_trends()

        self.status_label.configure(text=f"Analyzed {len(results)} files")
        self.btn_load.configure(state="normal")
        self.btn_csv.configure(state="normal")
        self.btn_copy.configure(state="normal")

    def format_run_data(self, d, folder_avg_ef=0):
        # 1. Device Sensor Logic
        def safe_fmt(val, unit=""):
            if val is None or str(val).lower() == "nan" or val == 0:
                return "-- (Requires Device w/ Sensor)"
            return f"{val}{unit}"

        # 2. Logic: Aerobic Decoupling (Fatigue)
        decoupling = d.get('decoupling')
        d_status = ""
        if decoupling < 5:
            d_status = " (✅ Excellent)"
        elif decoupling <= 10:
            d_status = " (⚠️ Moderate Drift)"
        else:
            d_status = " (🛑 High Fatigue)"

        # 3. Logic: Cadence (Form)
        cadence = d.get('avg_cadence')
        c_status = ""
        if cadence and cadence > 170:
            c_status = " (✅ Efficient)"
        elif cadence and cadence >= 160:
            c_status = " (👌 Good)"
        elif cadence:
            c_status = " (⚠️ Overstriding)"

        # 4. Logic: Efficiency Factor (Engine)
        ef = d.get('efficiency_factor')
        e_status = ""
        if folder_avg_ef > 0 and ef > folder_avg_ef:
            e_status = " (📈 Building Fitness)"
        elif folder_avg_ef > 0:
            e_status = " (📉 Below Average)"

        return f"""
RUN: {d.get('date')} ({d.get('filename')})
--------------------------------------------------
[1] PRIMARY STATS
    Distance:   {d.get('distance_mi')} mi
    Pace:       {d.get('pace')} /mi
    GAP:        {d.get('gap_pace')} /mi (Grade Adjusted)
    Elevation:  {d.get('elevation_ft')} ft gain

[2] EFFICIENCY & ENGINE
    Efficiency Factor (EF): {ef}{e_status} (Target: > 1.3)
    Aerobic Decoupling:     {decoupling}%{d_status}
    Avg Power:              {safe_fmt(d.get('avg_power'), " W")}

[3] INTERNAL LOAD (CONTEXT)
    Avg Heart Rate:   {d.get('avg_hr')} bpm
    Respiration Rate: {safe_fmt(d.get('avg_resp'), " brpm")}
    Avg Temperature:  {safe_fmt(d.get('avg_temp'), "°C")}

[4] FORM MECHANICS
    Cadence:         {cadence} spm{c_status}
    Vertical Ratio:  {safe_fmt(d.get('v_ratio'), "%")}
    GCT Balance:     {safe_fmt(d.get('gct_balance'), "% L/R")}
    GCT Drift:       {d.get('gct_change')} ms
"""

    def plot_trends(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        if self.df is None or self.df.empty:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        fig.patch.set_facecolor('#2b2b2b')
        
        dates = self.df['date_obj']
        
        # --- PLOT 1: EFFICIENCY (With Smart Dots) ---
        ax1.axhspan(-5, 5, color='#2CC985', alpha=0.15)
        ax1.text(dates.iloc[0], 0, " OPTIMAL STABILITY ZONE (Decoupling)", color='#2CC985', fontsize=8, va='center', fontweight='bold')
        
        line1, = ax1.plot(dates, self.df['decoupling'], color='#ff4d4d', alpha=0.5, linewidth=1, label='Decoupling % (Keep in Green Band)')
        
        ax1b = ax1.twinx()
        line2, = ax1b.plot(dates, self.df['efficiency_factor'], color='#2CC985', alpha=0.3, linestyle='--', label='Efficiency Factor (Should Trend Up)')
        
        ef_mean = self.df['efficiency_factor'].mean()
        dot_colors = []
        for index, row in self.df.iterrows():
            d = row['decoupling']
            e = row['efficiency_factor']
            if e >= ef_mean and d <= 5: dot_colors.append('#2CC985')
            elif e >= ef_mean and d > 5: dot_colors.append('#ff4d4d')
            elif e < ef_mean and d <= 5: dot_colors.append('#e6e600')
            else: dot_colors.append('black')

        ax1.scatter(dates, self.df['decoupling'], c=dot_colors, s=50, edgecolors='white', linewidth=1, zorder=5)

        ax1.set_ylabel('Decoupling (%)', color='white')
        ax1.set_title('Diagnostic Trend (Color = Run Quality)', color='white')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        ax1b.set_ylabel('Efficiency Factor', color='#2CC985')
        ax1b.tick_params(axis='y', colors='#2CC985')
        
        ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='upper left', fontsize=8, facecolor='#2b2b2b', labelcolor='white')

        # --- PLOT 2: FORM (With Cadence Band) ---
        if self.df['v_ratio'].max() > 0:
            ax2.plot(dates, self.df['v_ratio'], marker='^', color='#4d94ff', label='Vertical Ratio')
            ax2.set_ylabel('Vertical Ratio (%)', color='white')
            ax2.set_title('Form Efficiency', color='white')
        else:
            ax2.axhspan(170, 180, color='#2CC985', alpha=0.15, label='Goal Zone (170-180)')
            ax2.plot(dates, self.df['avg_cadence'], marker='o', color='#ffa31a', label='Cadence')
            ax2.set_ylabel('Cadence (spm)', color='white')
            ax2.set_title('Cadence Trend', color='white')

        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        ax2.legend(loc='upper left', fontsize=8, facecolor='#2b2b2b', labelcolor='white')
        plt.xticks(rotation=45)

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def save_csv(self):
        if self.df is None: return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filename:
            cols = [
                'date', 'filename', 'distance_mi', 'pace', 'gap_pace', 
                'efficiency_factor', 'decoupling', 
                'avg_hr', 'avg_resp', 'avg_temp', 
                'avg_power', 'avg_cadence', 
                'v_ratio', 'gct_balance', 'gct_change', 
                'elevation_ft', 'moving_time_min', 'rest_time_min'
            ]
            valid_cols = [c for c in cols if c in self.df.columns]
            self.df[valid_cols].to_csv(filename, index=False)
            messagebox.showinfo("Success", "CSV Exported Successfully!")

    def copy_to_clipboard(self):
        text = self.textbox.get("0.0", "end")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.btn_copy.configure(text="✅ Copied!", fg_color="white")
        self.after(2000, lambda: self.btn_copy.configure(text="📋 Copy for LLM", fg_color="#2CC985"))
        
    def open_guide(self):
        InfoModal(self)

def main():
    app = GarminAnalyzerApp()
    app.mainloop()

if __name__ == "__main__":
    main()