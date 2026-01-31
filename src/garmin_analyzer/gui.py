"""
Garmin FIT File Analyzer - GUI Interface
Cross-platform desktop app using tkinter (built into Python)
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
import sys
import os
import csv

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garmin_analyzer.analyzer import FitAnalyzer


class GarminAnalyzerGUI:
    """Main GUI application for Garmin FIT file analysis."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Garmin FIT Analyzer")
        self.root.geometry("750x650")
        self.root.minsize(650, 450)
        
        # Store results for CSV export
        self.analysis_results: List[Dict[str, Any]] = []
        self.last_folder_path: Optional[str] = None
        
        # Make it look decent on both platforms
        self._setup_styles()
        self._create_widgets()
        
    def _setup_styles(self):
        """Configure GUI styles."""
        self.bg_color = "#f5f5f5"
        self.root.configure(bg=self.bg_color)
        
    def _create_widgets(self):
        """Create all GUI widgets."""
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=10)
        header_frame.pack(fill=tk.X, padx=20)
        
        title_label = tk.Label(
            header_frame,
            text="🏃 Garmin FIT File Analyzer",
            font=("Arial", 18, "bold"),
            bg=self.bg_color,
            fg="#333"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Select a folder containing .fit files to analyze",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#666"
        )
        subtitle_label.pack()
        
        # Options frame
        options_frame = tk.Frame(self.root, bg=self.bg_color, pady=5)
        options_frame.pack(fill=tk.X, padx=20)
        
        # CSV Export checkbox
        self.export_csv_var = tk.BooleanVar(value=False)
        self.export_csv_cb = tk.Checkbutton(
            options_frame,
            text="📊 Export CSV (for spreadsheets/Excel)",
            variable=self.export_csv_var,
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#333",
            activebackground=self.bg_color
        )
        self.export_csv_cb.pack(anchor=tk.W)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg=self.bg_color, pady=10)
        button_frame.pack(fill=tk.X, padx=20)
        
        self.select_btn = tk.Button(
            button_frame,
            text="📁 Select Folder",
            command=self.select_folder,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=8,
            relief=tk.FLAT,
            bd=0
        )
        self.select_btn.pack(side=tk.LEFT)
        
        # Export CSV button (disabled by default)
        self.export_btn = tk.Button(
            button_frame,
            text="💾 Save CSV",
            command=self.export_csv,
            font=("Arial", 10),
            bg="#2196F3",
            fg="white",
            padx=15,
            pady=8,
            relief=tk.FLAT,
            bd=0,
            state=tk.DISABLED
        )
        self.export_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_output,
            font=("Arial", 10),
            bg="#e0e0e0",
            fg="#333",
            padx=15,
            pady=8,
            relief=tk.FLAT,
            bd=0
        )
        clear_btn.pack(side=tk.RIGHT)
        
        # Copy All button
        copy_btn = tk.Button(
            button_frame,
            text="📋 Copy All",
            command=self.copy_all,
            font=("Arial", 10),
            bg="#e0e0e0",
            fg="#333",
            padx=15,
            pady=8,
            relief=tk.FLAT,
            bd=0
        )
        copy_btn.pack(side=tk.RIGHT, padx=5)
        
        # Progress label
        self.status_label = tk.Label(
            self.root,
            text="Ready - Select a folder to begin",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#666"
        )
        self.status_label.pack(fill=tk.X, padx=20)
        
        # Output area
        output_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            font=("Consolas", 10),
            bg="white",
            fg="#333",
            wrap=tk.WORD,
            padx=10,
            pady=10,
            relief=tk.FLAT,
            bd=1
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#e0e0e0", pady=5)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_label = tk.Label(
            footer_frame,
            text="Analyzes: GAP, HR decoupling, form, cadence, power • Copy text directly to Claude/GPT/Gemini",
            font=("Arial", 9),
            bg="#e0e0e0",
            fg="#666"
        )
        footer_label.pack()
    
    def select_folder(self):
        """Open folder selection dialog."""
        folder_selected = filedialog.askdirectory(
            title="Select Folder with FIT Files",
            mustexist=True
        )
        
        if folder_selected:
            self.clear_output()
            self.last_folder_path = folder_selected
            self._emit(f"📂 Selected: {folder_selected}\n")
            self.status_label.config(text="Analyzing...", fg="#2196F3")
            self.select_btn.config(state=tk.DISABLED)
            
            # Run analysis in separate thread to keep GUI responsive
            thread = threading.Thread(
                target=self._run_analysis,
                args=(folder_selected,)
            )
            thread.daemon = True
            thread.start()
    
    def _run_analysis(self, folder_path: str):
        """Run analysis in a thread."""
        def output_callback(text):
            self.root.after(0, lambda: self._emit(text))
        
        analyzer = FitAnalyzer(output_callback=output_callback)
        self.analysis_results = []
        
        try:
            results = analyzer.analyze_folder(folder_path)
            self.analysis_results = results
            
            # Auto-export CSV if checkbox is checked
            if self.export_csv_var.get() and results:
                self.root.after(0, lambda: self._auto_export_csv(folder_path, results))
            
            self.root.after(0, lambda: self._analysis_complete(len(results)))
        except Exception as e:
            self.root.after(0, lambda: self._emit(f"\n❌ Error: {e}"))
            self.root.after(0, lambda: self._analysis_complete(0, error=str(e)))
    
    def _auto_export_csv(self, folder_path: str, results: List[Dict[str, Any]]):
        """Auto-export CSV after analysis completes."""
        default_filename = os.path.join(folder_path, "garmin_analysis.csv")
        self._save_csv(default_filename)
    
    def _analysis_complete(self, count: int, error: Optional[str] = None):
        """Handle analysis completion."""
        self.select_btn.config(state=tk.NORMAL)
        
        if error:
            self.status_label.config(text=f"Error: {error}", fg="#f44336")
            messagebox.showerror("Error", f"Analysis failed:\n{error}")
        else:
            self.status_label.config(text=f"Complete! Analyzed {count} file(s)", fg="#4CAF50")
            
            # Enable export button if we have results
            if count > 0:
                self.export_btn.config(state=tk.NORMAL)
            else:
                self.export_btn.config(state=tk.DISABLED)
            
            if count == 0:
                messagebox.showwarning("No Files", "No .fit files found in the selected folder.")
    
    def _emit(self, text: str):
        """Add text to output area."""
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def copy_all(self):
        """Copy all output text to clipboard."""
        text = self.output_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Copied to clipboard! 📋", fg="#4CAF50")
    
    def export_csv(self):
        """Open file dialog to save CSV."""
        if not self.analysis_results:
            messagebox.showwarning("No Data", "No analysis results to export.")
            return
        
        # Suggest filename based on folder
        default_name = "garmin_analysis.csv"
        if self.last_folder_path:
            default_name = os.path.join(self.last_folder_path, default_name)
        
        filepath = filedialog.asksaveasfilename(
            title="Save CSV Export",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            self._save_csv(filepath)
    
    def _save_csv(self, filepath: str):
        """Save results to CSV file."""
        if not self.analysis_results:
            return
        
        try:
            with open(filepath, 'w', newline='') as f:
                if not self.analysis_results:
                    return
                    
                # Get all keys from first result
                fieldnames = list(self.analysis_results[0].keys())
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.analysis_results:
                    # Convert datetime objects to strings
                    row = {}
                    for key, value in result.items():
                        if hasattr(value, 'strftime'):
                            row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                        elif value is None:
                            row[key] = ''
                        else:
                            row[key] = value
                    writer.writerow(row)
            
            self._emit(f"\n💾 CSV saved: {filepath}")
            self.status_label.config(text=f"CSV saved! 💾", fg="#4CAF50")
            messagebox.showinfo("Export Complete", f"CSV saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{e}")
    
    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.analysis_results = []
        self.last_folder_path = None
        self.export_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready - Select a folder to begin", fg="#666")


def main():
    """Entry point for the GUI application."""
    root = tk.Tk()
    
    # Set app icon (works on Windows, Mac needs .icns)
    try:
        if sys.platform == "win32":
            root.iconbitmap(default=None)  # Use default icon
    except:
        pass
    
    app = GarminAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
