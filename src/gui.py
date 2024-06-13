import time
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from network_emulator import NetworkEmulator

class NetworkEmulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Emulator")
        
        # Set default window size
        self.root.geometry("600x400")

        # Create frames
        self.main_frame = tk.Frame(root)
        self.load_link_state_frame = tk.Frame(root)
        self.loading_frame = tk.Frame(root)
        self.menu_frame = tk.Frame(root)

        self.setup_main_frame()
        self.setup_load_link_state_frame()
        self.setup_loading_frame()
        self.setup_menu_frame()

        self.main_frame.pack(fill='both', expand=True)
        
        self.net_emu = None

    def setup_main_frame(self):
        # Configure the grid to center the content
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Create Load Saved Network button
        self.load_network_button = tk.Button(self.main_frame, text="Load Saved Network", command=self.load_saved_network)
        self.load_network_button.grid(row=0, column=0, padx=10, pady=20)

        # Create Load Link State Files button
        self.load_link_state_button = tk.Button(self.main_frame, text="Load Link State Files", command=self.show_load_link_state_frame)
        self.load_link_state_button.grid(row=0, column=1, padx=10, pady=20)

    def setup_load_link_state_frame(self):
        # Create a back button
        self.back_button = tk.Button(self.load_link_state_frame, text="Back", command=self.show_main_frame)
        self.back_button.pack(anchor='nw', padx=10, pady=10)

        # Add more widgets to the load_network_frame as needed
        self.label = tk.Label(self.load_link_state_frame, text="Load Network Screen")
        self.label.pack(pady=20)
        
    def setup_loading_frame(self):
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.label = tk.Label(self.loading_frame, text="Loading...")
        self.label.grid(row=0, column=0)
        self.label.pack(pady=20)
    
    def setup_menu_frame(self):
        # Create a back button
        self.back_button = tk.Button(self.menu_frame, text="Back", command=self.show_main_frame)
        self.back_button.pack(anchor='nw', padx=10, pady=10)

        # Add more widgets to the load_network_frame as needed
        self.label = tk.Label(self.menu_frame, text="Menu Screen")
        self.label.pack(pady=20)

    def show_main_frame(self):
        self.loading_frame.pack_forget()
        self.main_frame.pack(fill='both', expand=True)

    def show_loading_frame(self):
        self.main_frame.pack_forget()
        self.loading_frame.pack(fill='both', expand=True)
        
    def show_load_link_state_frame(self):
        self.main_frame.pack_forget()
        self.setup_load_link_state_frame.pack(fill='both', expand=True)
        
    def show_menu_frame(self):
        self.loading_frame.pack_forget()
        self.menu_frame.pack(fill='both', expand=True)

    def load_saved_network(self):
        # Function to load a saved network
        file_paths = filedialog.askdirectory(title="Select Link State Files", mustexist=True)
        if file_paths:
            self.show_loading_frame()
            time.sleep(1)
            self.net_emu = NetworkEmulator(node_file=None, link_file=None, generation_rate=20, num_generation=1, 
                                      load_folder=file_paths, save_folder=None, treshold=5)
            self.net_emu.build()
            self.net_emu.start()
            self.show_menu_frame()

    def load_link_state_files(self):
        # Function to load link state files
        file_paths = filedialog.askdirectory(title="Select Link State Files", mustexist=True)
        if file_paths:
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkEmulatorGUI(root)
    root.mainloop()
