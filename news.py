import torch
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, filedialog, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import platform

class FootballTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Player Tracker")
        
        # Configure root window to use full screen based on platform
        if platform.system() == "Windows":
            self.root.state('zoomed')
        elif platform.system() == "Linux":
            self.root.attributes('-zoomed', True)
        else:  # macOS and others
            self.root.attributes('-fullscreen', True)
            
        # Initialize YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)
        
        # Tracking variables
        self.player_id_map = {}
        self.player_positions = {}
        self.id_counter = 1
        self.inactive_timeout = 1.5
        self.last_seen = {}
        self.tracking_data = {}  # Store tracking data for all players
        self.cap = None
        self.is_playing = False
        self.current_frame = None
        self.frame_dims = None
        self.selected_player_id = None
        self.display_mode = "all"  # "all" or "single"
        
        # Field dimensions (in pixels)
        self.field_width = 800
        self.field_height = 600
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main layout frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel at the top
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Video display frame
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control elements
        self.load_button = ttk.Button(self.control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(self.control_frame, text="Start Tracking", command=self.toggle_tracking)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Player selection dropdown
        self.player_var = tk.StringVar()
        self.player_dropdown = ttk.Combobox(self.control_frame, textvariable=self.player_var, state='readonly')
        self.player_dropdown.pack(side=tk.LEFT, padx=5)
        self.player_dropdown.set("Select Player")
        self.player_dropdown.bind('<<ComboboxSelected>>', self.on_player_selected)
        
        # Display mode toggle
        self.display_mode_var = tk.StringVar(value="all")
        self.display_mode_button = ttk.Button(self.control_frame, text="Toggle Single Player", 
                                            command=self.toggle_display_mode)
        self.display_mode_button.pack(side=tk.LEFT, padx=5)
        
        self.generate_heatmap_button = ttk.Button(self.control_frame, text="Generate Heatmap", 
                                                 command=self.show_heatmap_window)
        self.generate_heatmap_button.pack(side=tk.LEFT, padx=5)
        
        # Exit button
        self.exit_button = ttk.Button(self.control_frame, text="Exit", command=self.root.quit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Status message
        self.status_label = ttk.Label(self.control_frame, text="Select a video to begin tracking")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Video display
        self.video_canvas = tk.Canvas(self.display_frame)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind escape key to exit fullscreen
        self.root.bind('<Escape>', lambda e: self.root.quit())
    
    def toggle_display_mode(self):
        if self.display_mode == "all":
            self.display_mode = "single"
            self.display_mode_button.config(text="Show All Players")
            if not self.selected_player_id:
                self.status_label.config(text="Please select a player to display")
        else:
            self.display_mode = "all"
            self.display_mode_button.config(text="Toggle Single Player")
    
    def show_heatmap_window(self):
        selected = self.player_var.get()
        if not selected or selected == "Select Player":
            self.status_label.config(text="Please select a player first")
            return
            
        player_id = int(selected)
        if player_id not in self.tracking_data or not self.tracking_data[player_id]:
            self.status_label.config(text=f"No tracking data available for Player {player_id}")
            return
            
        # Create new window for heatmap
        heatmap_window = Toplevel(self.root)
        heatmap_window.title(f"Player {player_id} Movement Heatmap")
        heatmap_window.geometry(f"{self.field_width}x{self.field_height}")
        
        # Generate and display heatmap in new window
        self.generate_heatmap(heatmap_window, player_id)
    
    def create_field_background(self):
        """Create a football field background image"""
        field = np.ones((self.field_height, self.field_width, 3), dtype=np.uint8) * 255
        
        # Draw green field
        cv2.rectangle(field, (0, 0), (self.field_width, self.field_height), (50, 180, 50), -1)
        
        # Draw white lines
        cv2.rectangle(field, (50, 50), (self.field_width-50, self.field_height-50), (255, 255, 255), 2)
        cv2.line(field, (self.field_width//2, 50), (self.field_width//2, self.field_height-50), (255, 255, 255), 2)
        cv2.circle(field, (self.field_width//2, self.field_height//2), 60, (255, 255, 255), 2)
        cv2.rectangle(field, (50, self.field_height//2-100), (200, self.field_height//2+100), (255, 255, 255), 2)
        cv2.rectangle(field, (self.field_width-200, self.field_height//2-100), 
                     (self.field_width-50, self.field_height//2+100), (255, 255, 255), 2)
        
        return field
    
    def generate_heatmap(self, heatmap_window, player_id):
        # Create figure with field background
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create and display field background
        field_img = self.create_field_background()
        ax.imshow(cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB))
        
        # Get positions and scale them to field dimensions
        positions = np.array(self.tracking_data[player_id])
        if len(positions) > 0:
            x = positions[:, 0] * (self.field_width / self.frame_dims[1])
            y = positions[:, 1] * (self.field_height / self.frame_dims[0])
            
            # Generate heatmap
            sns.kdeplot(x=x, y=y, cmap='hot', alpha=0.5, fill=True, ax=ax)
        
        ax.set_title(f'Player {player_id} Movement Heatmap')
        ax.axis('off')
        
        # Embed in new window
        canvas = FigureCanvasTkAgg(fig, heatmap_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_frame(self, frame):
        self.current_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Get canvas dimensions
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        # Calculate scaling factors
        img_width, img_height = img.size
        width_factor = canvas_width / img_width
        height_factor = canvas_height / img_height
        scale_factor = min(width_factor, height_factor)
        
        # Scale image while maintaining aspect ratio
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate position to center the image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.delete("all")  # Clear previous frame
        self.video_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        self.video_canvas.image = photo

    def assign_player_id(self, bbox):
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        min_dist = float('inf')
        closest_id = None
        for player_id, old_bbox in self.player_id_map.items():
            dist = self.calculate_distance(bbox, old_bbox)
            if dist < min_dist and dist < 70:
                min_dist = dist
                closest_id = player_id
        
        if closest_id is not None:
            self.last_seen[closest_id] = time.time()
            self.player_id_map[closest_id] = bbox
            return closest_id
        
        new_id = self.id_counter
        self.id_counter += 1
        self.player_id_map[new_id] = bbox
        self.last_seen[new_id] = time.time()
        self.tracking_data[new_id] = []
        self.update_player_dropdown()
        return new_id
    
    def calculate_distance(self, bbox1, bbox2):
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    
    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(file_path)
            self.is_playing = False
            self.tracking_data.clear()
            self.player_id_map.clear()
            self.player_positions.clear()
            self.id_counter = 1
            self.selected_player_id = None
            self.display_mode = "all"
            self.display_mode_button.config(text="Toggle Single Player")
            self.start_button.config(text="Start Tracking")
            self.update_player_dropdown()
            
            ret, frame = self.cap.read()
            if ret:
                self.frame_dims = frame.shape[:2]
                self.show_frame(frame)
    
    def update_player_dropdown(self):
        player_ids = list(self.tracking_data.keys())
        self.player_dropdown['values'] = player_ids
        if not player_ids:
            self.player_dropdown.set("Select Player")

    def on_player_selected(self, event):
        selected = self.player_var.get()
        if selected and selected != "Select Player":
            self.selected_player_id = int(selected)
            if self.selected_player_id not in self.tracking_data:
                self.tracking_data[self.selected_player_id] = []
            self.status_label.config(text=f"Tracking Player {self.selected_player_id}")
    
    def toggle_tracking(self):
        if self.cap is None:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.start_button.config(text="Stop Tracking")
            self.track_video()
        else:
            self.start_button.config(text="Start Tracking")
    
    def track_video(self):
        if not self.is_playing:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.is_playing = False
            self.start_button.config(text="Start Tracking")
            return
            
        results = self.model(frame)
        detected_players = results.xyxy[0]
        
        # Create a clean frame for single player display
        clean_frame = frame.copy()
        
        current_bboxes = []
        for player in detected_players:
            bbox = player[:4].tolist()
            player_id = self.assign_player_id(bbox)
            
            if player_id is not None:
                # Only draw on frame if we're showing all players or this is the selected player
                should_draw = (self.display_mode == "all" or 
                             (self.display_mode == "single" and player_id == self.selected_player_id))
                
                if should_draw:
                    # Draw bounding box
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {player_id}', 
                              (int(bbox[0]), int(bbox[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                current_bboxes.append(player_id)# Track position for all players
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                if player_id not in self.tracking_data:
                    self.tracking_data[player_id] = []
                self.tracking_data[player_id].append((center_x, center_y))

                # If in single player mode and this is our selected player,
                # create a zoomed view centered on the player
                if (self.display_mode == "single" and 
                    player_id == self.selected_player_id):
                    # Calculate zoom box dimensions
                    box_width = int((bbox[2] - bbox[0]) * 2)  # Make box 2x the player width
                    box_height = int((bbox[3] - bbox[1]) * 2)  # Make box 2x the player height
                    
                    # Ensure minimum size
                    box_width = max(box_width, 200)
                    box_height = max(box_height, 200)
                    
                    # Calculate center point
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    
                    # Calculate zoom box coordinates
                    x1 = max(0, center_x - box_width // 2)
                    y1 = max(0, center_y - box_height // 2)
                    x2 = min(frame.shape[1], x1 + box_width)
                    y2 = min(frame.shape[0], y1 + box_height)
                    
                    # Adjust x1,y1 if x2,y2 hit the boundaries
                    if x2 == frame.shape[1]:
                        x1 = max(0, frame.shape[1] - box_width)
                    if y2 == frame.shape[0]:
                        y1 = max(0, frame.shape[0] - box_height)
                    
                    # Create zoomed view
                    frame = clean_frame[y1:y2, x1:x2].copy()
                    
                    # Draw bounding box and ID on zoomed view
                    # Adjust coordinates for zoomed view
                    local_x1 = int(bbox[0] - x1)
                    local_y1 = int(bbox[1] - y1)
                    local_x2 = int(bbox[2] - x1)
                    local_y2 = int(bbox[3] - y1)
                    
                    cv2.rectangle(frame, (local_x1, local_y1), 
                                (local_x2, local_y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {player_id}', 
                              (local_x1, local_y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update dropdown with all tracked players
        self.update_player_dropdown()
        
        # Clean up inactive players
        current_time = time.time()
        for player_id in list(self.last_seen.keys()):
            if (player_id not in current_bboxes and 
                (current_time - self.last_seen[player_id] > self.inactive_timeout)):
                del self.player_id_map[player_id]
                del self.last_seen[player_id]
        
        # If we're in single player mode but no player is selected,
        # display a message on the frame
        if (self.display_mode == "single" and 
            (not self.selected_player_id or 
             self.selected_player_id not in current_bboxes)):
            msg = "No player selected" if not self.selected_player_id else "Selected player not visible"
            h, w = frame.shape[:2]
            font_scale = 1.5
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(msg, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Draw background rectangle
            padding = 20
            cv2.rectangle(frame, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, msg, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        self.show_frame(frame)
        self.root.after(10, self.track_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = FootballTrackerGUI(root)
    root.mainloop()