import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load model (dummy implementation for demonstration)
# In a real application, you would uncomment the next line
model = load_model("model.h5")

# Mapping class index to label
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo] if classNo < len(classes) else "Unknown Sign"

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    prediction = model.predict(img)
    classIndex = np.argmax(prediction, axis=1)[0]
    return getClassName(classIndex)

class TrafficSignApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Traffic Sign Recognition")
        self.geometry("800x650")
        self.minsize(700, 600)
        
        # Configure grid layout (3x1)
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Image display
        self.grid_rowconfigure(2, weight=0)  # Results
        self.grid_columnconfigure(0, weight=1)
        
        # Create header frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        # App title
        self.title_label = ctk.CTkLabel(
            self.header_frame, 
            text="TRAFFIC SIGN RECOGNITION",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.grid(row=0, column=0, pady=(10, 5))
        
        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame, 
            text="Upload an image of a traffic sign for classification",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.subtitle_label.grid(row=1, column=0, pady=(0, 15))
        
        # Button frame
        self.button_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.button_frame.grid(row=2, column=0, pady=(0, 10))
        
        # Upload button
        self.upload_button = ctk.CTkButton(
            self.button_frame,
            text="Choose Image",
            command=self.upload_image,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            hover_color="#2AAA8A"  # Green hover effect
        )
        self.upload_button.grid(row=0, column=0, padx=10, pady=5)
        
        # Theme toggle button
        self.theme_button = ctk.CTkButton(
            self.button_frame,
            text="Switch to Dark Mode" if ctk.get_appearance_mode() == "Light" else "Switch to Light Mode",
            command=self.toggle_theme,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#444444" if ctk.get_appearance_mode() == "Light" else "#CCCCCC",
            hover_color="#666666" if ctk.get_appearance_mode() == "Light" else "#AAAAAA",
            text_color="white" if ctk.get_appearance_mode() == "Light" else "black"
        )
        self.theme_button.grid(row=0, column=1, padx=10, pady=5)
        
        # Create image display frame
        self.image_frame = ctk.CTkFrame(self, corner_radius=10)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # Image label
        self.image_label = ctk.CTkLabel(
            self.image_frame, 
            text="No image selected",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.image_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Create results frame
        self.results_frame = ctk.CTkFrame(self, corner_radius=10)
        self.results_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Results label
        self.result_label = ctk.CTkLabel(
            self.results_frame, 
            text="Prediction will appear here",
            font=ctk.CTkFont(size=18, weight="bold"),
            wraplength=700
        )
        self.result_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(
            self, 
            text="Ready",
            font=ctk.CTkFont(size=12),
            anchor="w",
            padx=10
        )
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 5))
        
        # Set initial theme
        self.update_theme_button()
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Update status
                self.status_bar.configure(text=f"Processing: {os.path.basename(file_path)}")
                self.update()
                
                # Open and display image
                img = Image.open(file_path)
                img.thumbnail((500, 500))  # Resize for display
                
                # Convert to CTkImage
                ctk_img = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=img.size
                )
                
                # Update image label
                self.image_label.configure(image=ctk_img, text="")
                self.image_label.image = ctk_img
                
                # Make prediction
                result = predict_image(file_path)
                
                # Update result label
                self.result_label.configure(
                    text=f"Prediction: {result}",
                    text_color="#4CCD99"  # Green color for result
                )
                
                # Update status
                self.status_bar.configure(text="Ready")
                
            except Exception as e:
                self.status_bar.configure(text=f"Error: {str(e)}")
                self.result_label.configure(
                    text="Error processing image",
                    text_color="#FF6B6B"  # Red color for error
                )
    
    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        new_mode = "Dark" if current_mode == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)
        self.update_theme_button()
    
    def update_theme_button(self):
        current_mode = ctk.get_appearance_mode()
        self.theme_button.configure(
            text=f"Switch to {'Light' if current_mode == 'Dark' else 'Dark'} Mode",
            fg_color="#444444" if current_mode == "Light" else "#CCCCCC",
            hover_color="#666666" if current_mode == "Light" else "#AAAAAA",
            text_color="white" if current_mode == "Light" else "black"
        )

if __name__ == "__main__":
    app = TrafficSignApp()
    app.mainloop()