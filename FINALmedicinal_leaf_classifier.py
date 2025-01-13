import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.font import Font
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2  

# Update the path to your model file
model_path = "best_classes.h5"
if not os.path.exists(model_path):
    print("Model file not found at:", model_path)
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the file path.")
else:
    print("Model file found.")

# Load the model
model = load_model(model_path)

# Recompile the model to address warnings about metrics and optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class names
class_names = ["Painted Nettle", "Aloe Vera", "Sweet Potato", "Horseradish Tree", "Oregano", "UNCLASSIFIED"]

# Define class information
class_information = {
    "Painted Nettle": {
        "Scientific Name": "Solenostemon scutellarioides (formerly Coleus scutellarioides)",
        "Local Name": "Filipino: Mayana\nEnglish: Painted Nettle, Coleus, Flame Nettle\nBisaya: Lapunaya",
        "Description": (
            "A popular ornamental plant known for its vibrant and diverse leaf colors.\n"
            "Traditionally used in some cultures to treat various ailments, including skin conditions, respiratory issues, and digestive problems.\n"
            "Some studies suggest potential anti-inflammatory and antioxidant properties."
        ),
        "Indication": "For wounds",
        "Plant part used": "Mature leaves",
        "Method of Preparation": (
            "Infusion - Boil 1 handful of mature leaves in 3 glasses of water, wait until it boils."
        ),
        "Direction for use": "Drink the infusion 3x a day until colds is gone."
    },
    "Aloe Vera": {
        "Scientific Name": "Aloe barbadensis Miller",
        "Local Name": "Filipino: Sabila\nEnglish: Barbados aloe, Aloe Vera\nBisaya: Alo Bera",
        "Description": (
            "Succulent plant with thick, fleshy leaves that store water.\n"
            "Medicinally, Aloe Vera gel is widely used to soothe burns, cuts, and skin irritations.\n"
            "It possesses anti-inflammatory, antibacterial, and antioxidant properties.\n"
            "Internally, it's sometimes used to aid digestion and support the immune system, though more research is needed in these areas."
        ),
        "Indication": "Burns and wounds",
        "Plant part used": "Leaves and Leaf pulp",
        "Method of Preparation": "Get an adequate amount of juice.",
        "Direction for use": "Apply the juice on the affected area."
    },
    "Sweet Potato": {
        "Scientific Name": "Ipomoea Batatas",
        "Local Name": "Filipino: Kamote\nEnglish: Sweet Potato\nBisaya: Kamote",
        "Description": (
            "A starchy root vegetable with a sweet flavor.\n"
            "Grown in various colors, including orange, purple, and white.\n"
            "Rich in antioxidants, vitamins (especially vitamin A and C), and minerals.\n"
            "May help improve blood sugar control, boost immune function, and support heart health."
        ),
        "Indication": (
            "Energy-giver for mothers after giving birth. To increase milk production for mothers after delivery."
        ),
        "Plant part used": "Young leaves or camote tops",
        "Method of Preparation": "Gather young leaves about 1 handful.",
        "Direction for use": (
            "Prepare/cook the gathered camote tops into a post-partum soup or can be eaten as salad."
        )
    },
    "Horseradish Tree": {
        "Scientific Name": "Moringa oleifera",
        "Local Name": "Filipino: Malunggay\nEnglish: Horse-radish tree",
        "Description": (
            "A fast-growing, drought-resistant tree with highly nutritious leaves, pods, and seeds.\n"
            "Considered a 'miracle tree' due to its numerous health benefits.\n"
            "Leaves are rich in vitamins, minerals, antioxidants, and anti-inflammatory compounds.\n"
            "Used to support immune function, improve blood sugar control, and promote heart health."
        ),
        "Indication": "To stop bleeding wound",
        "Plant part used": "Bark",
        "Method of Preparation": "Scrape the bark of Malunggay.",
        "Direction for use": "Patch the scraped bark in the affected area."
    },
    "Oregano": {
        "Scientific Name": "Origanum vulgare",
        "Local Name": "Filipino: Kalabo, Suganda\nEnglish: Wild marjoram\nBisaya: Kalabo",
        "Description": (
            "An aromatic herb with small, oval leaves and clusters of small, white or purple flowers.\n"
            "Possesses strong antioxidant and anti-inflammatory properties.\n"
            "May help alleviate digestive issues, boost immunity, and combat infections.\n"
            "Often used to soothe coughs and colds."
        ),
        "Indication": "For cough, ubo",
        "Plant part used": "Leaves",
        "Method of Preparation": (
            "Extracted juice - Gather a handful of mature leaves. Expose the leaves on steam coming from the boiling water. Squeeze the steamed leaves to extract the juice."
        ),
        "Direction for use": (
            "Drink about 2 tbsp. of the extracted juice, continue until cough is resolved."
        )
    },
    "UNCLASSIFIED": {
        "Description": "This class does not contain any classified information."
    }
}

# Initialize the main app
class LocalMedicinalLeafClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Medicinal Leaf Classifier")
        self.root.geometry("1000x700")  # Increased window size to accommodate image
        self.root.configure(bg="#f0f5f5")
        
        # Custom fonts
        title_font = Font(family="Helvetica", size=20, weight="bold")
        button_font = Font(family="Courier New", size=14, weight="bold")

        # Title label
        self.title_label = tk.Label(root, text="Local Medicinal Leaf Classifier", font=title_font, bg="#f0f5f5", fg="#005f73")
        self.title_label.pack(pady=10)

        # Image display
        self.img_label = tk.Label(root, text="Upload or capture an image to classify", bg="#f0f5f5", fg="#0a9396")
        self.img_label.pack(pady=20)

        # Buttons
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image, font=button_font, bg="#94d2bd", fg="#001219")
        self.upload_button.pack(pady=10)

        self.camera_button = tk.Button(root, text="Camera", command=self.open_camera, font=button_font, bg="#94d2bd", fg="#001219")
        self.camera_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_image, font=button_font, bg="#94d2bd", fg="#001219")
        self.predict_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_app, font=button_font, bg="#94d2bd", fg="#001219")
        self.exit_button.pack(pady=10)

        # Prediction and confidence display
        self.prediction_label = tk.Label(root, text="", font=button_font, bg="#f0f5f5", fg="#ae2012")
        self.prediction_label.pack(pady=20)

        self.confidence_label = tk.Label(root, text="", font=button_font, bg="#f0f5f5", fg="#0a9396")
        self.confidence_label.pack(pady=20)

        # Variables to store image
        self.image_path = None

    def upload_image(self):
        # Open file dialog to choose image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            try:
                # Open and display the image
                img = Image.open(file_path)
                img.thumbnail((300, 300))  # Resize the image for display
                img_tk = ImageTk.PhotoImage(img)
                self.img_label.config(image=img_tk, text="")  # Remove default text
                self.img_label.image = img_tk
            except Exception as e:
                messagebox.showerror("Error", f"Unable to open image: {e}")
                self.image_path = None

    def open_camera(self):
        # Open the camera to capture an image
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera not accessible")
            return

        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to access the camera")
                break

            # Display the live feed in a window
            cv2.imshow("Press Space to Capture Image, Esc to Exit", frame)

            key = cv2.waitKey(1)
            if key == 27:  # Esc key to exit
                break
            elif key == 32:  # Spacebar to capture
                captured = True
                temp_path = "captured_image.jpg"
                cv2.imwrite(temp_path, frame)
                self.image_path = temp_path

                # Convert BGR to RGB for displaying in tkinter
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((300, 300))  # Resize the image for display
                img_tk = ImageTk.PhotoImage(img)
                self.img_label.config(image=img_tk, text="")  # Remove default text
                self.img_label.image = img_tk
                break

        cap.release()
        cv2.destroyAllWindows()

        if not captured:
            messagebox.showinfo("Info", "No image captured.")

    def predict_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image uploaded or captured!")
            return

        try:
            # Preprocess the image (resize and normalize)
            img = Image.open(self.image_path).resize((128, 128))  # Resize as per model requirement

            # Ensure the image is in RGB format (model might expect RGB inputs)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Convert image to numpy array and normalize the pixel values
            img_array = np.array(img)

            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (shape becomes (1, height, width, channels))

            # Make prediction
            predictions = model.predict(img_array)

            if predictions.shape[1] != len(class_names):
                messagebox.showerror("Error", f"Mismatch: Model predicts {predictions.shape[1]} classes, but {len(class_names)} class names provided.")
                return

            predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
            predicted_class_name = class_names[predicted_class_index]  # Map to class name
            confidence_score = np.max(predictions) * 100  # Get the confidence score

            if confidence_score < 70:
                predicted_class_name = "UNCLASSIFIED"

            # Display prediction and confidence
            self.prediction_label.config(text=f"Predicted Class: {predicted_class_name}")
            self.confidence_label.config(text=f"Confidence: {confidence_score:.2f}%")

            # Open a new window to display class information
            if predicted_class_name in class_information:
                self.show_class_information(predicted_class_name, class_information[predicted_class_name])
            else:
                messagebox.showinfo("Info", "No additional information available for this class.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")

    def show_class_information(self, class_name, info):
        info_window = tk.Toplevel(self.root)
        info_window.title(f"Information - {class_name}")
        info_window.geometry("500x400")
        info_window.configure(bg="#f0f5f5")

        title_font = Font(family="Helvetica", size=16, weight="bold")
        info_font = Font(family="Arial", size=12)

        # Display class name
        tk.Label(info_window, text=class_name, font=title_font, bg="#f0f5f5", fg="#005f73").pack(pady=10)

        # Display class information
        if "Description" in info:
            formatted_info = "\n\n".join([f"{key}: {value}" for key, value in info.items()])
        else:
            formatted_info = "This class does not contain any classified information."
        tk.Label(info_window, text=formatted_info, font=info_font, bg="#f0f5f5", wraplength=450, justify="left").pack(pady=10)

        # Close button
        tk.Button(info_window, text="Close", command=info_window.destroy, font=info_font, bg="#94d2bd", fg="#001219").pack(pady=20)

    def exit_app(self):
        self.root.destroy()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = LocalMedicinalLeafClassifierApp(root)
    root.mainloop()
