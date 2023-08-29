import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        sketch_image = create_pencil_sketch(image)
        display_images(image, sketch_image)


def create_pencil_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (21, 21), 0)
    edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=5)
    inverted_edges = 255 - edges
    pencil_sketch = cv2.divide(gray_image, inverted_edges, scale=256.0)
    return pencil_sketch


def adjust_contrast(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def display_images(original, sketch):
    original_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    sketch_image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    # Adjust contrast of the sketch image to make it bolder
    sketch_image = adjust_contrast(sketch_image, alpha=1.5, beta=10)

    original_image = Image.fromarray(original_image)
    sketch_image = Image.fromarray(sketch_image)

    original_photo = ImageTk.PhotoImage(original_image)
    sketch_photo = ImageTk.PhotoImage(sketch_image)

    original_label.config(image=original_photo)
    sketch_label.config(image=sketch_photo)

    original_label.image = original_photo
    sketch_label.image = sketch_photo


# Create a simple Tkinter window
root = tk.Tk()
root.title("Image to Pencil Sketch App")

# Create buttons for opening and processing images
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# Create labels for displaying images
original_label = tk.Label(root)
original_label.pack(side=tk.LEFT)

sketch_label = tk.Label(root)
sketch_label.pack(side=tk.RIGHT)

# Start the main Tkinter event loop
root.mainloop()
