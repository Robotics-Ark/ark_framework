
import numpy as np
import cv2
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from arktypes import image_t


# Define num_channels for different pixel formats
num_channels = {
    image_t.PIXEL_FORMAT_RGB: 3,   # RGB has 3 channels
    image_t.PIXEL_FORMAT_RGBA: 4,  # RGBA has 4 channels
    image_t.PIXEL_FORMAT_GRAY: 1,  # Grayscale has 1 channel
}

def process_and_display_image(image_msg):
    # Decode the image data
    img_data = np.frombuffer(image_msg.data, dtype=np.uint8)

    # Handle compression
    if image_msg.compression_method in (image_t.COMPRESSION_METHOD_JPEG, image_t.COMPRESSION_METHOD_PNG):
        # Decompress image
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decompress image")
            return
    elif image_msg.compression_method == image_t.COMPRESSION_METHOD_NOT_COMPRESSED:
        # Determine the number of channels based on pixel_format
        try:
            nchannels = num_channels[image_msg.pixel_format]
        except KeyError:
            print("Unsupported pixel format")
            return

        # Reshape the data to the original image dimensions
        try:
            img = img_data.reshape((image_msg.height, image_msg.width, nchannels))
        except ValueError as e:
            print(f"Error reshaping image data: {e}")
            return

        # Handle pixel format conversion if necessary
        if image_msg.pixel_format == image_t.PIXEL_FORMAT_RGB:
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif image_msg.pixel_format == image_t.PIXEL_FORMAT_RGBA:
            # Convert RGBA to BGRA for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        elif image_msg.pixel_format == image_t.PIXEL_FORMAT_GRAY:
            # No conversion needed for grayscale
            pass
        # For BGR and BGRA, no conversion is needed as OpenCV uses BGR format
    else:
        print("Unsupported compression method")
        return

    # Now display the image dynamically in Jupyter
    clear_output(wait=True)  # Clear previous image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
    plt.axis('off')  # Hide axes
    plt.show()