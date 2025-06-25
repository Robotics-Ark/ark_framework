
import argparse

from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import image_t

import cv2
import numpy as np
import typer

num_channels = {
    image_t.PIXEL_FORMAT_GRAY: 1,
    image_t.PIXEL_FORMAT_RGB: 3,
    image_t.PIXEL_FORMAT_BGR: 3,
    image_t.PIXEL_FORMAT_RGBA: 4,
    image_t.PIXEL_FORMAT_BGRA: 4,
}

app = typer.Typer()

class ImageViewNode(BaseNode):

    def __init__(self, channel_name: str = "image/sim"):
        super().__init__("image viewer")
        print(f"Listening to channel: {channel_name}")
        self.create_subscriber(channel_name, image_t, self._image_callback)
        
    def _image_callback(self, t, channel_name, msg):
        img_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Handle compression
        if msg.compression_method in (image_t.COMPRESSION_METHOD_JPEG, image_t.COMPRESSION_METHOD_PNG):
            # Decompress image
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                print("Failed to decompress image")
                return
        elif msg.compression_method == image_t.COMPRESSION_METHOD_NOT_COMPRESSED:
            # Determine the number of channels based on pixel_format
            try:
                nchannels = num_channels[msg.pixel_format]
            except KeyError:
                print("Unsupported pixel format")
                return

            # Reshape the data to the original image dimensions
            try:
                img = img_data.reshape((msg.height, msg.width, nchannels))
            except ValueError as e:
                print(f"Error reshaping image data: {e}")
                return

            # Handle pixel format conversion if necessary
            if msg.pixel_format == image_t.PIXEL_FORMAT_RGB:
                # Convert RGB to BGR for OpenCV
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pass
            elif msg.pixel_format == image_t.PIXEL_FORMAT_RGBA:
                # Convert RGBA to BGRA for OpenCV
                #img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                pass
            elif msg.pixel_format == image_t.PIXEL_FORMAT_GRAY:
                # No conversion needed for grayscale
                pass
            # For BGR and BGRA, no conversion is needed as OpenCV uses BGR format
        else:
            print("Unsupported compression method")
            return

        cv2.imshow(self.get_used_channel_name("image"), img)
        cv2.waitKey(1)  # Needed to update the display window
        if cv2.getWindowProperty(self.get_used_channel_name("image"), cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed!")
            raise KeyboardInterrupt
            
    def kill_node(self):
        cv2.destroyAllWindows()
        super().kill_node()

@app
def start(
    channel: str = typer.Option("image", help="The channel to listen to. Default is 'image/sim' (Note only supports RGB)."),
):
    """
    Start the image viewer node.

    Args:
        channel (str): The channel to listen to. Default is "image/sim".
    """
    main(ImageViewNode, channel)

def main():
    app()

if __name__ == '__main__':
    # args parse for channel_name
    main()
    