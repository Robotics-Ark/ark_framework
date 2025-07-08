
from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import image_t
from arktypes.utils import unpack
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

    def __init__(self, channel_name: str = "image/sim", image_type: str = "image"):
        super().__init__("image viewer")
        self.channel_name = channel_name

        # Select the message type based on the requested image_type
        if image_type == "rgbd":
            try:
                from arktypes import rgbd_t
                msg_type = rgbd_t
                self.image_type = "rgbd"
                self.create_subscriber(channel_name, rgbd_t, self._display_image)
                print("Subscribed to rgbd_t messages")
            except Exception:
                print("rgbd_t not available, falling back to image_t")
        elif image_type == "depth":
            try:
                msg_type = image_t
                self.image_type = "depth"
                self.create_subscriber(channel_name, image_t, self._display_image)
            except Exception:
                print("depth not available, falling back to image_t")
        elif image_type == "image":
            try:
                msg_type = image_t
                self.image_type = "image"
                self.create_subscriber(channel_name, image_t, self._display_image)
            except Exception:
                print("image_t not available")
        else:
            raise ValueError(f"Unsupported image type: {image_type}")

    def _display_image(self, channel_name: str, t, msg: image_t):
        print(f"Received message on channel {channel_name} at time {t}")
        if self.image_type == "rgbd":
            image,depth = unpack.rgbd(msg)
        elif self.image_type == "depth":
            image = unpack.image(msg)
        elif self.image_type == "image":
            image = unpack.image(msg)

        print(image.shape if image is not None else "No image data received")
        # display the image
        if image is not None:
            if isinstance(image, np.ndarray):
                if image.ndim == 3:
                    # Color image
                    if image.shape[2] == 3:
                        cv2.imshow(self.channel_name, image)
                if image.ndim == 2:
                    # Grayscale image
                    cv2.imshow(self.channel_name, image)
        if depth is not None:
            if isinstance(depth, np.ndarray):
                # Display depth image
                depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow(f"{self.channel_name}_depth", depth_display)
        
        cv2.waitKey(1)

    def kill_node(self):
        cv2.destroyAllWindows()
        super().kill_node()

@app.command()
def start(
    channel: str = typer.Option(
        "image/sim", help="Channel to listen to"
    ),
    image_type: str = typer.Option(
        "image",
        help="Type of image message: image, depth, or rgbd",
    ),
):
    """Start the image viewer node."""
    main(ImageViewNode, channel, image_type)

def cli_main():
    app()

if __name__ == '__main__':
    cli_main()
    