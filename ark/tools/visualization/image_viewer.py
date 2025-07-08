
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

    def __init__(self, channel_name: str = "image/sim", image_type: str = "image"):
        super().__init__("image viewer")
        self.channel_name = channel_name
        self.image_type = image_type

        # Select the message type based on the requested image_type
        msg_type = image_t
        if image_type == "rgbd":
            try:
                from arktypes import rgbd_image_t
                msg_type = rgbd_image_t
            except Exception:
                print("rgbd_image_t not available, falling back to image_t")
        elif image_type == "depth":
            try:
                from arktypes import depth_image_t
                msg_type = depth_image_t
            except Exception:
                print("depth_image_t not available, falling back to image_t")

        print(f"Listening to channel: {channel_name} (type: {image_type})")
        self.create_subscriber(channel_name, msg_type, self._image_callback)

    def _decode_image(self, msg, is_depth: bool = False):
        """Decode an image message into a numpy array."""
        img_data = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.compression_method in (
            image_t.COMPRESSION_METHOD_JPEG,
            image_t.COMPRESSION_METHOD_PNG,
        ):
            img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            if img is None:
                print("Failed to decompress image")
                return None
        elif msg.compression_method == image_t.COMPRESSION_METHOD_NOT_COMPRESSED:
            try:
                nchannels = num_channels[msg.pixel_format]
            except KeyError:
                print("Unsupported pixel format")
                return None

            try:
                img = img_data.reshape((msg.height, msg.width, nchannels))
            except ValueError as e:
                print(f"Error reshaping image data: {e}")
                return None
        else:
            print("Unsupported compression method")
            return None

        if is_depth:
            depth_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

        return img
        
    def _image_callback(self, t, channel_name, msg):
        """Callback for incoming image messages."""
        if self.image_type == "rgbd":
            try:
                color_img = self._decode_image(msg.rgb, is_depth=False)
                depth_img = self._decode_image(msg.depth, is_depth=True)
                if color_img is None or depth_img is None:
                    return
                img = np.hstack([color_img, depth_img])
            except AttributeError:
                print("RGBD message missing 'rgb' or 'depth' fields")
                return
        else:
            is_depth = self.image_type == "depth"
            img = self._decode_image(msg, is_depth=is_depth)
            if img is None:
                return

        cv2.imshow(self.channel_name, img)
        cv2.waitKey(1)
        if cv2.getWindowProperty(self.channel_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed!")
            raise KeyboardInterrupt
            
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

def main():
    app()

if __name__ == '__main__':
    main()
    