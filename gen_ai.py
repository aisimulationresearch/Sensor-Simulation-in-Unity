import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
from transformers import BlipProcessor, BlipForConditionalGeneration


class gen_text(Node):
    def __init__(self):
        super().__init__('obj_detector')
        # Load a model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'emg_img',  # Replace with the actual topic name
            self.listener_callback,
            10)

    def listener_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV format
            print('Converting image to cv2')
            image_cv2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            image_pil = PILImage.fromarray(image_rgb)

            text = "a photography of"
            inputs = self.processor(image_pil, text, return_tensors="pt").to("cuda")

            out = self.model.generate(**inputs)
            print(self.processor.decode(out[0], skip_special_tokens=True))

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)

    detector = gen_text()

    rclpy.spin(detector)

    # Destroy the node explicitly
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
