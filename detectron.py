from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

class detr(Node):
    def __init__(self):
        super().__init__('obj_detector')
        # Load a model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
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

            print('About to process image')
            inputs = self.processor(images=image_pil, return_tensors="pt")

            print('Detection')
            outputs = self.model(**inputs)

            # convert outputs (bounding boxes and class logits) to COCO API
            # let's only keep detections with score > 0.9
            target_sizes = torch.tensor([image_pil.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)

    detector = detr()

    rclpy.spin(detector)

    # Destroy the node explicitly
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
