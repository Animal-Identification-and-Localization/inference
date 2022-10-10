from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class Detector:
    def __init__(self, model_path, labels_path, threshold=.4):
        self.model_path = model_path
        self.labels_path = labels_path
        self.threshold = threshold
        self.number_objects = 10
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(labels_path)

    def draw_boxes(self, image, objects_detected, labels):
        canvas = ImageDraw.Draw(image)
        for object in objects_detected:
            bounding_box = object.bbox
            id = object.id
            score = object.score
            x0 = bounding_box.xmin
            x1 = bounding_box.xmax
            y0 = bounding_box.ymin
            y1 = bounding_box.ymax
            canvas.rectangle([(x0, y0), (x1, y1)], outline='red')
            canvas.text((x0 + 10, y0 + 10), "%s\n%.3f" % (labels.get(id,id), score), fill="red")
    
    def get_bbox_list(self, objects_detected, labels):
        object_list = []
        for object in objects_detected:
            bounding_box = object.bbox
            id = object.id
            score = object.score
            x0 = bounding_box.xmin
            x1 = bounding_box.xmax
            y0 = bounding_box.ymin
            y1 = bounding_box.ymax
            label = labels.get(id, id)
            list_entry = [label, score, x0, x1, y0, y1]
            object_list.append(list_entry)
        return object_list

    def run_detection(self, input_image_path, output_image_path):
        #Rescaling Image
        image = Image.open(input_image_path)
        _, scale = common.set_resized_input(self.interpreter, image.size, lambda size : image.resize(size, Image.ANTIALIAS))

        #Running detection
        for _ in range(self.number_objects):
            self.interpreter.invoke()
            objects_detected = detect.get_objects(self.interpreter, self.threshold, scale)

        if not objects_detected:
            print("No objects were detected")
        else:
            for object in objects_detected:
                print(self.labels.get(object.id, object.id))

        #Drawing bounding boxes on images
        image = image.convert("RGB")
        self.draw_boxes(image, objects_detected, self.labels)
        image.save(output_image_path)

        #Printing bounding box info
        object_list = self.get_bbox_list(objects_detected, self.labels)
        return object_list