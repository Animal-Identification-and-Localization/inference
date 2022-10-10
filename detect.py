from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

def draw_boxes(image, objects_detected, labels):
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


def main():
    #Setting variables
    model_path = "../model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    labels_path = "../model/coco_labels.txt"
    input_image_path = "../images/cats2.jpg"
    output_image_path = "../images/output2.jpg"
    threshold = .4
    number_objects = 10

    #Creating Model and Allocating Tensors
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    labels = read_label_file(labels_path)

    #Rescaling Image
    image = Image.open(input_image_path)
    _, scale = common.set_resized_input(interpreter, image.size, lambda size : image.resize(size, Image.ANTIALIAS))

    #Running detection
    for _ in range(number_objects):
        interpreter.invoke()
        objects_detected = detect.get_objects(interpreter, threshold, scale)

    if not objects_detected:
        print("No objects were detected")
    else:
        for object in objects_detected:
            print(labels.get(object.id, object.id))

    #Drawing bounding boxes on images
    image = image.convert("RGB")
    draw_boxes(image, objects_detected, labels)
    image.save(output_image_path)
    image.show()
    

if __name__ == "__main__":
    main()
