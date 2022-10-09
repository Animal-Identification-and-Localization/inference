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
        canvas.text((x0 + 10, y0 + 10), "%s\n%.3f"%(labels.get(id, score)), fill="red")
    

def main():
    model_path = ""
    labels_path = ""
    input_image_path = ""
    output_image_path = ""
    threshold = .4

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    labels = read_label_file(labels_path)

    image = Image.open(input_image_path)
    scale = common.set_resized_input(interpreter, image.size, lambda size : image.resize(size, Image.ANTIALIAS))

    for _ in range(args.count):
        interpreter.invoke()
        objects_detected = detect.get_objects(interpreter, threshold, scale)

    if not objects_detected:
        print("No objects were detected")
    else:
        for object in objects_detected:
            print(labels.get(object.id, object.id))

    image = image.convert("RGB")
    draw_boxes(image, objects_detected, labels)
    image.save(output_image_path)
    

if __name__ == "__main__":
    main()
