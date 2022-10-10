from detector import Detector

def main():
    model_path = "../model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    labels_path = "../model/coco_labels.txt"
    input_image_path = "../images/cats2.jpg"
    output_image_path = "../images/Doutput.jpg"

    d1 = Detector(model_path, labels_path, .4)
    d1.run_detection(input_image_path, output_image_path)

if __name__ == "__main__":
    main()