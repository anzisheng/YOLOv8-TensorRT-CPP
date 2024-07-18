//./detect_object_image --model /home/an/projects/YOLOv8-TensorRT-CPP/models/yolov8n.onnx --input ../images/person.jpg //detect person
//./detect_object_image --model /home/an/projects/YOLOv8-TensorRT-CPP/models/yoloface_8n.onnx --input ../images/person.jpg  //detect face
// ./detect_object_image --model ../models/inswapper_128.onnx  --input ../images/6.jpg //swap face

#include "cmd_line_util.h"
#include "yolov8.h"
#include <iostream>
using namespace std;
using namespace cv;
// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath = "../models/yolo8n.onnx";
    std::string inputImage = "images/6.jpg";

    // Parse the command line arguments
    if (!parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
        return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, config);
    

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    //const auto objects = yoloV8.detectObjects(img);

    //borrow two vectors
    //1. 
    std::vector<float> source_face_embedding(512);
    //read the data from groundtruth
    //face_embedding_net    
    ifstream srcFile_emb("embedding.txt", ios::in); 
    if(!srcFile_emb.is_open())
    {
        cout << "cann't open embedding.txt"<<endl;
    }
    std::cout <<"embedding.txt:" << endl;
    float x; 
    for(int i = 0; i< 512/*source_face_embedding.size()*/; i++)
    {
        cout << i <<"  :";
        srcFile_emb >> x; 
        //std::cout << x <<"  ";
        source_face_embedding[i] = x;
        std::cout << source_face_embedding[i] <<"  ";
        cout << endl;        
    }
    srcFile_emb.close();
    cout << endl;

    //2. 
    std::vector<cv::Point2f> target_landmark_5(5);
    ifstream srcFile_2target("target_5.txt", ios::in); 
    if(!srcFile_2target.is_open())
    {
        cout << "cann't open the target_5.txt"<<endl;
    }

    //std::cout << "befor transform \n";
    for (int i = 0; i < 5; i++)
    {
        float x; srcFile_2target >> x; 
        float y; srcFile_2target >> y;
        //exchange this for right effect.
        //float x = pdata[i * 3] / 64.0 * 256.0;        
        //float y = pdata[i * 3 + 1] / 64.0 * 256.0;
        target_landmark_5[i] = Point2f(x, y);
        cout <<i <<": "<< target_landmark_5[i].x <<"   "<<target_landmark_5[i].y <<std::endl;
        //circle(m_srcImg, target_landmark_5[i], 3 ,Scalar(0,255,0),-1);
    }
    srcFile_2target.close();
    

    cout << "begin process" <<endl;
    const auto objects = yoloV8.process(img, source_face_embedding, target_landmark_5);


/*
    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;
*/
    return 0;
}