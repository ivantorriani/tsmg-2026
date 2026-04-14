"""Compute depth maps for images in the input folder.

Note: Chrome Remote Desktop (for next demo )
"""


import os
import glob
import torch
from ultralytics import YOLO
import utils
import cv2
import argparse
import time
import numpy as np
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

first_execution = True


SCENE_1_SHIFT = 0.00176834659


#https://astikdahal.com.np/blog/Depth-Aware-Background-Blur
def blur_background(img, depth_map, blur_ksize=21, depth_threshold=None):
    '''
    Blurs background behind depth threshold. Depth threshold 
    can be set to a constant or be constatly updated to the closest object 
    (Second may be more versatile). 
    '''
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    d_min, d_max = depth_map.min(), depth_map.max()
    if depth_threshold is None:
        depth_threshold = d_min + 0.6 * (d_max - d_min)
    background_mask = (depth_map < depth_threshold).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = 1 - background_mask
    blurred_img_bgr = cv2.blur(img_bgr, (blur_ksize, blur_ksize))
    result_bgr = np.where(foreground_mask[..., None] == 1, img_bgr, blurred_img_bgr) # Combining foreground and blurred background
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb

def solid_background(img, depth_map, bg_color=(0, 0, 0), depth_threshold=1130.5822):
    """
    Replaces background behind depth threshold with a solid color.
    img is RGB. bg_color is RGB.
    """
    d_min, d_max = depth_map.min(), depth_map.max()
    if depth_threshold is None:
        depth_threshold = d_min + 0.6 * (d_max - d_min)

    background_mask = (depth_map < depth_threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = 1 - background_mask  
    bg = np.full_like(img, bg_color, dtype=img.dtype) # full rgb bg
    result = np.where(foreground_mask[..., None] == 1, img, bg)
    return result

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction


def create_side_by_side(image, depth, grayscale):

    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max() 
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)
import numpy as np



def run(input_path, output_path, model_path, model_type="dpt_swin2_tiny_256", optimize=True, side=False, height=None,
        square=False, grayscale=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #check if gpu is connected properly. 
    if (torch.cuda.is_available()): print("GPU CONNECTED SUCCESSFULLY") 


    #load object detection model and optimize
    odModel = YOLO("weights/coco.pt") 
    odModel.to('cuda')
    odModel.fuse()
    odModel.model.half() 

    #load monocular depth model
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square) 

    #FP16 optimization
    if optimize and device.type == "cuda": 
        model = model.to(memory_format=torch.channels.last)
        model = model.half()

    # get input
    if input_path is not None:
        image_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(image_names)
    else:
        print("No input path specified. Grabbing images from camera.")

    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    if input_path is not None:
        if output_path is None:
            print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
        for index, image_name in enumerate(image_names):

            print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

            # input
            original_image_rgb = utils.read_image(image_name)  # in [0, 1]
            image = transform({"image": original_image_rgb})["image"]

            # compute
            with torch.no_grad():
                prediction = process(device, model, model_type, image, (net_w, net_h), (net_w, net_h),
                                     optimize, True)

            # output
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
                )
                if not side:
                    utils.write_depth(filename, prediction, grayscale, bits=2)
                else:
                    original_image_bgr = np.flip(original_image_rgb, 2)
                    prediction_for_display = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) #optimized 
                    content = create_side_by_side(original_image_bgr*255, prediction_for_display, grayscale)
                    cv2.imwrite(filename + ".png", content)
                utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))

    else:
        with torch.no_grad():
            fps = 30
            video = VideoStream(0).start() #connect camera
            time_start = time.time()
            frame_index = 0
            
            #video writer for output mp4
            video_writer = None

            while True:
                frame = video.read()
                if frame is not None:
                    original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
                    image = transform({"image": original_image_rgb/255})["image"]

                    #monocular depth prediction
                    prediction = process(device, model, model_type, image, (net_w, net_h),
                                         original_image_rgb.shape[1::-1], optimize, True)
                    
                    dmax_scene_1 = 1130.5822 #calibration for scene 1 (2 ft away)

                    dmax_scene_2 = 4000 #calibration for 1 foot away

                    sensitivity = 3.2 #metric for scaling
                   

                    #blurred_rgb = blur_background(original_image_rgb, prediction, blur_ksize=31, depth_threshold=dmax_scene_1)

                    blurred_rgb = solid_background(original_image_rgb, prediction, bg_color=(0, 0, 0), depth_threshold=dmax_scene_1)

                    
                    blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)


                    #object detection predictions and optimization

                    od_results = odModel(frame) # assign the object detection results

                    od_prediction = od_results[0].plot() #plot the object detection results.

                    
                    
                    '''
                    Metric Approximation:
                    Get dmax for a known distance (measure and object 2ft away and get the dmax value). 
                    Calculate SCENE_SHIFT = 2 / dmax and record that value


                    First approach to metric distance uses 2 as a baseline and accounts for growth. 
                    When the prediction drops below 0 (due to unusual growth in depth model), it can be 
                    relatively safely assumed that object is close. However this method is flawed because the 
                    same may be said when the object is far.

                    The second approach uses a standard 'inverse growth' relationship. It more correctly interprets distance
                    (especially when closer) but bottlenecks at very close and very far points. When we are out on the field, asolution could
                    be to only calculate the depth inside the box, because random pixels could very much throw off the interpretations.
                    I use https://www.sciencedirect.com/org/science/article/pii/S1546221825004655 D = 1 / ad + b (closely inspire)
                    '''

                    dmax = float(np.max(prediction)) #closest object to camera

                    metric_prediction = SCENE_1_SHIFT * dmax  #relative distance to metric distance conversion

                    refined_metric_prediction = 2.0 * (dmax_scene_1 / dmax)**sensitivity #refined prediction of distance

                    '''Metric Depth PRediction Precise 

                    if (metric_prediction > 2):
                            metric_prediction = 2 - abs(2-metric_prediction)
                    elif (metric_prediction < 2):
                            metric_prediction = 2 + abs(2-metric_prediction)

                    
                    if (metric_prediction < 0):
                        metric_prediction = "Object Very Close to Camera"
                    '''
                    
                    if (metric_prediction > 2):
                            metric_prediction = 2 - abs(2-metric_prediction)
                    elif (metric_prediction < 2):
                            metric_prediction = 2 + abs(2-metric_prediction)

                    
                    
                    if (metric_prediction < 0):
                        metric_prediction = "Object Detected Close to Camera"
                    else: 
                        metric_prediction = "No Objects Detected"
                        
                    if frame_index % 10 == 0:
                      print(f"Arbitrary prediction: {dmax}", flush=True)
                      print(f"Metric Prediction: {metric_prediction}ft" )
                      print(f"Refined Metric Prediction: { refined_metric_prediction}")
                      
                    
                    

                    # original_image_bgr = np.flip(original_image_rgb, 2) if side else None

                    original_image_bgr = blurred_bgr if side else None

                    content = create_side_by_side(original_image_bgr, prediction, grayscale)
                    
                    #Paste prediction to screen
                    cv2.putText(
                        content,                    # image
                        f"Metric Prediction 1: {refined_metric_prediction}ft, Conclusion: {metric_prediction}",                       # text
                        (50, 50),                   # bottom-left corner (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,   # font
                        .4,                        # font scale
                        (255, 255, 255),            # color (B, G, R) - white
                        2,                          # thickness
                        cv2.LINE_AA                 # line type
                        )


                    #combine both predictions in one overlay 
                    content = blurred_bgr
                    
                    if (len(od_results[0]) > 0):
                        content = cv2.addWeighted(content, 0.5, od_prediction, 0.5, 0) 

                    

                    cv2.imshow('Blurred output - Press Escape', content)

                    #content = cv2.addWeighted(content, 0.8, od_prediction, 0.8, 0)
                    

                    #remove heatmap




                    # --- init video writer ONCE, using content size ---
                    if video_writer is None:
                        h, w, _ = content.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # or 'XVID'
                        save_path = (
                            os.path.join(output_path, "output.mp4")
                            if output_path is not None else
                            "test_metric_distance_2.mp4"
                        )
                        video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
                    # -----------------------------------------------

                    # write this frame to the video file (uint8, not divided by 255)
                    video_writer.write(content)

                    cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', content/255)

                    if output_path is not None:
                        filename = os.path.join(output_path, 'Camera' + '-' + model_type + '_' + str(frame_index))
                        cv2.imwrite(filename + ".png", content)

                    alpha = 0.1
                    if time.time()-time_start > 0:
                        fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                        time_start = time.time()
                    print(f"\rFPS: {round(fps,2)}", end="")

                    if cv2.waitKey(1) == 27:  # Escape key
                        break

                    frame_index += 1
        print()


    #Write output video
    if video_writer is not None:
        video_writer.release()
    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )

    args = parser.parse_args()


    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)
