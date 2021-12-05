import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/gato.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['todo','gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--target_category', type=int, default=None, help='Target category')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    #model = models.resnet50(pretrained=True)
    model = models.inception_v3(pretrained=True)
    
    #model = models.vgg19(pretrained=True)    
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    #target_layers = [model.layer4[-1]] #para resnet50
    #target_layers = [model.features[-1]]
    target_layers = [model.Mixed_7c] #para inception v3


    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
 
    '''
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    '''
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    target_category = args.target_category
    
    
    
    if args.method == 'todo':
    
        # Nuevo =================
        args.method= ["gradcam","scorecam","gradcam++","xgradcam","eigengradcam","layercam"]
        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        listadoFotos = ["f1.jpg","f2.jpg","f3.jpg","g1.jpg","g2.jpg", "g3.jpg", "g4.jpg", "g5.jpg", "g6.jpg","g7.jpg","g8.jpg","g9.jpg","g10.jpg","g11.jpg","g12.jpg"]
        
        for foto in listadoFotos:
            args.image_path = "./examples/"+foto
            rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
            for metodo in args.method:
                cam_algorithm = methods[metodo]
                with cam_algorithm(model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda) as cam:

                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32

                    grayscale_cam = cam(input_tensor=input_tensor,
                                        target_category=target_category,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)

                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
                gb = gb_model(input_tensor, target_category=target_category)

                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)

                nombre = args.image_path.split('/')[-1]
                nombre_sin_extension = nombre.split('.')[0]
                

                cv2.imwrite(f'{nombre_sin_extension}_{target_category}_{metodo}_cam.jpg', cam_image)
                cv2.imwrite(f'{nombre_sin_extension}_{target_category}_{metodo}_gb.jpg', gb)
                cv2.imwrite(f'{nombre_sin_extension}_{target_category}_{metodo}_cam_gb.jpg', cam_gb)   

        # Fin nuevo =============================================================  #  
    else:
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(input_tensor, target_category=target_category)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        nombre = args.image_path.split('/')[-1]
        nombre_sin_extension = nombre.split('.')[0]
        

        cv2.imwrite(f'{nombre_sin_extension}_{target_category}_{args.method}_cam.jpg', cam_image)
        cv2.imwrite(f'{nombre_sin_extension}_{target_category}_{args.method}_gb.jpg', gb)
        cv2.imwrite(f'{nombre_sin_extension}_{target_category}_{args.method}_cam_gb.jpg', cam_gb)
    
