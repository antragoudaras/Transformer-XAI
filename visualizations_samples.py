from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import argparse
from samples.CLS2IDX import CLS2IDX

from baselines.ViT.ViT_explanation_generator import Baselines, LRP
from baselines.ViT.ViT_new import vit_base_patch16_224
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# initialize ViT pretrained
def return_visualization(original_image, method, class_index=None):

    
    if method == 'rollout':
        transformer_attribution = baselines.generate_rollout(original_image.unsqueeze(0).cuda(), start_layer=1).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14).data.cpu()

    elif method == 'gradient_rollout_cls_spec':
        transformer_attribution = baselines.generate_gradient_att_cls_rollout(original_image.unsqueeze(0).cuda(), start_layer=1, index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14).data.cpu()
    elif method == 'full_lrp':
        transformer_attribution = orig_lrp.generate_LRP(original_image.unsqueeze(0).cuda(), method="full", index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 224, 224).data.cpu()
    
    elif method == 'transformer_attribution':
        transformer_attribution = lrp.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14).data.cpu()

    elif method == 'lrp_last_layer':
        transformer_attribution = orig_lrp.generate_LRP(original_image.unsqueeze(0).cuda(), method="last_layer",  index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14).data.cpu()
    elif method == 'attn_last_layer':
        transformer_attribution = orig_lrp.generate_LRP(original_image.unsqueeze(0).cuda(), index=class_index, method="last_layer_attn").detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14).data.cpu()
    elif method == 'attn_gradcam':
        transformer_attribution = baselines.generate_cam_attn(original_image.unsqueeze(0).cuda(), index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14).data.cpu()
    
    if method != 'full_lrp':
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear').numpy()

    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    transformer_attribution = transformer_attribution.reshape(transformer_attribution.shape[-2], transformer_attribution.shape[-1])

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def print_top_classes(predictions, method, **kwargs):    
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])
    
    print(f'Top 5 classes: for method: {method}')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)

if __name__ == "__main__":
    methods=["rollout", "gradient_rollout_cls_spec", "transformer_attribution", "lrp_last_layer", "attn_last_layer", "attn_gradcam", "full_lrp"]

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    model = vit_base_patch16_224(pretrained=True).cuda()
    model.eval()
    baselines = Baselines(model)

    # LRP
    model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)

    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])



    image = Image.open('samples/catdog.png')
    dog_cat_image = transform(image)

    cat_specific_sal_maps = []
    dog_specific_sal_maps = []

    for method in methods:
        print(f'--------------------- Examining Method: {method} ----------------------')    
        if method == 'rollout' or method == 'gradient_rollout_cls_spec' or method == 'attn_gradcam':
            output = model(dog_cat_image.unsqueeze(0).cuda())
            print_top_classes(output, method)

        elif method == 'full_lrp' or method == 'lrp_last_layer' or method == 'attn_last_layer':
            output = model_orig_LRP(dog_cat_image.unsqueeze(0).cuda())
            print_top_classes(output, method)
        
        elif method == 'transformer_attribution':
            output = model_LRP(dog_cat_image.unsqueeze(0).cuda())
            print_top_classes(output, method)

        # cat - the predicted class
        cat = return_visualization(dog_cat_image, method)
        cat_specific_sal_maps.append(cat)
        # dog 
        # generate visualization for class 243: 'bull mastiff'
        dog = return_visualization(dog_cat_image, method, class_index=243)
        dog_specific_sal_maps.append(dog)

    # Create a figure with two rows and len(methods) columns
    fig, axs = plt.subplots(2, len(methods), figsize=(20, 10))

    # Plot cat_specific_sal_maps
    for i, method in enumerate(methods):
        axs[0, i].imshow(cat_specific_sal_maps[i])
        axs[0, i].set_title(f"{method}", fontsize=12)  # Increase fontsize for better visibility
        axs[0, i].axis('off')

    # Plot dog_specific_sal_maps
    for i, method in enumerate(methods):
        axs[1, i].imshow(dog_specific_sal_maps[i])
        axs[1, i].set_title(f"{method}", fontsize=12)  # Increase fontsize for better visibility
        axs[1, i].axis('off')

    # Adjust spacing between subplots and increase spacing for titles
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save the plot as an image
    plt.savefig('dog_cat_sal_maps.png')

    image = Image.open('samples/el2.png')
    tusker_zebra_image = transform(image)

    
    tusker_specific_sal_maps = []
    zebra_specific_sal_maps = []
    for method in methods:
        print(f'--------------------- Examining Method: {method} ----------------------')
        if method == 'rollout' or method == 'attn_gradcam':
            output = model(tusker_zebra_image.unsqueeze(0).cuda())
            print_top_classes(output, method)
        if method == 'full_lrp' or method == 'lrp_last_layer' or method == 'attn_last_layer':
            output = model_orig_LRP(tusker_zebra_image.unsqueeze(0).cuda())
            print_top_classes(output, method)
        if method == 'transformer_attribution':
            output = model_LRP(tusker_zebra_image.unsqueeze(0).cuda())
            print_top_classes(output, method)
        else:
            raise ValueError(f'Unknown method: {method}')
    

        # tusker - the predicted class
        tusker = return_visualization(tusker_zebra_image, method)
        tusker_specific_sal_maps.append(tusker)

        # zebra 
        # generate visualization for class 340: 'zebra'
        zebra = return_visualization(tusker_zebra_image, method, class_index=340)
        zebra_specific_sal_maps.append(zebra)


    # image = Image.open('samples/dogbird.png')
    # dog_bird_image = transform(image)

    # dog_specific_sal_maps = []
    # bird_specific_sal_maps = []

    # for method in methods:
    #     print(f'--------------------- Examining Method: {method} ----------------------')
    #     if method == 'rollout' or method == 'attn_gradcam':
    #         output = model(dog_bird_image.unsqueeze(0).cuda())
    #         print_top_classes(output, method)
    #     if method == 'full_lrp' or method == 'lrp_last_layer' or method == 'attn_last_layer':
    #         output = model_orig_LRP(dog_bird_image.unsqueeze(0).cuda())
    #         print_top_classes(output, method)
    #     if method == 'transformer_attribution':
    #         output = model_LRP(dog_bird_image.unsqueeze(0).cuda())
    #         print_top_classes(output, method)
    #     else:
    #         raise ValueError(f'Unknown method: {method}')

    #     # basset - the predicted class
    #     basset = return_visualization(dog_bird_image, method, class_index=161)
    #     dog_specific_sal_maps.append(basset)

    #     # generate visualization for class 87: 'African grey, African gray, Psittacus erithacus (grey parrot)'
    #     parrot = return_visualization(dog_bird_image, method, class_index=87)
    #     bird_specific_sal_maps.append(parrot)
    
    # plt.figure(figsize=(10, 5))
    # for i, method in enumerate(methods):
    #     plt.subplot(2, (len(methods)+1)//2, i+1)
    #     plt.imshow(dog_specific_sal_maps[i])
    #     plt.title(f"Method: {method}")
    #     plt.axis('off')

    #     plt.subplot(2, (len(methods)+1)//2, i+1+(len(methods)+1)//2)
    #     plt.imshow(bird_specific_sal_maps[i])
    #     plt.title(f"Method: {method}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('dog_bird_sal_maps.png')