import cv2
import numpy as np

### get the mapping from index to classname 
def get_classname_mapping(classfile):
    mapping = dict()
    with open(classfile, 'r') as fin:
        lines = fin.readlines()
        for ind, line in enumerate(lines):
            mapping[ind] = line.strip()
    return mapping

### Resize image with unchanged aspect ratio using padding
def img_prepare(img, inp_dim):    
    img_w, img_h = img.shape[1], img.shape[0]
    print img_w
    print img_h
    w, h = inp_dim
    print w
    print h
    print min(1.0*w/img_w, 1.0*h/img_h)
    new_w = int(img_w * min(1.0*w/img_w, 1.0*h/img_h))
    print new_w
    new_h = int(img_h * min(1.0*w/img_w, 1.0*h/img_h))
    print new_h
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    print '#27'

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas[:,:,::-1].transpose([2,0,1]) / 255.0
    #return canvas[:,:,::-1].transpose([2,1,0]) / 255.0

### Transform the logspace offset to linear space coordinates
### and rearrange the row-wise output
### num_classes=35 is for AMTK OCR 
def predict_transform(prediction, anchors, inp_w=416, inp_h=224, num_classes=1):
    batch_size = prediction.shape[0]
    print('41 prediction.shape ='+ str(prediction.shape))
    stride =  inp_w // prediction.shape[3]
    #grid_size = inp_dim // stride     
    grid_size_w = inp_w // stride
    grid_size_h = inp_h // stride

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    print('49 bbox_attrs= ' + str(bbox_attrs))
    print('50 num_anchors= ' + str(num_anchors))
    
    prediction = np.reshape(prediction, (batch_size, bbox_attrs*num_anchors, grid_size_h*grid_size_w))
    prediction = np.swapaxes(prediction, 1, 2)
    prediction = np.reshape(prediction, (batch_size, grid_size_w*grid_size_h*num_anchors, bbox_attrs))
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    print('55 prediction.shape ='+ str(prediction.shape))
    print('56 anchors =' + str(anchors))

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = 1 / (1 + np.exp(-prediction[:,:,0]))
    prediction[:,:,1] = 1 / (1 + np.exp(-prediction[:,:,1]))
    prediction[:,:,4] = 1 / (1 + np.exp(-prediction[:,:,4]))
    
    #Add the center offsets
    grid_h = np.arange(grid_size_h)
    grid_w = np.arange(grid_size_w)
    #a,b = np.meshgrid(grid_h, grid_w)
    a,b = np.meshgrid(grid_w, grid_h)

    x_offset = a.reshape(-1,1)
    y_offset = b.reshape(-1,1)

    x_y_offset = np.concatenate((x_offset, y_offset), 1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.expand_dims(x_y_offset.reshape(-1,2), axis=0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height, width and box corner point x-y
    anchors = np.tile(anchors, (grid_size_h*grid_size_w, 1))
    anchors = np.expand_dims(anchors, axis=0)

    prediction[:,:,2:4] = np.exp(prediction[:,:,2:4])*anchors
    prediction[:,:,5: 5 + num_classes] = 1 / (1 + np.exp(-prediction[:,:, 5 : 5 + num_classes]))
    prediction[:,:,:4] *= stride

    box_corner = np.zeros(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    return prediction

### Compute intersection of union score between bounding boxes
def bbox_iou(bbox1, bbox2):
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:,0], bbox1[:,1], bbox1[:,2], bbox1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:,0], bbox2[:,1], bbox2[:,2], bbox2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=None) \
                 * np.clip(inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=None)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

### Input: the model's output dict
### Output: list of tuples in ((cx1, cy1), (cx2, cy2), cls, prob)
def rects_prepare(output, inp_dim_w=416, inp_dim_h=224 , num_classes=80):
    prediction = None
    outCount=0
    # transform prediction coordinates to correspond to pixel location
    for key, value in output.items():
        # anchor sizes are borrowed from YOLOv3 config file
        print(key)

        if key == 'layer139-conv': 
            anchors = [(116, 90), (156, 198), (373, 326)] 
            prediction0 = predict_transform(value, anchors=anchors, inp_w=inp_dim_w, inp_h=inp_dim_h)
            outCount+=1
        elif key == 'layer150-conv':
            anchors = [(30, 61), (62, 45), (59, 119)]
            prediction1 = predict_transform(value, anchors=anchors, inp_w=inp_dim_w, inp_h=inp_dim_h)
            outCount+=1
        elif key == 'layer161-conv': 
            anchors = [(10, 13), (16, 30), (33, 23)]
            prediction2 = predict_transform(value, anchors=anchors, inp_w=inp_dim_w, inp_h=inp_dim_h)
            outCount+=1
#        if prediction is None:
#            prediction = predict_transform(value, anchors=anchors)
#        else:
#            prediction = np.concatenate([prediction, predict_transform(value, anchors=anchors)], axis=1)
        if outCount==3:
            prediction = np.concatenate([prediction0,prediction1,prediction2], axis=1) 
    # confidence thresholding
    confidence = 0.9
    conf_mask = np.expand_dims((prediction[:,:,4] > confidence), axis=2)
    prediction = prediction * conf_mask
    prediction = prediction[np.nonzero(prediction[:, :, 4])]

    # rearrange results
    img_result = np.zeros((prediction.shape[0], 6))
    max_conf_cls = np.argmax(prediction[:, 5:5+num_classes], 1)
    #max_conf_score = np.amax(prediction[:, 5:5+num_classes], 1)

    img_result[:, :4] = prediction[:, :4]
    img_result[:, 4] = max_conf_cls
    img_result[:, 5] = prediction[:, 4]     
    #img_result[:, 5] = max_conf_score
    
    # non-maxima suppression
    result = []
    nms_threshold = 0.5 

    img_result = img_result[img_result[:, 5].argsort()[::-1]] 

    ind = 0
    while ind < img_result.shape[0]:
        bbox_cur = np.expand_dims(img_result[ind], 0)
        ious = bbox_iou(bbox_cur, img_result[(ind+1):])
        nms_mask = np.expand_dims(ious < nms_threshold, axis=2)
        img_result[(ind+1):] = img_result[(ind+1):] * nms_mask
        img_result = img_result[np.nonzero(img_result[:, 5])]
        ind += 1
    
    for ind in range(img_result.shape[0]):
        pt1 = [int(img_result[ind, 0]), int(img_result[ind, 1])]
        pt2 = [int(img_result[ind, 2]), int(img_result[ind, 3])]
        cls, prob = int(img_result[ind, 4]), img_result[ind, 5]
        result.append((pt1, pt2, cls, prob))

    return result

    '''
    img_classes = np.unique(img_result[:, 4])
    for cls in img_classes:
        # get predictions per class
        cls_mask = np.expand_dims((img_result[:, 4] == cls), axis=2)
        img_per_cls = img_result * cls_mask
        img_per_cls = img_per_cls[np.nonzero(img_per_cls[:, 5])]

        # descendingly sort the predictions by probability
        img_per_cls = img_per_cls[img_per_cls[:, 5].argsort()[::-1]]

        ind = 0
        while ind < img_per_cls.shape[0]:
            bbox_cur = np.expand_dims(img_per_cls[ind], 0)
            ious = bbox_iou(bbox_cur, img_per_cls[(ind+1):])
            nms_mask = np.expand_dims(ious < nms_threshold, axis=2)
            img_per_cls[(ind+1):] = img_per_cls[(ind+1):] * nms_mask
            img_per_cls = img_per_cls[np.nonzero(img_per_cls[:, 5])]
            ind += 1

        for ind in range(img_per_cls.shape[0]):
            pt1 = [int(img_per_cls[ind, 0]), int(img_per_cls[ind, 1])]
            pt2 = [int(img_per_cls[ind, 2]), int(img_per_cls[ind, 3])]
            cls, prob = int(img_per_cls[ind, 4]), img_per_cls[ind, 5]
            result.append((pt1, pt2, cls, prob))
    return result
    '''

