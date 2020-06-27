import caffe
import cv2
from utils_y4 import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser('YOLOv3')

    parser.add_argument('--prototxt', type=str, default='model/yolov4-custom_leaky.prototxt')
    #parser.add_argument('--prototxt', type=str, default='model/matthew_hi_yolov3.prototxt')
    parser.add_argument('--caffemodel', type=str, default='model/yolov4-custom_leaky.caffemodel')
    #parser.add_argument('--caffemodel', type=str, default='model/hi_yolov3.caffemodel')
    parser.add_argument('--classfile', type=str, default='model/lp.names')
    #parser.add_argument('--image', type=str, default='images/dog-cycle-car.png')
    #parser.add_argument('--image', type=str, default='images/1019.jpeg')
    parser.add_argument('--image', type=str, default='images/5978-ya416224.jpg')
    #parser.add_argument('--image', type=str, default='5978.png')
    #parser.add_argument('--image', type=str, default='images/7704579252187.jpg')
    parser.add_argument('--resolutionW', type=int, default=416)
    parser.add_argument('--resolutionH', type=int, default=224)

    return parser.parse_args()

def main():
    args = parse_args()

    model = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    img_ori = cv2.imread(args.image)
    print img_ori.shape[1]
    print img_ori.shape[0]
    #return None
    inp_dim = args.resolutionW, args.resolutionH
    #inp_dim = 416,224
    #inp_dim = 320,160
    print inp_dim
    #return None
    img = img_prepare(img_ori, inp_dim)

    #return None

    #cv2.imshow("?", img.transpose([1,2,0]))
    #cv2.waitKey()
    model.blobs['data'].data[:] = img
    output = model.forward()
    print(output)
    print('layer139-conv shape=' + str(output['layer139-conv'].shape))
    print('layer150-conv shape=' + str(output['layer150-conv'].shape))
    print('layer161-conv shape=' + str(output['layer161-conv'].shape))

    rects = rects_prepare(output, inp_dim_w=args.resolutionW, inp_dim_h=args.resolutionH , num_classes=1)
    mapping = get_classname_mapping(args.classfile)

    scaling_factor = min(1.0, 1.0*args.resolutionW / img_ori.shape[1])
    for pt1, pt2, cls, prob in rects:
        pt1[0] -= (args.resolutionW - scaling_factor*img_ori.shape[1])/2
        pt2[0] -= (args.resolutionW - scaling_factor*img_ori.shape[1])/2
        pt1[1] -= (args.resolutionH - scaling_factor*img_ori.shape[0])/2
        pt2[1] -= (args.resolutionH - scaling_factor*img_ori.shape[0])/2

        pt1[0] = np.clip(int(1.0*pt1[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt2[0] = np.clip(int(1.0*pt2[0] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt1[1] = np.clip(int(1.0*pt1[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])
        pt2[1] = np.clip(int(1.0*pt2[1] / scaling_factor), a_min=0, a_max=img_ori.shape[1])

        print("prob={}".format(prob))
        label = "{}:{:.2f}".format(mapping[cls], prob)
        color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

        cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        pt2 = pt1[0] + t_size[0] + 3, pt1[1] + t_size[1] + 4
        cv2.rectangle(img_ori, tuple(pt1), tuple(pt2), color, -1)
        cv2.putText(img_ori, label, (pt1[0], t_size[1] + 4 + pt1[1]), cv2.FONT_HERSHEY_PLAIN,
                    cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)
    #cv2.imshow(args.image, img_ori)
    cv2.imwrite("y4_out.png", img_ori)
    #cv2.waitKey()


if __name__ == '__main__':
    main()
