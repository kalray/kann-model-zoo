import cv2
import numpy
from colormap import color_map


def decode_segmask(mask, cmap, nb_classes):
    r = numpy.zeros_like(mask).astype(numpy.uint8)
    g = numpy.zeros_like(mask).astype(numpy.uint8)
    b = numpy.zeros_like(mask).astype(numpy.uint8)
    for l in range(0, nb_classes):
        idx = mask == l
        r[idx] = cmap[l, 0]
        g[idx] = cmap[l, 1]
        b[idx] = cmap[l, 2]
    rgb = numpy.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image_uint8, colored_mask):
    overlay = cv2.addWeighted(image_uint8, 1., colored_mask, 0.85, 0)
    return overlay


def get_classes_from_list(classes_list):
    classes = dict()
    classes[0] = 'background'
    for lines in classes_list:
        idx, obj_class = lines.replace('\n', '').split(' ')[:2]
        classes[int(idx)] = str(obj_class)
    return classes


def get_contours(map):
    """ From Semantic segmentation map, get contour """
    b, g, r = cv2.split(map)
    cb, hb = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cg, hg = cv2.findContours(g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cr, hr = cv2.findContours(r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cb + cg + cr


def get_contour_center(contour):
    """ From Image moments, get barycenter from contours """
    cx, cy = None, None
    m = cv2.moments(contour)
    if int(m['m00']) != 0:
        cx, cy = int(m["m10"] / m['m00']), int(m["m01"] / m['m00'])
    return cx, cy


def draw_labels(image, contours, prediction_colormap, classes):
    """ Draw labels from contours """
    for c in contours:
        x, y = get_contour_center(c)
        if x is not None and y is not None and cv2.contourArea(c) > 300:
            rbg = prediction_colormap[c[0][0, 1], c[0][0, 0]]
            class_id = color_map.index(rbg.tolist())
            if class_id != 0:
                text_size, _ = cv2.getTextSize(
                    classes[class_id], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_w, text_h = text_size
                cv2.rectangle(image,
                              (x - 5, y + 5),
                              (x + text_w + 5, y - text_h - 5),
                              (200, 200, 200), -1)
                cv2.putText(image, classes[class_id], (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (100, 100, 100), 1)


def draw_contours(image, prediction_cmp):
    """ Draw contours with predictions """
    contours = get_contours(prediction_cmp)
    cv2.drawContours(image=image, contours=contours,
                     contourIdx=-1, color=(255, 255, 255),
                     thickness=1, lineType=cv2.LINE_AA)
    return contours


def annotate(img, prediction_mask, classes, verbose=False):
    cmap = numpy.array(color_map, dtype=numpy.uint8)
    # show results
    for class_id in numpy.unique(prediction_mask):
        if verbose and class_id != 0 and class_id in classes:
            print('Model: found id {} : {}'.format(
                class_id, classes[class_id]))
    # decode semantic segmentation mask
    prediction_colormap = decode_segmask(
        prediction_mask, cmap, len(classes))
    # draw overlay mask on origin image
    img_overlay = get_overlay(img, prediction_colormap)
    # draw labels and contours
    cs = draw_contours(img_overlay, prediction_colormap)
    draw_labels(img_overlay, cs, prediction_colormap, classes)
    return img_overlay


def post_process(cfg, frame, predictions, device='mppa', **kwargs):
    classes = get_classes_from_list(cfg['classes'])
    for output_name, prediction in predictions.items():
        if device == 'cpu':
            pred = prediction.astype(dtype=numpy.float32)[0]
        elif device == 'mppa':
            pred = prediction.astype(dtype=numpy.float32)
            pred = numpy.reshape(pred, cfg['output_nodes_shape'][0])
            pred = numpy.transpose(pred, (1, 0, 2, 3))[0]
        result = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        frame = annotate(frame, result, classes)
    return frame
