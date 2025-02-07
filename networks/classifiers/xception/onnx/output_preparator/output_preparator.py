import cv2


head = "\x1b[0;30;42m"
reset = "\x1b[0;0m"
classes = None


def drawText(frame, lines, origin):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    for line in lines:
        textsize, baseline = cv2.getTextSize(line, font, scale, thick)
        origin = (origin[0], origin[1] + textsize [1] + baseline)
        cv2.rectangle (
            frame,
            (origin[0], origin[1] + baseline),
            (origin[0] + textsize [0] + baseline, origin[1] - textsize[1]),
            (255, 255, 255),
            -1)
        cv2.putText(frame, line, origin, font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def process_nn_outputs(o):
    o = o.squeeze()
    return o


def post_process(cfg, frame, nn_outputs, **kwargs):
    """
    Takes net output, draw metadata on the input image, and return the new image to draw
    """
    global classes

    for name, shape in zip (nn_outputs.keys(), cfg ['output_nodes_shape']):
        nn_outputs [name] = nn_outputs[name].reshape(shape)

    display = 3
    if classes is None:
        classes = [cl.strip() for cl in cfg ['classes'] if cl.strip() != '']
    nb_classes = len (classes)
    # analyze the result
    assert len(nn_outputs) == 1
    output = list(nn_outputs.values())[0]
    output = process_nn_outputs(output)
    sorted_indices = output.argsort()
    legend = []
    # last <display> classes of the list, starting from the end
    for i in sorted_indices[nb_classes - 1:nb_classes - 1 - display:-1]:
        legend.append("{0:0.3f} - {1}".format(output[i], classes[i]))
    drawText(frame, legend, (10, 30))
    if kwargs["dbg"]:
        print(f"{head}  >> [Post-proc] prediction: {legend[0]}{reset}")
    return frame
