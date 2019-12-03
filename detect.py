import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


model1 = load_model('model1.h5')
input_shape = (32, 24, 1)
stride = 6

print('Load complete...')

NON_IMG_LABEL = 10

PYR_RESCALE = [1., .75, .5, .25]


def classify_cut(cut, model=model1):
    sample = cut.astype(np.float32) / np.max(cut)
    sample = sample.reshape((1, *input_shape))

    pred = model.predict(sample)
    label = pred.argmax(axis=1)[0]
    p = np.max(pred)

    return label, p


def classify_cuts(cuts, model=model1):
    samples = cuts.astype(np.float32) / np.max(cuts, axis=0)
    samples = samples.reshape((-1, *input_shape))

    pred = model.predict(samples)
    labels = pred.argmax(axis=1)
    ps = np.max(pred, axis=1)

    return labels, ps

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return boxes

    pick = []

    x1 = np.array([box['x'] for box in boxes])
    y1 = np.array([box['y'] for box in boxes])
    x2 = np.array([box['x'] + box['w'] for box in boxes])
    y2 = np.array([box['y'] + box['h'] for box in boxes])
    p = np.array([box['p'] for box in boxes])

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(p)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    suppressed_boxes = [boxes[idx] for idx in np.sort(pick)]
    return suppressed_boxes


def euc_dist(box1, box2):
    return ((box1['x'] - box2['x'])**2 + (box1['y'] - box2['y'])**2) ** .5


def combine_number(boxes):
    boxes_sorted = sorted(boxes, key=lambda k: k['x'])

    label_comb = ''.join([box['label'] for box in boxes_sorted])

    min_x = min([b['x'] for b in boxes_sorted])
    max_x = max([b['x']+b['w'] for b in boxes_sorted])
    min_y = min([b['y'] for b in boxes_sorted])
    max_y = max([b['y']+b['h'] for b in boxes_sorted])

    max_p = max([b['p'] for b in boxes_sorted])

    return {'x': min_x, 'y': min_y,
            'w': max_x-min_x, 'h': max_y-min_y,
            'label': label_comb,
            'p': max_p
            }


def is_close(boxes, new_box, thresh):
    for box in boxes:
        if euc_dist(box, new_box) < thresh:
            return True
    return False


def detect_number(img, p_thresh=.98, to_gray=True, nms_overlap_thresh=.6, max_len=3):
    if to_gray:
        img = img.convert('L')

    img = np.array(img)

    # Identify digits
    boxes = []
    h = input_shape[0]
    w = input_shape[1]

    # for y in range(0, img.shape[0] - h, stride):
    #     for x in range(0, img.shape[1] - w, stride):
    #         cut = img[y:y + h, x:x + w]
    #         label, p = classify_cut(cut)
    #
    #         # filter
    #         if label != NON_IMG_LABEL and p > p_thresh:
    #             boxes.append({
    #                 'x': x, 'y': y,
    #                 'w': w, 'h': h,
    #                 'label': str(label),
    #                 'p': p
    #             })
    cuts = []
    coords = []
    for y in range(0, img.shape[0] - h, stride):
        for x in range(0, img.shape[1] - w, stride):
            cut = img[y:y + h, x:x + w]
            cuts.append(cut)
            coords.append((x, y))

    labels, ps = classify_cuts(np.array(cuts))

    # filter
    for i in range(len(ps)):
        if labels[i] != NON_IMG_LABEL and ps[i] > p_thresh:
            boxes.append({
                'x': coords[i][0], 'y': coords[i][1],
                'w': w, 'h': h,
                'label': str(labels[i]),
                'p': ps[i]
            })

    # Suppress overlapped digit boxes
    # suppressed_boxes = non_max_suppression(boxes, nms_overlap_thresh)
    suppressed_boxes = []
    for i in range(10):
        suppressed_boxes.extend(
            non_max_suppression(
                [box for box in boxes if box['label'] == str(i)],
                nms_overlap_thresh
            )
        )

    # Find best resulted number
    if len(suppressed_boxes) == 0:
        return None

    # best digit
    ps = np.array([box['p'] for box in suppressed_boxes])
    idx_max = ps.argmax()
    box_max = suppressed_boxes[idx_max]
    # find and combine digits that are as close to the number as the best digit box diagonal
    thresh = (box_max['w'] ** 2 + box_max['h'] ** 2) ** .5

    available_boxes = suppressed_boxes
    available_boxes.pop(idx_max)

    number_boxes = [box_max]
    found = True

    while found:
        found = False
        for i in range(len(available_boxes)):
            b = available_boxes[i]
            if is_close(number_boxes, b, thresh):
                number_boxes.append(b)
                available_boxes.pop(i)
                if len(number_boxes) < max_len:
                    found = True
                break

    number_box = combine_number(number_boxes)

    return number_box


def detect_number_pyr(img, scales=None, p_thresh=.98, to_gray=True, nms_overlap_thresh=.4, debug=False):
    if scales is None:
        scales = PYR_RESCALE

    best_box = None

    for scale in scales:
        if debug:
            print('detecting number at scale {}'.format(scale))
        img_scaled = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.BILINEAR)
        box = detect_number(img_scaled, p_thresh, to_gray, nms_overlap_thresh)
        if box is None:
            continue

        if debug:
            print('detected number {} with prob {} at scale {}'.format(box['label'], box['p'], scale))
        if best_box is None or (best_box['p'] < box['p']):
            box['x'] /= scale
            box['y'] /= scale
            box['w'] /= scale
            box['h'] /= scale
            best_box = box
    if debug:
        print('best detected number {}'.format(best_box['label']))

    return best_box

