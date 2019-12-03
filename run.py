import os
import time
from PIL import Image, ImageDraw, ImageFont

import detect


def process_img(in_path, out_path, scales=None, fnt_size=30, debug=False):
    img = Image.open(in_path)

    box = detect.detect_number_pyr(img, scales=scales, debug=debug)

    if box is not None:
        fnt = ImageFont.truetype('FreeSerif.ttf', size=fnt_size)
        draw = ImageDraw.Draw(img)
        draw.rectangle([box['x'], box['y'], box['x']+box['w'], box['y']+box['h']], outline=(0, 255, 0))
        draw.text((box['x']+box['w'], box['y']+box['h']), box['label'], fill=(0, 255, 0), font=fnt)
        del draw

    # write to stdout
    img.save(out_path)

    return box['label']


def process_imgs():
    input_path = 'input'
    output_path = 'output'
    file_names = ['1', '2', '3', '4', '5']
    file_ext = '.jpg'

    for file_name in file_names:
        print('\nprocessing file {}{}'.format(file_name, file_ext))
        process_img(os.path.join(input_path, file_name+file_ext),
                    os.path.join(output_path, file_name+'.png'),
                    scales=[.5, .4, .3, .2, .1, .05],
                    fnt_size=40,
                    debug=True)


def process_vid():
    input_path = os.path.join('input', 'video', 'frames')
    input_ext = '.png'
    output_path = os.path.join('output', 'video', 'frames')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ts = time.time()
    for i in range(1, 400):
        label = process_img(os.path.join(input_path, str(i)+input_ext),
                            os.path.join(output_path, str(i)+'.png'),
                            scales=[1., .85, .6, .5, .35, .1])
        print(i, int(time.time()-ts), label)


if __name__ == '__main__':
    process_imgs()
    # process_vid()
