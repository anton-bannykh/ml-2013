__author__ = 'valentinka'

from point import Point
import struct


def read_images(images_name, count):
    images = []
    images_bytes = []
    with open(images_name, 'rb') as f:
        file_bytes = f.read(16)
        magic_number = struct.unpack(">I", file_bytes[0:4])[0]
        length = struct.unpack(">I", file_bytes[4:8])[0]
        len = min(count, length)
        rows = struct.unpack(">I", file_bytes[8:12])[0]
        columns = struct.unpack(">I", file_bytes[12:16])[0]
        if magic_number != 2051:
            raise Exception('Wrong image format')
        for i in xrange(len * rows * columns):
            images_bytes.append(struct.unpack(">B", f.read(1))[0])
    cur = 0
    for k in xrange(len):
        img_rows = []
        for y in xrange(rows):
            img_rows.append(images_bytes[cur:cur + columns])
            cur += columns
        images.append(img_rows)
    return images


def read_labels(labels_name, count):
    labels = []
    with open(labels_name, 'rb') as f:
        file_bytes = f.read(8)
        magic_number = struct.unpack(">I", file_bytes[0:4])[0]
        length = struct.unpack(">I", file_bytes[4:8])[0]
        len = min(count, length)
        if magic_number != 2049:
            raise Exception('Wrong label format')
        for i in xrange(len):
            labels.append(struct.unpack(">B", f.read(1))[0])
    return labels


def read(images_name, labels_name, count):
    ps = []
    images = read_images(images_name, count)
    labels = read_labels(labels_name, count)
    for (i, image) in enumerate(images):
        ps.append(Point(image, labels[i]))
    return ps