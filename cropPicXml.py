import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os


class Face(object):
    def __init__(self):
        self.label = ""
        self.pos = dict()
        self.img = None


def get_pos(xml_file_path):
    """
    :param xml_file_path: xml file path
    :return: [face1,face2] x横着y竖着
    """
    ret_list = []
    tree = ET.ElementTree(file=xml_file_path)
    for elem in tree.iterfind('object'):
        new_face = Face()
        new_face.label = elem.find('name').text
        for item in elem.find('bndbox').iter():
            new_face.pos[item.tag] = item.text
        ret_list.append(new_face)

    return ret_list


def get_sub_imgs(img_file_path, xml_file_path):
    """
    :param img_file_path:img_file_path
    :param xml_file_path:xml_file_path
    :return: list of face
    """
    img_origin = cv2.imread(img_file_path)
    img_np = np.asarray(img_origin)

    faces = get_pos(xml_file_path)
    for face in faces:
        xmin = int(face.pos["xmin"])
        ymin = int(face.pos["ymin"])
        xmax = int(face.pos["xmax"])
        ymax = int(face.pos["ymax"])
        face.img = img_np[ymin:ymax, xmin: xmax]

    return faces


if __name__ == '__main__':
    count = 0
    for i in range(1536):
        file_name = "%06d" % i
        root_read_data = "."
        img_path = os.path.join(root_read_data, "pics", file_name + ".jpg")
        xml_path = os.path.join(root_read_data, "xml", file_name + ".xml")
        faces = get_sub_imgs(img_path, xml_path)

        for face in faces:
            if count > 1800:
                save_path = os.path.join("./dataset/val", face.label, "%06d.jpg" % count)
            else:
                save_path = os.path.join("./dataset/train", face.label, "%06d.jpg" % count)

            count += 1
            cv2.imwrite(save_path, face.img)
