import glob
from natsort import natsorted
import xml.etree.ElementTree as ET
import re
import numpy as np

from bert_serving.client import BertClient
bc = BertClient()

xml_path = "/root/workspace/gym/gym/envs/mujoco/assets/sim_push_xmls/*"
xmls = natsorted(glob.glob(xml_path))

# get objects' names and transfer them into vector and save.
for xml in xmls:
    if xml[-4:] != ".xml": continue
    
    tree = ET.parse(xml)
    root = tree.getroot()

    xml_name = xml.split('/')[-1][:-4]
    # print(xml_name)
    object_name = []
    for child in root:
        if child.tag == "asset":
            for i in child:
                if i.tag == "mesh":
                    # print(i.tag, i.attrib)
                    # print(i.attrib["name"], i.attrib["file"])
                    name = i.attrib["file"].split('/')[-1][:-4]
                    name = name.replace("_", " ")
                    name = re.sub("\d+", "", name)
                    object_name.append(name)
                    # precosee object_name.
    # print(object_name)
    text = "There are one " + object_name[0] + " and one " + object_name[1] + "."
    text_vector = bc.encode([text])
    save_dic = "/root/xin/tecnets-pytorch/datasets/2021_there_is_a_and_b/"
    # np.save(save_dic + xml_name, text_vector)
    print("Saved ", save_dic + xml_name, text_vector.shape)