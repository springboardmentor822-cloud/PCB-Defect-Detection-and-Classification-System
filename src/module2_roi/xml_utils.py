import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")

        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)

        boxes.append((name, xmin, ymin, xmax, ymax))

    return boxes



def extract_rois_from_xml(xml_path):
    """
    Returns: list of (label, xmin, ymin, xmax, ymax)
    """
    return parse_xml(xml_path)
