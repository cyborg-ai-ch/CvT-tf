from bs4 import BeautifulSoup


def parse_file(file_path):
    with open(file_path, "r") as f:
        soup = BeautifulSoup(f, 'lxml')
    return soup


def get_label_text(xml: BeautifulSoup):
    comp = xml.find("AbstractText", {"label": "COMPARISON"}).get_text()
    ind = xml.find("AbstractText", {"label": "INDICATION"}).get_text()
    find = xml.find("AbstractText", {"label": "FINDINGS"}).get_text()
    imp = xml.find("AbstractText", {"label": "IMPRESSION"}).get_text()
    images = [i["id"] for i in xml.findAll("parentImage")]
    return comp, ind, find, imp, images

