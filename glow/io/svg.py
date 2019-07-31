__all__ = ('Svg', )

from pathlib import Path
from typing import Iterable, Tuple, List
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, tostring

import cv2
import numpy as np

_SVG_NS = 'http://www.w3.org/2000/svg'
_SCRIPT = """\
const path = location.pathname.split("/").pop();
const url = path.substr(0, path.lastIndexOf(".")) + ".jpg";

let svg = document.documentElement;
svg.getElementsByTagName("image")[0].setAttribute("href", url);

let i = new Image();
i.onload = () => {
    svg.setAttribute("height", i.height);
    svg.setAttribute("width", i.width);
};
i.src = url;

for (let group of svg.getElementsByTagName("g")) {
    group.setAttribute("fill", "none");
};
"""

def hsv_colors(count):
    for hue in np.linspace(0, 360, num=count, endpoint=False, dtype='int32'):
        yield f'hsl({hue},100%,50%)'


def indent(elem, level=0):
    """Fixes indentation in ElementTree"""
    i = '\n' + level * '  '
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + '  '
        for e in elem:
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if not elem.tail or not elem.tail.strip():
        elem.tail = i


class Svg:
    """
    Converts raster `mask` (2d numpy.ndarray of integers) to SVG-file

    `classes` - list of all class names (except for 0th class),
    whose could appear in `mask`,
    or names of `set.union(set(np.unique(m)) for m in masks)`.

    Usage:
        >>> mask = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
        >>> Svg(mask, ['neg', 'pos']).save('sample.svg')

    """

    def __init__(self, mask: np.ndarray, classes: List[str]):
        root = Element('svg', xmlns=_SVG_NS)
        root.append(Element('image'))

        for uniq in np.unique(mask.ravel()):
            if uniq == 0:  # skip background
                continue
            *_, contours, _2 = cv2.findContours(
                (mask == uniq).astype('u1'),
                cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
            )
            group = Element('g', {'class': classes[uniq - 1]})
            group.extend(
                Element('polygon', points=' '.join(map(str, contour.ravel())))
                for contour in contours
                if len(contour) >= 3
            )
            root.append(group)

        root.append(Element('script', href='main.js'))
        style = Element('style')
        style.text = '@import url(main.css)'
        root.append(style)
        indent(root)
        self.body = tostring(root, encoding='unicode')

        pattern = f'{{:{max(len(c) for c in classes)}s}}'
        self.classes = [pattern.format(c) for c in classes]

    def save(self, path: Path) -> None:
        path = Path(path)

        script = path.parent / 'main.js'
        if not script.exists():
            script.write_text(_SCRIPT)

        style = path.parent / 'main.css'
        if not style.exists():
            labels = zip(hsv_colors(len(self.classes)), self.classes)
            style.write_text(
                '\n'.join(
                    f'.{name} {{ stroke: {color} }}' for color, name in labels
                )
            )

        path.with_suffix('.svg').write_text(self.body)

    @staticmethod
    def load(path: Path) -> Iterable[Tuple[str, List[np.ndarray]]]:
        """
        Yields contours, contour is 2d numpy array of shape [count, (x, y)]
        """
        path = Path(path).with_suffix('.svg')
        root = ElementTree.parse(str(path)).getroot()
        for group in root.iter(tag=f'{{{_SVG_NS}}}g'):
            points = (polygon.attrib['points'].split(' ') for polygon in group)
            contours = [
                np.array([int(p) for p in pset], dtype='int32').reshape(-1, 2)
                for pset in points
            ]
            yield group.attrib['class'], contours
