if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero
from PIL import Image

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/' \
      'raw/images/zebra.jpg'

img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
img.show()

