from distutils.core import setup
from alpharelu import __version__

setup(name='alpharelu',
      version=__version__,
      url="https://github.com/MaxatTezekbayev/alpha-relu",
      author="Maxat Tezekbayev, Vassilina Nikoulina, Matthias Gallé, Zhenisbek Assylbekov",
      author_email="maksat013@gmail.com",
      description=("The alpha-relu mapping and its loss"),
      license="MIT",
      packages=['alpharelu'],
      install_requires=['torch>=1.0'],
      python_requires=">=3.5")