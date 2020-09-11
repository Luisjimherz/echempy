from setuptools import setup

setup(name='echempy',
      version='0.0.1',
      description='Python\'s package for numeric simulation of electrochemical systems',
      author='Luis Fernando Jimenez-Hernandez',
      author_email='luisfernandojhz@comunidad.unam.mx',
      classifiers=['Development Status :: 1-Planning',
                   'Programming Language :: Python :: 3'],
      url='https://github.com/Luisjimherz/echempy.git',
      packages=['echempy'],
      install_requires=['numpy', 'matplotlib'],
      include_package_data=True,
      zip_safe=False)
