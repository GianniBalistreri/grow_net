import setuptools
import subprocess
import sys

# Install jupyter notebook extensions for using EasyExplore_examples.ipynb more conveniently:
subprocess.run(['python{} -m pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install'.format('3' if sys.platform.find('win') != 0 else '')], shell=True)

with open('README.md', 'r') as _read_me:
    long_description = _read_me.read()

with open('requirements.txt', 'r') as _requirements:
    requires = _requirements.read()

requires = [r.strip() for r in requires.split('\n') if ((r.strip()[0] != "#") and (len(r.strip()) > 3) and "-e git://" not in r)]

setuptools.setup(
    name='grow_net',
    version='0.0.1',
    author='Gianni Francesco Balistreri',
    author_email='gbalistreri@gmx.de',
    description='Gradient Boosting Neural Network',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='machine-learning deep-learning pytorch',
    license='GNU',
    url='https://github.com/GianniBalistreri/grow_net',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'grow_net': ['LICENSE',
                               'README.md',
                               'requirements.txt',
                               'setup.py'
                               ]
                  },
    data_file=[('test', [

    ]
                )],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        #'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=requires
)
