from distutils.core import setup

setup(
    name='CTL',
    version='0.1.0',
    author='R. Cao',
    author_email='akks.crx@gmail.com',
    packages=['CTL', 'CTL.tests'],
    scripts=['bin/example.py','bin/hotrg-gilt-example.py'],
    license='LICENSE.txt',
    description='Tensor Network Library.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.20.0",
    ],
)