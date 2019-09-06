import setuptools

setuptools.setup(
    name='DensityPlot',
    version='0.1.5',
    author='YSX',
    author_email='xuesoso@gmail.com',
    packages=setuptools.find_packages(),
    url='https://github.com/xuesoso/DensityPlot',
    license='LICENSE',
    description='A simple density plotting tool for FACS-like data.',
    long_description=open('README.md').read(),
    install_requires=[
       "scipy >= 0.14.0",
       "matplotlib >= 2.0.0",
   ],
)
