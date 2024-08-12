from setuptools import setup

setup(
   name='aTMi',
   version='0.1',
   description='aTMi: approximated transition matrix inference.',
   author='Kevin Korfmann',
   author_email='kevin.korfmann@gmail.com',
   packages=['aTMi'], 
   install_requires=['msprime', 'tszip','numpy', 'pathlib', 'torch',
                      'x_transformers', 'scipy', 'matplotlib', 'seaborn', 'tqdm', 'accelerate'], 
)
