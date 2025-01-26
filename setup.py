from setuptools import setup,find_packages

setup(name='another_pytorch',
      version='0.1.0',
      license='MIT',
      install_requires=['numpy','matplotlib','graphviz'],
      extras_require={'cuda11': ['cupy-cuda11x'],'cuda12': ['cupy-cuda12x']},
      description='A simple deep learning framework inspired by Dezero and PyTorch',
      author='owenliang',
      author_email='owenliang1990@gmail.com',
      url='https://github.com/owenliang/another-pytorch',
      packages=find_packages(),
      python_requires='>=3.10',
)