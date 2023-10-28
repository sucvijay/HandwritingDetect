from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

long_description = 'A package allows you to detect the text in a handwriiten photo.'

setup(
    name="HandwritingDetect",
    version='0.1.0',
    author="AstroCB -- Vijay & Deeksha",
    author_email="<suxvijay@gmail.com>",
    description="Detect Handwriting in a picture",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['opencv-python', 'pyautogui', 'pyaudio'],
    keywords=['python', 'textblob', 'tensorflow', 'opencv-python', 'numpy'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)