from setuptools import setup, find_packages

setup(
    name="blinking-mediapipe",
    version="0.1.0",
    author="Dorus Rutten",
    author_email="Dorsrutten@gmail.com",
    description="A Mediapipe-based project to track blinking and analyze eye movements.",
    url="https://github.com/Dorus-rutten/blinking-mediapipe",
    packages=find_packages(),
    install_requires=[
        "opencv-contrib-python==4.8.0.74",
        "mediapipe==0.10.14",
        "numpy==1.25.1",
        "pylsl",
        "pyxdf",
    ],
    python_requires='>=3.9.20',
)
