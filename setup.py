import setuptools

setuptools.setup(
    name="emotion_recognition",
    author="Jakub Wińczuk, Kamil Uchnast",
    description="Głębokie sieci neuronowe - projekt",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "argparse",
        "pandas",
        "torch",
        "pandas-stubs",
        "matplotlib",
        "seaborn",
        "torchvision"
    ],
    entry_points={'console_scripts': ['face_recognition = main:main']}
)
