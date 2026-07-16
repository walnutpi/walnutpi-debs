import setuptools
import os

# 获取README.md作为long_description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as fh:
        long_description = fh.read()
else:
    long_description = ''

setuptools.setup(
    name="walnutpi-kpu",
    version="0.1.0",
    author="WalnutPi Team",
    author_email="",  #
    description="A Python library for accessing KPU functionality on K230 chip",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WalnutPi/walnutpi.kpu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.6',
    install_requires=[
        # 依赖项
    ],

    entry_points={

    },
    include_package_data=True,
    package_data={
        # 包含.so文件
        'walnutpi_kpu': ['nncase_2_10/*.so'],
    },
    platforms=['riscv64', 'linux'],
)