#!/bin/bash

make -j $(nproc) clean all install
pip3 install --break-system-packages -e .