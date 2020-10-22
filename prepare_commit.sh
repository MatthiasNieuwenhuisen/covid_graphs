#!/bin/bash

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace covid19_incidence.ipynb
jupyter nbconvert --to script covid19_incidence.ipynb
