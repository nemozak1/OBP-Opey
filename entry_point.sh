#!/bin/bash

if [ "$BUILD_OPEY_VECTORDB"  == "true"  -o "$BUILD_OPEY_VECTORDB"  == "True" ]; then
    echo "BUILD_OPEY_VECTORDB is set. Building the vector index"
    python create_vector_index.py; fastapi run app.py --port 5000
else
    echo "BUILD_OPEY_VECTORDB is not 'True' or 'true', but: '$BUILD_OPEY_VECTORDB' . Not building the vector index"
    fastapi run app.py --port 5000
fi