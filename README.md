# ML T-education at the Sirius Center

Build image

```bash
docker build -f  ./docker/lighting.Dockerfile --build-arg REQUIREMENTS_FILE=cu_12_2.txt . -t daniinxorchenabo/emotions-classify-env:lighting-cu122-latest
```

Run jupiter notebook

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 0.0.0.0:8888:8888 -p 0.0.0.0:6006:6006 --rm -it -v .:/workspace/NN  --volume  /$(pwd)/notebooks/jupyter_config:/root/.jupyter  daniinxorchenabo/emotions-classify-env:lighting-cu122-latest ./docker/before_learn.sh 
```