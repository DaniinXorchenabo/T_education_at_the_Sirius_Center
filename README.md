# ML T-education at the Sirius Center

# Запуск
1. Склонируйте репозиторий

```bash
git clone https://github.com/DaniinXorchenabo/T_education_at_the_Sirius_Center.git
cd T_education_at_the_Sirius_Center
```

2. Скачайте или соберите образ `docker`

```bash
docker pull daniinxorchenabo/emotions-classify-env:lighting-cu122-latest
docker build -f  ./docker/lighting.Dockerfile --build-arg REQUIREMENTS_FILE=cu_12_2.txt . -t daniinxorchenabo/emotions-classify-env:lighting-cu122-latest
```

3. Создайте файл переменных среды и вставьте в него секретный ключ от ChatGPT (нужно для генерации подписей к карточкам товаров)

```bash
cp ./env/example.env ./env/.env 
```

4. Скачайте файл весов для нейронной сети и поместите скачанный файл в папку `./weights/sam/`

[sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

5. Скачайте задние фоны для вставки и сохраните их в `./backgrounds`

[backgrounds](https://disk.yandex.ru/d/7MulssjrgPeArQ)

6. Запустите образ
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 0.0.0.0:8888:8888 -p 0.0.0.0:6006:6006 --rm -it -v .:/workspace/NN  --volume  /$(pwd)/notebooks/jupyter_config:/root/.jupyter  daniinxorchenabo/emotions-classify-env:lighting-cu122-latest python src/main.py --input-folder /workspace/NN/data/raw --output-folder /workspace/NN/data/res 
```

Параметр `--input-folder` -- путь к папке, в которой лежат файлы для обработки
Параметр `--output-folder` -- путь к папке, в которую будут помещены результаты обработки
Параметр `--add-background` -- отвечает за выбор заднего фона обработанной картинки. Принимает одно значение из:
* blue
* grad
* green
* pink-white-1
* pink-white-2
* white
* или путь до файла с изображением, которое необходимо вставить на задний фон

# Выполненные работы

