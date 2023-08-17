# Comparación de técnicas de redes neuronales en grafos para la clasificación de audios

Este repositorio hace parte de un trabajo basado en redes neuronales basadas en grafos (GNN) para abordar las tareas de clasificación de audios. En particular, los redes GNN se han implementados para realizar la clasificación de los audios del conjunto de datos ```UrbanSound8k```.


3. OBJETIVOS

    3.1. General

    Comparar el desempeño de técnicas de redes neuronales en grafos para la identificación de características relevantes orientado al análisis automático de audios.

    3.2. Específicos

    - Identificar arquitecturas de redes neuronales en grafos del estado del arte para el análisis
automático de audios

    - Caracterizar datos de audios con arquitecturas pre-entrenadas para la representación de
grafos

    - Comparar los desempeños de los modelos de redes neuronales en grafos identificados
    previamente, usando como entrada los grafos creados a partir de la caracterización de
    audios.




## Instalación 

```sh
git clone
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Este trabajo tiene tres etapas:

### Primera Etapa: Generación de caracteristicas


Se generan caracteristicas profundas a partir de modelos pre-entrenados para la clasificación y caracterización de señales de audio. Los modelos pre-entrenados son: PANNs, VGGish y YAMNet. Para llevar a cabo la caracterizacion de los audios es necesario dirigirse a la carpeta llamada ``` Opt_caract```.

```sh
cd Opt_aract
```

### Segunda etapa: Construcción de Dataloaders 


En la segunda etapa, se generan los dataloader que guardaran en memoria los grafos para el entrenamiento de los modelos GNN implementados. Para generar los dataloades con los grafos de entrenamiento, se debe ingresar a la carpeta ```own_dataset```.


```sh
cd own_dataset
```

### Tercera etapa: Implementación y Entrenamiento de Modelos GNN


La tercera etapa abarca la implementación de los modelos GNN y su posterir generación de resultados. Si se desea llevar a cabo el entrenamiento de los modelos GNN, es necesario dirigirse a la carpeta ```Models_GNN```.

```sh
cd Models_GNN
```


