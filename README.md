# countcrowd_TFG
TFG para conteo de multitudes basado en audio e imagen. Código en Python.

`countcrowd_TFG` es un proyecto experimental que investiga la viabilidad de los sistemas ya existentes de conteo de personas con Deep Learning, incluyendo nuevas alternativas. Como objetivo principal, se estudia la posibilidad del desarrollo de un sistema de conteo de personas exclusivo por audio. Para hacer esto posible, se someten a prueba distintas arquitecturas de redes neuronales profundas, algunas enfocadas para el conteo de personas a través de imagen y otras para sonido. 

## Uso
1. Primero hay que descargar el conjunto de datos con el que funcionarán los modelos. Se propone el uso del conjunto de datos **DISCO** disponible para su descarga en el siguiente enlace: https://zenodo.org/record/3828468#.YM8JKpMzZKM.
2. Elegir la arquitectura a ejecutar (ya sea de imagen o de audio). Para el caso de una arquitectura de imagen (CSRNet, CANNet y UNet), en el archivo *datasets.py* cambiar el parámetro **PATH** por la ruta donde se encuentran las tres carpetas descargadas de *'imgs'*, *'density'* y *'auds'*. Para el caso en el que se elija una arquitectura de audio (cualquier variante de VGGish o CrisNet), se han de cambiar los parámetros **audio_path**, **train_density_path**, **val_density_path**, **test_density_path** con las correspondientes rutas donde se encuentren los archivos de audio, los mapas de densidad de entrenamiento, los mapas de densidad de validación y los mapas de densidad de prueba respectivamente.
3. Ejecutar el script para realizar el entrenamiento, validación y test de la red seleccionada.

## Cita
```
@article{countcrowd_TFG},
  title={Sistema para la estimación del aforo mediante Deep Learning},
  author={Cristina Reyes Daneri},
  organization={Universidad de Málaga},
  year={2021}
}
``` 


