# Training Code

This is an directory showing the use of Mask RCNN in our application.
We train the model to detect dynamic regions only.

## Run Jupyter notebooks
Open the `inspect_carla_data-new.ipynb` or `inspect_carla_model-new.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Work Pipeline
1. Collect simulation images from CARLA simulator.
2. Collect video clips from the real-world environment.
3. Run samples/mask_extraction/mask_extraction.py to build real-world data (images and instance masks).
4. Start training (see next step).

## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 carla.py train --dataset=/path/to/original carla/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 carla.py train --dataset=/path/to/original carla/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 carla.py train --dataset=/path/to/original carla/dataset --weights=imagenet
```

The code in `carla.py` is set to train for 15k steps (100 epochs of 150 steps each), and using a batch size of 8. 
Update the schedule to fit your needs.
