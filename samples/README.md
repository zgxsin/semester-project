# Core Code of the project

The directory includes:
* Model training code and relevant jupyter notebooks
* Evaluation code (F1 score) 
* Mask extraction code for real-world data
* Code of other methods for mask extracting

## Training
* See the `README.md` included in the `carla` directory.

## Evaluation (F1 score)
* Process corresponding images sets in Cityscape dataset.
* Use the trained weights in the above step and run `evaluate_model_F1.py` to calculate the F1 score.

## Inference
* Run `demo_carla_zurich_prediction.ipynb` to predict on images or videos. You can also make a video from the predicted results.

