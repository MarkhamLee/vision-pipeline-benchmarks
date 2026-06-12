## Pipeline Configurations

This folder contains sample configs you can use to run benchmarks. Currently, these config files are built around YOLO models from Ultralytics, but will be expanded to include more types in the future. 

Keep in mind that you'll need an InfluxDB and Postgres instance to run these benchmarks, these are "production style" benchmarks, so you'll need to have the broader infra in place that a production pipeline would depend on. 


###  Setting up the Configs 
1) Select the models you want to use, and either put them a "models" folder in the folder for the pipeline you're running, or in the case of an off the shelf model like YOLOv8, allow the pipeline to download them for you. 
2) Input the class numbers and names 
3) Set the queue sizes, timeouts and the like or use the default values
4) Set the table ane measurement names 
5) Set the hardware label to your GPU - *Note: only NVIDIA GPU are currently supported*

