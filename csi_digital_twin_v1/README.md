# Automated Defect Detection via Synthetic Digital Twins

This system identifies defects of various types in vials placed in the pipeline.  Issues listed as having been identified are having no placed vials in a position, missing or improperly placed (crooked) caps on a positioned vial, and damaged vials.  Results are displayed with a label of variables provided in the initial setup in directions.  
To get the results and examples of possible outputs for the physical system, the Digital Twin was used to generate a set of example data that is possible from the pipeline.  I did experience a difficulty in the simulator of generating a realisticly accurate result for the defect of a damaged vial, but utilized the results I was having generated to train the identification system.  Using a system named RoboFlow, each of the possible variations were labeled manually for a set of generated images.  The model used these to be trained and validate the results of any generated image.

Below are a set of images for the results of the detection system.  Each vial has a label of being passed or identifying the type of defect it has:

![Result 1](inference_results/test_batch_005_0.png)

![Result 2](inference_results/test_batch_005_4.png)

![Result 3](inference_results/test_batch_005_2.png)

![Result 4](inference_results/test_batch_005_3.png)
