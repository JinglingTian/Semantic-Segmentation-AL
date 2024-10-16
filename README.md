# Semantic-Segmentation-AL
Applied CNN for semantic segmentation of battery CT images to create 3D models of electrode material particles and extract their image features. Conducted data mining on these features to study the structural evolution of electrode material particles during battery operation.

Applied FiJi software for image processing, applying threshold segmentation and watershed algorithm to obtain rough segmentation labels for building a dataset to train the segmentation model.

A UNet with two decoders is used to simultaneously achieve boundary prediction and foreground segmentation.

Introduced active learning to further optimize the model. Utilized information entropy to quantify the uncertainty of segmentation results, selecting the most valuable images for manual labeling to reduce the annotation workload.
